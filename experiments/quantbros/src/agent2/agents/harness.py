"""LLM-driven outer harness loop for Agent 2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent2.agents.artifacts import ArtifactManager
from agent2.agents.models import (
    AgentState,
    AgenticRunFailed,
    FailureArtifact,
    HarnessAction,
    HarnessDecision,
    HarnessTraceEntry,
    Observation,
    StateSnapshot,
    TerminalRecord,
    TerminalStatus,
    ToolManifestEntry,
)
from agent2.agents.roles import (
    AnalystAgent,
    CriticAgent,
    ExtractorAgent,
    NarrativeVerifierAgent,
    RepairAgent,
    StrategyCodegenAgent,
    StrategyDebuggerAgent,
    VerifierAgent,
)
from agent2.agents.tools import ToolContext, build_tool_registry
from agent2.openrouter import OpenRouterClient, parse_json_text
from agent2.schemas import Agent1ToAgent2Input, OpenRouterCallMetadata


LOGGER = logging.getLogger(__name__)

HARNESS_SYSTEM_PROMPT = """
You are HarnesserAgent for QuantBros Agent 2.

You control the entire paper run. Nothing advances unless you explicitly choose one tool call or explicitly stop.

Architecture rules:
- There is no fixed pipeline.
- You must use observe -> reason -> act repeatedly.
- Loop on blueprint extraction/verification/repair until the blueprint is good enough.
- Loop on code generation/static scan/execution/debug until the strategy runs and passes sanity checks.
- Loop on analyst_turn/narrative_verify until the narrative is acceptable.
- Only stop with success after publish_final_artifacts has already succeeded.
- If information is irrecoverably missing, stop with failed or aborted instead of guessing.

Strategy rules:
- Do not force papers into prebuilt strategy families.
- The blueprint and generated code must stay paper-specific.
- The generated code is the strategy executor; deterministic Python outside the code only computes metrics and manages infra.
- Do not assume a generic ETF, benchmark, rebalance schedule, or parameter set unless the paper or current artifacts justify it.

Decision rules:
- Return JSON only matching HarnessDecision.
- Invoke at most one tool per turn.
- If the last tool failed, use the resulting observation and choose the next action explicitly.
- Use the available tool schemas exactly.
""".strip()


class HarnesserAgent:
    def __init__(self, *, client: OpenRouterClient | None = None) -> None:
        self.client = client or OpenRouterClient()

    def decide(self, snapshot: StateSnapshot) -> tuple[dict[str, Any], HarnessDecision | None, OpenRouterCallMetadata, str]:
        tool_lines = [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema,
            }
            for tool in snapshot.available_tools
        ]
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": HARNESS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state_snapshot": snapshot.model_dump(mode="json"),
                            "available_tools": tool_lines,
                            "response_schema": HarnessDecision.model_json_schema(),
                            "success_condition": "publish_final_artifacts succeeded and all required artifacts exist",
                        }
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        metadata = OpenRouterCallMetadata(
            model=response.model,
            latency_seconds=response.latency_seconds,
            usage=response.usage,
        )
        raw_text = response.content
        try:
            raw_json = parse_json_text(raw_text)
        except Exception as exc:  # pragma: no cover
            return {"raw_text": raw_text}, None, metadata, f"Could not parse harness JSON: {exc}"
        try:
            decision = HarnessDecision.model_validate(raw_json)
            return raw_json, decision, metadata, ""
        except ValidationError as exc:
            return raw_json, None, metadata, str(exc)


@dataclass
class HarnessRunner:
    harnesser: HarnesserAgent
    extractor: ExtractorAgent
    verifier: VerifierAgent
    critic: CriticAgent
    repair: RepairAgent
    codegen: StrategyCodegenAgent
    debugger: StrategyDebuggerAgent
    analyst: AnalystAgent
    narrative_verifier: NarrativeVerifierAgent

    def run_paper(
        self,
        agent1: Agent1ToAgent2Input,
        *,
        start: date | None = None,
        end: date | None = None,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        artifacts_dir: str | Path = "artifacts",
        run_id: str | None = None,
        max_harness_turns: int = 16,
        max_tool_calls: int = 32,
        max_wall_clock_seconds: int = 900,
        max_blueprint_attempts: int = 4,
        max_codegen_attempts: int = 5,
        max_execute_attempts: int = 5,
        max_analyst_turns: int = 3,
    ) -> TerminalRecord:
        run_id = run_id or f"{agent1.paper_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        output_dir = Path(artifacts_dir) / run_id
        artifact_manager = ArtifactManager(output_dir)
        tool_registry = build_tool_registry()
        state = AgentState(
            run_id=run_id,
            paper_id=agent1.paper_id,
            title=agent1.title,
            max_turns=max_harness_turns,
            max_tool_calls=max_tool_calls,
            max_wall_clock_seconds=max_wall_clock_seconds,
            max_blueprint_attempts=max_blueprint_attempts,
            max_codegen_attempts=max_codegen_attempts,
            max_execute_attempts=max_execute_attempts,
            max_analyst_turns=max_analyst_turns,
        )
        invalid_decisions = 0

        while state.status == TerminalStatus.running:
            snapshot = self._build_snapshot(state, tool_registry)
            raw_decision, decision, metadata, decision_error = self.harnesser.decide(snapshot)
            state.turn_count += 1
            if state.turn_count > state.max_turns:
                self._fail(state, artifact_manager, "Harness exceeded max_harness_turns.")
            if self._elapsed_seconds(state) > state.max_wall_clock_seconds:
                self._fail(state, artifact_manager, "Harness exceeded max_wall_clock_seconds.")

            tool_error = ""
            if decision is not None:
                tool_error = self._validate_tool_decision(state, decision, tool_registry)
                if tool_error.endswith("budget exhausted"):
                    self._fail(state, artifact_manager, tool_error)

            if decision is None or tool_error:
                invalid_decisions += 1
                observation = Observation(
                    ok=False,
                    source="harnesser",
                    summary="Harness decision rejected.",
                    errors=[decision_error or tool_error or "invalid harness decision"],
                )
                self._merge_observation(state, observation)
                self._append_trace(
                    artifact_manager=artifact_manager,
                    turn=state.turn_count,
                    decision_payload=raw_decision,
                    observation=observation,
                    metadata=metadata,
                )
                if invalid_decisions > 1:
                    self._fail(state, artifact_manager, "Harness produced invalid decisions twice in a row.")
                continue

            invalid_decisions = 0
            state.last_decision = decision.model_dump(mode="json")

            if decision.action == HarnessAction.stop or decision.stop:
                observation = Observation(
                    ok=decision.terminal_status == "success",
                    source="harnesser",
                    summary=decision.notes or f"Harness requested stop with status={decision.terminal_status}.",
                    errors=[] if decision.terminal_status == "success" else [decision.notes or "stop requested"],
                )
                self._append_trace(
                    artifact_manager=artifact_manager,
                    turn=state.turn_count,
                    decision_payload=raw_decision,
                    observation=observation,
                    metadata=metadata,
                )
                if decision.terminal_status == "success" and not state.finalized:
                    self._fail(state, artifact_manager, "Harness tried to stop with success before publish_final_artifacts.")
                return self._stop(state, artifact_manager, decision)

            if state.tool_calls >= state.max_tool_calls:
                self._fail(state, artifact_manager, "Harness exceeded max_tool_calls.")

            tool = tool_registry[decision.tool_name or ""]
            args = tool.args_model.model_validate(decision.tool_args)
            ctx = ToolContext(
                agent1=agent1,
                state=state,
                artifact_manager=artifact_manager,
                output_dir=output_dir,
                start=start,
                end=end,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                extractor=self.extractor,
                verifier=self.verifier,
                critic=self.critic,
                repair=self.repair,
                codegen=self.codegen,
                debugger=self.debugger,
                analyst=self.analyst,
                narrative_verifier=self.narrative_verifier,
            )
            observation = tool.handler(args, ctx)
            state.tool_calls += 1
            self._merge_observation(state, observation)
            self._append_trace(
                artifact_manager=artifact_manager,
                turn=state.turn_count,
                decision_payload=raw_decision,
                observation=observation,
                metadata=metadata,
                tool_invoked=tool.name,
            )

        self._fail(state, artifact_manager, "Harness loop terminated unexpectedly.")

    def _build_snapshot(self, state: AgentState, registry: dict[str, Any]) -> StateSnapshot:
        manifests = [
            ToolManifestEntry(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_model.model_json_schema(),
            )
            for tool in registry.values()
        ]
        return StateSnapshot(
            run_id=state.run_id,
            paper_id=state.paper_id,
            title=state.title,
            status=state.status,
            turn_count=state.turn_count,
            tool_calls=state.tool_calls,
            elapsed_seconds=self._elapsed_seconds(state),
            budgets={
                "max_harness_turns": state.max_turns,
                "max_tool_calls": state.max_tool_calls,
                "max_wall_clock_seconds": state.max_wall_clock_seconds,
                "max_blueprint_attempts": state.max_blueprint_attempts,
                "max_codegen_attempts": state.max_codegen_attempts,
                "max_execute_attempts": state.max_execute_attempts,
                "max_analyst_turns": state.max_analyst_turns,
            },
            counters={
                "blueprint_attempts": state.blueprint_attempts,
                "codegen_attempts": state.codegen_attempts,
                "execute_attempts": state.execute_attempts,
                "analyst_turns": state.analyst_turns,
            },
            current_blueprint=state.current_blueprint,
            verification=state.verification,
            critique=state.critique,
            current_strategy=state.current_strategy,
            execution=state.execution,
            metrics=state.metrics,
            narrative=state.narrative,
            narrative_review=state.narrative_review,
            artifact_paths=state.artifact_paths,
            notes=state.notes[-12:],
            last_error=state.last_error,
            last_decision=state.last_decision,
            last_observation=state.last_observation,
            available_tools=manifests,
        )

    def _validate_tool_decision(self, state: AgentState, decision: HarnessDecision, registry: dict[str, Any]) -> str:
        if decision.action != HarnessAction.invoke_tool:
            return ""
        if not decision.tool_name or decision.tool_name not in registry:
            return f"Unknown tool {decision.tool_name!r}"
        try:
            registry[decision.tool_name].args_model.model_validate(decision.tool_args)
        except ValidationError as exc:
            return str(exc)

        if decision.tool_name in {"extract_blueprint", "repair_blueprint"} and state.blueprint_attempts >= state.max_blueprint_attempts:
            return "blueprint budget exhausted"
        if decision.tool_name in {"draft_strategy_code", "revise_strategy_code"} and state.codegen_attempts >= state.max_codegen_attempts:
            return "codegen budget exhausted"
        if decision.tool_name == "execute_strategy" and state.execute_attempts >= state.max_execute_attempts:
            return "execute budget exhausted"
        if decision.tool_name == "analyst_turn" and state.analyst_turns >= state.max_analyst_turns:
            return "analyst budget exhausted"
        return ""

    def _merge_observation(self, state: AgentState, observation: Observation) -> None:
        for key, value in observation.state_updates.items():
            if key == "artifact_paths" and isinstance(value, dict):
                state.artifact_paths.update({str(k): str(v) for k, v in value.items()})
                continue
            if hasattr(state, key):
                setattr(state, key, value)
        if observation.artifact_paths:
            state.artifact_paths.update({str(k): str(v) for k, v in observation.artifact_paths.items()})
        state.last_observation = observation.model_dump(mode="json")
        if observation.summary:
            state.notes.append(observation.summary)
        if observation.errors:
            state.last_error = "; ".join(observation.errors)

    def _append_trace(
        self,
        *,
        artifact_manager: ArtifactManager,
        turn: int,
        decision_payload: dict[str, Any],
        observation: Observation,
        metadata: OpenRouterCallMetadata,
        tool_invoked: str | None = None,
    ) -> None:
        artifact_manager.append_trace(
            HarnessTraceEntry(
                turn=turn,
                decision=decision_payload,
                tool_invoked=tool_invoked,
                observation_summary=observation.summary,
                observation_ok=observation.ok,
                errors=observation.errors,
                artifact_paths=observation.artifact_paths,
                stdout=observation.stdout[:4000],
                stderr=observation.stderr[:4000],
                model=metadata.model,
                latency_seconds=metadata.latency_seconds,
                usage=metadata.usage,
            )
        )

    def _stop(self, state: AgentState, artifact_manager: ArtifactManager, decision: HarnessDecision) -> TerminalRecord:
        if decision.terminal_status != "success":
            self._fail(state, artifact_manager, decision.notes or f"Harness stopped with status={decision.terminal_status}.")
        state.status = TerminalStatus.success
        state_json_path = artifact_manager.output_dir / "state.json"
        state_json_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        state.artifact_paths.update({"trace": str(artifact_manager.trace_path), "state": str(state_json_path)})
        return TerminalRecord(
            run_id=state.run_id,
            paper_id=state.paper_id,
            terminal_status="success",
            message=decision.notes,
            output_dir=str(artifact_manager.output_dir),
            trace_path=str(artifact_manager.trace_path),
            artifact_paths=state.artifact_paths,
            state=state,
            last_decision=decision.model_dump(mode="json"),
        )

    def _fail(self, state: AgentState, artifact_manager: ArtifactManager, message: str) -> None:
        state.status = TerminalStatus.failed
        state.last_error = message
        state_json_path = artifact_manager.output_dir / "state.json"
        state_json_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        failure = FailureArtifact(
            run_id=state.run_id,
            paper_id=state.paper_id,
            terminal_status=TerminalStatus.failed,
            message=message,
            errors=[message],
            artifact_paths={**state.artifact_paths, "state": str(state_json_path)},
            trace_path=str(artifact_manager.trace_path),
        )
        failure_path = artifact_manager.write_failure(failure)
        failure.artifact_paths["failure"] = str(failure_path)
        raise AgenticRunFailed(failure)

    @staticmethod
    def _elapsed_seconds(state: AgentState) -> float:
        return max(0.0, (datetime.utcnow() - state.started_at).total_seconds())


def run_paper(
    agent1: Agent1ToAgent2Input,
    *,
    start: date | None = None,
    end: date | None = None,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
    max_harness_turns: int = 16,
    max_tool_calls: int = 32,
    max_wall_clock_seconds: int = 900,
    max_blueprint_attempts: int = 4,
    max_codegen_attempts: int = 5,
    max_execute_attempts: int = 5,
    max_analyst_turns: int = 3,
    harnesser: HarnesserAgent | None = None,
    extractor: ExtractorAgent | None = None,
    verifier: VerifierAgent | None = None,
    critic: CriticAgent | None = None,
    repair: RepairAgent | None = None,
    codegen: StrategyCodegenAgent | None = None,
    debugger: StrategyDebuggerAgent | None = None,
    analyst: AnalystAgent | None = None,
    narrative_verifier: NarrativeVerifierAgent | None = None,
) -> TerminalRecord:
    runner = HarnessRunner(
        harnesser=harnesser or HarnesserAgent(),
        extractor=extractor or ExtractorAgent(),
        verifier=verifier or VerifierAgent(),
        critic=critic or CriticAgent(),
        repair=repair or RepairAgent(),
        codegen=codegen or StrategyCodegenAgent(),
        debugger=debugger or StrategyDebuggerAgent(),
        analyst=analyst or AnalystAgent(),
        narrative_verifier=narrative_verifier or NarrativeVerifierAgent(),
    )
    return runner.run_paper(
        agent1,
        start=start,
        end=end,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        max_harness_turns=max_harness_turns,
        max_tool_calls=max_tool_calls,
        max_wall_clock_seconds=max_wall_clock_seconds,
        max_blueprint_attempts=max_blueprint_attempts,
        max_codegen_attempts=max_codegen_attempts,
        max_execute_attempts=max_execute_attempts,
        max_analyst_turns=max_analyst_turns,
    )
