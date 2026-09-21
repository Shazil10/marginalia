"""Tool registry for the LLM-driven Agent 2 harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, Field

from agent2.agents.artifacts import ArtifactManager
from agent2.agents.base import RoleInvocationOutcome
from agent2.agents.models import (
    BlueprintCritiqueResponse,
    BlueprintExtractionResponse,
    BlueprintRepairResponse,
    BlueprintVerificationResponse,
    GeneratedStrategyArtifact,
    MetricBundle,
    NarrativeDraft,
    NarrativeReview,
    Observation,
    StaticScanResult,
    StrategyCodeResponse,
    StrategyExecutionResult,
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
from agent2.codegen.runtime import execute_strategy_code, static_scan_strategy_code
from agent2.pipeline import (
    build_metric_bundle_from_execution,
    build_strategy_cfg,
    prepare_blueprint_prices,
    publish_strategy_outputs,
)
from agent2.schemas import Agent1ToAgent2Input


@dataclass
class ToolContext:
    agent1: Agent1ToAgent2Input
    state: Any
    artifact_manager: ArtifactManager
    output_dir: Path
    start: date | None
    end: date | None
    commission_bps: float
    slippage_bps: float
    extractor: ExtractorAgent
    verifier: VerifierAgent
    critic: CriticAgent
    repair: RepairAgent
    codegen: StrategyCodegenAgent
    debugger: StrategyDebuggerAgent
    analyst: AnalystAgent
    narrative_verifier: NarrativeVerifierAgent


@dataclass
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Observation]


class EmptyArgs(BaseModel):
    pass


class ExtractBlueprintArgs(BaseModel):
    targeted_questions: list[str] = Field(default_factory=list)


class CritiqueBlueprintArgs(BaseModel):
    stage: str = "pre_codegen"


class RepairBlueprintArgs(BaseModel):
    validation_error: str = ""


class DraftStrategyCodeArgs(BaseModel):
    note: str = ""


class ReviseStrategyCodeArgs(BaseModel):
    validation_error: str = ""
    note: str = ""


class ReadArtifactArgs(BaseModel):
    path: str


class WriteNoteArgs(BaseModel):
    filename: str
    content: str


class ListArtifactsArgs(BaseModel):
    glob: str = "*"


def _role_failure(name: str, outcome: RoleInvocationOutcome) -> Observation:
    return Observation(
        ok=False,
        source=name,
        summary=f"{name} returned invalid structured JSON.",
        errors=[outcome.validation_error or "invalid structured JSON"],
        data={
            "raw_content": outcome.raw_content,
            "model": outcome.metadata.model if outcome.metadata else None,
            "latency_seconds": outcome.metadata.latency_seconds if outcome.metadata else None,
            "usage": outcome.metadata.usage if outcome.metadata else {},
        },
    )


def _read_current_code(ctx: ToolContext) -> str:
    if ctx.state.current_strategy is None:
        return ""
    return Path(ctx.state.current_strategy.path).read_text(encoding="utf-8")


def _write_strategy_artifact(
    *,
    ctx: ToolContext,
    response: StrategyCodeResponse,
    model: str,
    prompt_role: str,
) -> tuple[GeneratedStrategyArtifact, Path]:
    version = ctx.state.current_strategy_version + 1
    generated_dir = ctx.output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    code_path = generated_dir / f"strategy_v{version}.py"
    code_path.write_text(response.code.rstrip() + "\n", encoding="utf-8")
    artifact = GeneratedStrategyArtifact(
        version=version,
        path=str(code_path),
        summary=response.summary,
        assumptions=response.assumptions,
        known_limitations=response.known_limitations,
        static_scan=None,
        source_model=model,
        prompt_role=prompt_role,
    )
    metadata_path = ctx.artifact_manager.write_model("strategy_artifact", artifact)
    return artifact, metadata_path


def _tool_extract_blueprint(args: ExtractBlueprintArgs, ctx: ToolContext) -> Observation:
    outcome = ctx.extractor.run(
        agent1=ctx.agent1,
        prior_blueprint=ctx.state.current_blueprint,
        targeted_questions=args.targeted_questions,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("extract_blueprint", outcome)
    extracted = cast(BlueprintExtractionResponse, outcome.parsed)
    blueprint = extracted.blueprint.model_copy(
        update={
            "paper_id": ctx.agent1.paper_id,
            "title": extracted.blueprint.title or ctx.agent1.title,
        }
    )
    path = ctx.artifact_manager.write_model("strategy_blueprint", blueprint)
    return Observation(
        ok=True,
        source="extract_blueprint",
        summary="Strategy blueprint extracted from paper.",
        artifact_paths={"strategy_blueprint_versioned": str(path)},
        state_updates={
            "current_blueprint": blueprint,
            "current_blueprint_version": ctx.state.current_blueprint_version + 1,
            "blueprint_attempts": ctx.state.blueprint_attempts + 1,
            "verification": None,
            "critique": None,
            "current_strategy": None,
            "execution": None,
            "metrics": None,
            "narrative": None,
            "narrative_review": None,
            "last_error": "",
        },
    )


def _tool_verify_blueprint(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None:
        return Observation(ok=False, source="verify_blueprint", summary="Blueprint is required before verification.", errors=["missing blueprint"])
    outcome = ctx.verifier.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        failure_context=ctx.state.last_error,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("verify_blueprint", outcome)
    verification = cast(BlueprintVerificationResponse, outcome.parsed)
    path = ctx.artifact_manager.write_model("verification", verification)
    return Observation(
        ok=True,
        source="verify_blueprint",
        summary="Blueprint verification completed.",
        artifact_paths={"verification": str(path)},
        state_updates={"verification": verification, "last_error": ""},
    )


def _tool_critique_blueprint(args: CritiqueBlueprintArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None:
        return Observation(ok=False, source="critique_blueprint", summary="Blueprint is required before critique.", errors=["missing blueprint"])
    outcome = ctx.critic.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        stage=args.stage,
        code=_read_current_code(ctx),
        execution=ctx.state.execution,
        metrics=ctx.state.metrics,
        narrative=ctx.state.narrative,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("critique_blueprint", outcome)
    critique = cast(BlueprintCritiqueResponse, outcome.parsed)
    path = ctx.artifact_manager.write_model("critique", critique)
    return Observation(
        ok=True,
        source="critique_blueprint",
        summary=f"Critique completed for stage {critique.stage}.",
        artifact_paths={"critique": str(path)},
        state_updates={"critique": critique, "last_error": ""},
    )


def _tool_repair_blueprint(args: RepairBlueprintArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None:
        return Observation(ok=False, source="repair_blueprint", summary="Blueprint is required before repair.", errors=["missing blueprint"])
    outcome = ctx.repair.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        validation_error=args.validation_error or ctx.state.last_error,
        verifier=ctx.state.verification,
        critique=ctx.state.critique,
        strategy_code=_read_current_code(ctx),
        execution=ctx.state.execution,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("repair_blueprint", outcome)
    repaired = cast(BlueprintRepairResponse, outcome.parsed)
    repair_path = ctx.artifact_manager.write_model("repair_instructions", repaired)
    blueprint_path = ctx.artifact_manager.write_model("strategy_blueprint", repaired.blueprint)
    return Observation(
        ok=True,
        source="repair_blueprint",
        summary="Repair agent produced a revised blueprint.",
        artifact_paths={"repair": str(repair_path), "strategy_blueprint_versioned": str(blueprint_path)},
        state_updates={
            "current_blueprint": repaired.blueprint,
            "current_blueprint_version": ctx.state.current_blueprint_version + 1,
            "blueprint_attempts": ctx.state.blueprint_attempts + 1,
            "verification": None,
            "critique": None,
            "current_strategy": None,
            "execution": None,
            "metrics": None,
            "narrative": None,
            "narrative_review": None,
            "last_error": "",
        },
    )


def _tool_draft_strategy_code(args: DraftStrategyCodeArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None:
        return Observation(ok=False, source="draft_strategy_code", summary="Blueprint is required before code generation.", errors=["missing blueprint"])
    outcome = ctx.codegen.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        critique=ctx.state.critique,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("draft_strategy_code", outcome)
    generated = cast(StrategyCodeResponse, outcome.parsed)
    artifact, metadata_path = _write_strategy_artifact(
        ctx=ctx,
        response=generated,
        model=outcome.metadata.model if outcome.metadata else "",
        prompt_role="codegen",
    )
    return Observation(
        ok=True,
        source="draft_strategy_code",
        summary="Generated strategy code draft written.",
        artifact_paths={"strategy_artifact": str(metadata_path), "strategy_code_versioned": artifact.path},
        state_updates={
            "current_strategy": artifact,
            "current_strategy_version": artifact.version,
            "codegen_attempts": ctx.state.codegen_attempts + 1,
            "execution": None,
            "metrics": None,
            "narrative": None,
            "narrative_review": None,
            "last_error": "",
        },
        data={"note": args.note},
    )


def _tool_static_scan_strategy_code(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_strategy is None:
        return Observation(ok=False, source="static_scan_strategy_code", summary="Strategy code is required before static scan.", errors=["missing strategy code"])
    code = _read_current_code(ctx)
    scan = static_scan_strategy_code(code)
    path = ctx.artifact_manager.write_model("static_scan", scan)
    updated_strategy = ctx.state.current_strategy.model_copy(update={"static_scan": scan})
    return Observation(
        ok=scan.passed,
        source="static_scan_strategy_code",
        summary=scan.summary,
        errors=[item.message for item in scan.findings if item.severity == "error"],
        artifact_paths={"static_scan": str(path)},
        state_updates={
            "current_strategy": updated_strategy,
            "last_error": "" if scan.passed else "; ".join(item.message for item in scan.findings if item.severity == "error"),
        },
    )


def _tool_execute_strategy(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None or ctx.state.current_strategy is None:
        return Observation(ok=False, source="execute_strategy", summary="Blueprint and strategy code are required before execution.", errors=["missing blueprint or strategy code"])
    if ctx.state.current_strategy.static_scan is None or not ctx.state.current_strategy.static_scan.passed:
        return Observation(ok=False, source="execute_strategy", summary="Static scan must pass before sandbox execution.", errors=["static scan missing or failed"])
    data_plan, prices = prepare_blueprint_prices(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        start=ctx.start,
        end=ctx.end,
    )
    data_plan_path = ctx.output_dir / "data_plan.preview.json"
    data_plan_path.write_text(data_plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    cfg = build_strategy_cfg(
        blueprint=ctx.state.current_blueprint,
        data_plan=data_plan,
        commission_bps=ctx.commission_bps,
        slippage_bps=ctx.slippage_bps,
        start=data_plan.start,
        end=data_plan.end,
    )
    execution = execute_strategy_code(
        code_path=Path(ctx.state.current_strategy.path),
        prices=prices,
        cfg=cfg,
        output_dir=ctx.output_dir,
        data_plan=data_plan,
    )
    execution_path = ctx.artifact_manager.write_model("execution", execution)
    errors = [item.message for item in execution.sanity_checks if item.severity == "error"]
    updates = {
        "execution": execution,
        "execute_attempts": ctx.state.execute_attempts + 1,
        "last_error": "" if execution.passed else "; ".join(errors) or f"execution failed with code {execution.exit_code}",
    }
    return Observation(
        ok=execution.passed,
        source="execute_strategy",
        summary="Sandbox execution succeeded." if execution.passed else "Sandbox execution failed.",
        errors=errors if errors else ([] if execution.passed else [f"execution failed with code {execution.exit_code}"]),
        artifact_paths={
            "data_plan_preview": str(data_plan_path),
            "execution": str(execution_path),
            **execution.artifact_paths,
        },
        state_updates=updates,
        stdout=execution.stdout,
        stderr=execution.stderr,
    )


def _tool_revise_strategy_code(args: ReviseStrategyCodeArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None or ctx.state.current_strategy is None:
        return Observation(ok=False, source="revise_strategy_code", summary="Blueprint and current strategy code are required before revision.", errors=["missing blueprint or strategy code"])
    outcome = ctx.debugger.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        current_code=_read_current_code(ctx),
        validation_error=args.validation_error or ctx.state.last_error,
        execution=ctx.state.execution,
        critique=ctx.state.critique,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("revise_strategy_code", outcome)
    revised = cast(StrategyCodeResponse, outcome.parsed)
    artifact, metadata_path = _write_strategy_artifact(
        ctx=ctx,
        response=revised,
        model=outcome.metadata.model if outcome.metadata else "",
        prompt_role="debugger",
    )
    return Observation(
        ok=True,
        source="revise_strategy_code",
        summary="Revised strategy code written.",
        artifact_paths={"strategy_artifact": str(metadata_path), "strategy_code_versioned": artifact.path},
        state_updates={
            "current_strategy": artifact,
            "current_strategy_version": artifact.version,
            "codegen_attempts": ctx.state.codegen_attempts + 1,
            "execution": None,
            "metrics": None,
            "narrative": None,
            "narrative_review": None,
            "last_error": "",
        },
        data={"note": args.note},
    )


def _tool_compute_metrics(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.execution is None or not ctx.state.execution.passed:
        return Observation(ok=False, source="compute_metrics", summary="A successful execution is required before computing metrics.", errors=["missing successful execution"])
    metrics = build_metric_bundle_from_execution(
        execution=ctx.state.execution,
        output_dir=ctx.output_dir,
        title=f"{ctx.agent1.paper_id}: generated strategy",
        start=ctx.state.execution.data_plan.start if ctx.state.execution.data_plan else (ctx.start or date.today()),
        end=ctx.state.execution.data_plan.end if ctx.state.execution.data_plan else (ctx.end or date.today()),
    )
    artifact_paths = {
        "metrics_preview": str(ctx.output_dir / "metrics.preview.json"),
        "robustness_preview": str(ctx.output_dir / "robustness.preview.json"),
        **metrics.artifact_paths,
    }
    return Observation(
        ok=True,
        source="compute_metrics",
        summary="Deterministic metrics computed from executed outputs.",
        artifact_paths=artifact_paths,
        state_updates={"metrics": metrics, "last_error": ""},
    )


def _tool_compute_robustness(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.metrics is None:
        return Observation(ok=False, source="compute_robustness", summary="Metrics must exist before robustness review.", errors=["missing metrics"])
    return Observation(
        ok=True,
        source="compute_robustness",
        summary="Robustness artifacts are available from the deterministic metric bundle.",
        artifact_paths={"robustness_preview": str(ctx.output_dir / "robustness.preview.json")},
        state_updates={"last_error": ""},
    )


def _tool_analyst_turn(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None or ctx.state.current_strategy is None or ctx.state.metrics is None:
        return Observation(ok=False, source="analyst_turn", summary="Blueprint, strategy code, and metrics are required before analysis.", errors=["missing blueprint, strategy code, or metrics"])
    outcome = ctx.analyst.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        strategy_artifact_summary=ctx.state.current_strategy.model_dump(mode="json"),
        metrics=ctx.state.metrics,
        critique=ctx.state.critique,
        prior_narrative=ctx.state.narrative,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("analyst_turn", outcome)
    narrative = cast(NarrativeDraft, outcome.parsed)
    path = ctx.artifact_manager.write_model("narrative", narrative)
    return Observation(
        ok=True,
        source="analyst_turn",
        summary="Analyst narrative draft generated.",
        artifact_paths={"narrative": str(path)},
        state_updates={
            "narrative": narrative,
            "analyst_turns": ctx.state.analyst_turns + 1,
            "last_error": "",
        },
    )


def _tool_narrative_verify(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if ctx.state.current_blueprint is None or ctx.state.metrics is None or ctx.state.narrative is None:
        return Observation(ok=False, source="narrative_verify", summary="Blueprint, metrics, and a narrative draft are required before narrative verification.", errors=["missing blueprint, metrics, or narrative"])
    outcome = ctx.narrative_verifier.run(
        agent1=ctx.agent1,
        blueprint=ctx.state.current_blueprint,
        metrics=ctx.state.metrics,
        narrative=ctx.state.narrative,
    )
    if not outcome.success or outcome.parsed is None:
        return _role_failure("narrative_verify", outcome)
    review = cast(NarrativeReview, outcome.parsed)
    path = ctx.artifact_manager.write_model("narrative_review", review)
    errors = [item.message for item in review.findings if item.severity == "error"]
    return Observation(
        ok=not review.requires_repair,
        source="narrative_verify",
        summary="Narrative verification completed." if not review.requires_repair else "Narrative requires revision.",
        errors=errors,
        artifact_paths={"narrative_review": str(path)},
        state_updates={"narrative_review": review, "last_error": "" if not review.requires_repair else "; ".join(errors)},
    )


def _tool_publish_final_artifacts(_args: EmptyArgs, ctx: ToolContext) -> Observation:
    if (
        ctx.state.current_blueprint is None
        or ctx.state.current_strategy is None
        or ctx.state.execution is None
        or ctx.state.metrics is None
        or ctx.state.narrative is None
    ):
        return Observation(ok=False, source="publish_final_artifacts", summary="Blueprint, strategy code, execution, metrics, and narrative are required before publish.", errors=["missing final artifacts"])
    if not ctx.state.execution.passed:
        return Observation(ok=False, source="publish_final_artifacts", summary="Execution must pass before publish.", errors=["execution did not pass"])
    if ctx.state.narrative_review is not None and ctx.state.narrative_review.requires_repair:
        return Observation(ok=False, source="publish_final_artifacts", summary="Narrative review still requires repair.", errors=["narrative review requires repair"])
    tear_sheet = publish_strategy_outputs(
        agent1=ctx.agent1,
        run_id=ctx.state.run_id,
        output_dir=ctx.output_dir,
        blueprint=ctx.state.current_blueprint,
        strategy_artifact=ctx.state.current_strategy,
        execution=ctx.state.execution,
        metrics=ctx.state.metrics,
        narrative=ctx.state.narrative,
    )
    return Observation(
        ok=True,
        source="publish_final_artifacts",
        summary="Final artifacts published.",
        artifact_paths=tear_sheet.artifacts,
        state_updates={"finalized": True, "artifact_paths": tear_sheet.artifacts, "last_error": ""},
    )


def _tool_list_artifacts(args: ListArtifactsArgs, ctx: ToolContext) -> Observation:
    paths = sorted(str(path) for path in ctx.output_dir.glob(args.glob))
    return Observation(ok=True, source="list_artifacts", summary=f"Listed {len(paths)} artifacts.", data={"paths": paths})


def _tool_read_artifact(args: ReadArtifactArgs, ctx: ToolContext) -> Observation:
    path = Path(args.path)
    if not path.is_absolute():
        path = ctx.output_dir / args.path
    if not path.exists():
        return Observation(ok=False, source="read_artifact", summary="Artifact not found.", errors=[f"{path} does not exist"])
    return Observation(
        ok=True,
        source="read_artifact",
        summary=f"Read artifact {path.name}.",
        data={"path": str(path), "content": path.read_text(encoding="utf-8")},
    )


def _tool_write_note(args: WriteNoteArgs, ctx: ToolContext) -> Observation:
    path = ctx.output_dir / args.filename
    path.write_text(args.content, encoding="utf-8")
    return Observation(ok=True, source="write_note", summary=f"Wrote note {path.name}.", artifact_paths={"written_note": str(path)})


def build_tool_registry(*, allow_legacy_engine: bool = False) -> dict[str, ToolDefinition]:
    registry = {
        "extract_blueprint": ToolDefinition(
            "extract_blueprint",
            "Extract a paper-specific StrategyBlueprint from the paper text.",
            ExtractBlueprintArgs,
            _tool_extract_blueprint,
        ),
        "verify_blueprint": ToolDefinition(
            "verify_blueprint",
            "Verify that the current StrategyBlueprint is supported by the paper.",
            EmptyArgs,
            _tool_verify_blueprint,
        ),
        "critique_blueprint": ToolDefinition(
            "critique_blueprint",
            "Critique the current blueprint, code, execution, metrics, or narrative for actionable issues.",
            CritiqueBlueprintArgs,
            _tool_critique_blueprint,
        ),
        "repair_blueprint": ToolDefinition(
            "repair_blueprint",
            "Revise the StrategyBlueprint using verifier or critic findings.",
            RepairBlueprintArgs,
            _tool_repair_blueprint,
        ),
        "draft_strategy_code": ToolDefinition(
            "draft_strategy_code",
            "Draft runnable strategy Python code for the current blueprint.",
            DraftStrategyCodeArgs,
            _tool_draft_strategy_code,
        ),
        "static_scan_strategy_code": ToolDefinition(
            "static_scan_strategy_code",
            "Statically scan the generated strategy code for disallowed imports and constructs.",
            EmptyArgs,
            _tool_static_scan_strategy_code,
        ),
        "execute_strategy": ToolDefinition(
            "execute_strategy",
            "Run the generated strategy code in a locked-down subprocess on fetched price data.",
            EmptyArgs,
            _tool_execute_strategy,
        ),
        "revise_strategy_code": ToolDefinition(
            "revise_strategy_code",
            "Revise generated strategy code using scan failures, critiques, or execution diagnostics.",
            ReviseStrategyCodeArgs,
            _tool_revise_strategy_code,
        ),
        "compute_metrics": ToolDefinition(
            "compute_metrics",
            "Compute deterministic metrics from executed strategy outputs.",
            EmptyArgs,
            _tool_compute_metrics,
        ),
        "compute_robustness": ToolDefinition(
            "compute_robustness",
            "Surface deterministic robustness outputs derived from executed strategy returns.",
            EmptyArgs,
            _tool_compute_robustness,
        ),
        "analyst_turn": ToolDefinition(
            "analyst_turn",
            "Draft or revise the human-readable narrative for the tear sheet.",
            EmptyArgs,
            _tool_analyst_turn,
        ),
        "narrative_verify": ToolDefinition(
            "narrative_verify",
            "Verify the current narrative against the blueprint and deterministic metrics.",
            EmptyArgs,
            _tool_narrative_verify,
        ),
        "publish_final_artifacts": ToolDefinition(
            "publish_final_artifacts",
            "Publish canonical final artifacts for the run.",
            EmptyArgs,
            _tool_publish_final_artifacts,
        ),
        "list_artifacts": ToolDefinition(
            "list_artifacts",
            "List artifact files in the run directory.",
            ListArtifactsArgs,
            _tool_list_artifacts,
        ),
        "read_artifact": ToolDefinition(
            "read_artifact",
            "Read an artifact from the run directory.",
            ReadArtifactArgs,
            _tool_read_artifact,
        ),
        "write_note": ToolDefinition(
            "write_note",
            "Write a text note into the run directory.",
            WriteNoteArgs,
            _tool_write_note,
        ),
    }
    if allow_legacy_engine:
        registry["legacy_engine_disabled_notice"] = ToolDefinition(
            "legacy_engine_disabled_notice",
            "Legacy engine mode is not available through the default harness tool registry.",
            EmptyArgs,
            lambda _args, _ctx: Observation(
                ok=False,
                source="legacy_engine_disabled_notice",
                summary="Legacy engine tools are disabled in the default harness registry.",
                errors=["use --legacy-engine from CLI instead"],
            ),
        )
    return registry
