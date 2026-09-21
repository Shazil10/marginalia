"""Thin compatibility wrapper around the harness runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from agent2.agents.harness import HarnesserAgent, run_paper
from agent2.agents.models import TerminalRecord
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
from agent2.schemas import Agent1ToAgent2Input


@dataclass
class AgenticRunArtifacts:
    run_id: str
    output_dir: Path
    state: object
    result: TerminalRecord
    last_step: str


class OrchestratorAgent:
    """Compatibility shim over the harness-native entrypoint."""

    def __init__(
        self,
        *,
        harnesser: HarnesserAgent | None = None,
        extractor: ExtractorAgent | None = None,
        verifier: VerifierAgent | None = None,
        critic: CriticAgent | None = None,
        repair: RepairAgent | None = None,
        codegen: StrategyCodegenAgent | None = None,
        debugger: StrategyDebuggerAgent | None = None,
        analyst: AnalystAgent | None = None,
        narrative_verifier: NarrativeVerifierAgent | None = None,
    ) -> None:
        self.harnesser = harnesser
        self.extractor = extractor
        self.verifier = verifier
        self.critic = critic
        self.repair = repair
        self.codegen = codegen
        self.debugger = debugger
        self.analyst = analyst
        self.narrative_verifier = narrative_verifier

    def run(
        self,
        agent1: Agent1ToAgent2Input,
        *,
        start: date | None = None,
        end: date | None = None,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        artifacts_dir: str | Path = "artifacts",
        run_id: str | None = None,
        max_turns: int = 16,
        max_tool_calls: int = 32,
        max_wall_clock_seconds: int = 900,
        max_blueprint_attempts: int = 4,
        max_codegen_attempts: int = 5,
        max_execute_attempts: int = 5,
        max_analyst_turns: int = 3,
    ) -> AgenticRunArtifacts:
        terminal = run_paper(
            agent1,
            start=start,
            end=end,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            max_harness_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_wall_clock_seconds=max_wall_clock_seconds,
            max_blueprint_attempts=max_blueprint_attempts,
            max_codegen_attempts=max_codegen_attempts,
            max_execute_attempts=max_execute_attempts,
            max_analyst_turns=max_analyst_turns,
            harnesser=self.harnesser,
            extractor=self.extractor,
            verifier=self.verifier,
            critic=self.critic,
            repair=self.repair,
            codegen=self.codegen,
            debugger=self.debugger,
            analyst=self.analyst,
            narrative_verifier=self.narrative_verifier,
        )
        return AgenticRunArtifacts(
            run_id=terminal.run_id,
            output_dir=Path(terminal.output_dir),
            state=terminal.state,
            result=terminal,
            last_step="stop",
        )
