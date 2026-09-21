"""Typed contracts for the Agent 2 harness, blueprint, codegen, and execution loops."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent2.schemas import (
    AssetUniverseType,
    DataPlan,
    FrictionSpec,
    MetricBlock,
    PaperCategory,
    RegimeClaim,
    RobustnessBlock,
)


class AgentRole(str, Enum):
    harnesser = "harnesser"
    extractor = "extractor"
    verifier = "verifier"
    critic = "critic"
    repair = "repair"
    codegen = "codegen"
    debugger = "debugger"
    analyst = "analyst"
    narrative_verifier = "narrative_verifier"


class TerminalStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"
    aborted = "aborted"


class StructuredFinding(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    paper_quote: str = ""
    location: str = ""
    requires_repair: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_finding_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "message" not in out:
            for key in ("issue", "description", "detail", "text", "explanation"):
                value = out.get(key)
                if value is not None:
                    out["message"] = str(value)
                    break
        if not out.get("code"):
            out["code"] = "finding"
        if not out.get("location"):
            for key in ("line_ref", "line_reference", "chunk_ref", "field", "path"):
                value = out.get(key)
                if value:
                    out["location"] = str(value)
                    break
        return out


class PaperDecision(BaseModel):
    """Legacy paper classification contract retained for ingest and legacy engine mode."""

    paper_category: PaperCategory = PaperCategory.unknown
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    asset_universe_notes: str = ""
    regime_claim: RegimeClaim = RegimeClaim.unknown
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_quotes: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            fval = float(value)
            if fval > 1.0 and fval <= 100.0:
                fval = fval / 100.0
            return min(1.0, max(0.0, fval))
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"", "unknown", "n/a", "na", "none"}:
                return None
            mapping = {
                "very high": 0.95,
                "high": 0.85,
                "moderate": 0.55,
                "medium": 0.55,
                "low": 0.25,
                "very low": 0.1,
            }
            if raw in mapping:
                return mapping[raw]
            try:
                fval = float(raw.rstrip("%"))
            except ValueError:
                return None
            if raw.endswith("%") or fval > 1.0:
                fval = fval / 100.0
            return min(1.0, max(0.0, fval))
        return None


class StrategyUniverse(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    benchmark: str | None = None
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    description: str = ""
    notes: str = ""

    @field_validator("tickers", mode="before")
    @classmethod
    def _normalize_tickers(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip().upper() for item in value if str(item).strip()]
        return [str(value).strip().upper()]


class BlueprintDataRequirement(BaseModel):
    name: str
    series_type: str = "adjusted_close"
    tickers: list[str] = Field(default_factory=list)
    frequency: str = "1d"
    notes: str = ""

    @field_validator("tickers", mode="before")
    @classmethod
    def _normalize_ticker_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return [str(item).strip().upper() for item in value if str(item).strip()]


class BlueprintCitation(BaseModel):
    quote: str
    section: str = ""
    rationale: str = ""


class StrategyBlueprint(BaseModel):
    paper_id: str
    title: str = ""
    universe: StrategyUniverse = Field(default_factory=StrategyUniverse)
    benchmark: str | None = None
    data_requirements: list[BlueprintDataRequirement] = Field(default_factory=list)
    signal_definition: str
    rebalance_schedule: str
    holding_period: str
    risk_controls: list[str] = Field(default_factory=list)
    leverage_rules: str = ""
    timing_and_delay_rules: list[str] = Field(default_factory=list)
    corporate_action_assumptions: list[str] = Field(default_factory=list)
    required_parameters: dict[str, Any] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    citations: list[BlueprintCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_defaults(self) -> "StrategyBlueprint":
        if self.benchmark is None and self.universe.benchmark:
            self.benchmark = self.universe.benchmark
        if self.universe.benchmark is None and self.benchmark:
            self.universe.benchmark = self.benchmark
        if not self.data_requirements:
            tickers = list(dict.fromkeys(self.universe.tickers))
            if self.benchmark and self.benchmark not in tickers:
                tickers.append(self.benchmark)
            self.data_requirements = [
                BlueprintDataRequirement(
                    name="price_history",
                    series_type="adjusted_close",
                    tickers=tickers,
                    frequency="1d",
                    notes="Defaulted to adjusted close because the strategy executable consumes daily price history.",
                )
            ]
        return self


class BlueprintExtractionResponse(BaseModel):
    blueprint: StrategyBlueprint
    evidence_quotes: list[str] = Field(default_factory=list)
    reasoning: str = ""
    open_questions: list[str] = Field(default_factory=list)


def _normalize_pass_or_repair_status(data: dict[str, Any]) -> None:
    raw = data.get("status")
    if raw is None:
        raw = data.get("verdict") or data.get("result") or data.get("decision")
    if raw is None:
        data["status"] = "pass"
        return
    normalized = str(raw).strip().lower().replace(" ", "_")
    if normalized in {"repair_needed", "needs_repair", "repair", "fail", "failed", "no"}:
        data["status"] = "repair_needed"
    else:
        data["status"] = "pass"


class BlueprintVerificationResponse(BaseModel):
    status: Literal["pass", "repair_needed"]
    findings: list[StructuredFinding] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "missing_information" not in out:
            for key in ("missing_parameters", "missing_fields", "open_questions"):
                value = out.get(key)
                if isinstance(value, list):
                    out["missing_information"] = value
                    break
        _normalize_pass_or_repair_status(out)
        return out

    @property
    def requires_repair(self) -> bool:
        return self.status == "repair_needed" or any(item.requires_repair for item in self.findings)


class BlueprintCritiqueResponse(BaseModel):
    stage: Literal["pre_codegen", "post_execution", "post_metrics", "narrative"] = "pre_codegen"
    status: Literal["pass", "repair_needed"]
    findings: list[StructuredFinding] = Field(default_factory=list)
    lookahead_risk: bool = False
    ambiguity_flags: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        stage = str(out.get("stage", "") or out.get("critique_stage", "")).strip().lower().replace(" ", "_")
        if not stage:
            out["stage"] = "pre_codegen"
        elif "post" in stage and "metric" in stage:
            out["stage"] = "post_metrics"
        elif "post" in stage and "execution" in stage:
            out["stage"] = "post_execution"
        elif "narrative" in stage:
            out["stage"] = "narrative"
        else:
            out["stage"] = "pre_codegen"
        if isinstance(out.get("lookahead_risk"), dict):
            risk = out["lookahead_risk"]
            out["lookahead_risk"] = bool(risk.get("present") or risk.get("risk") or risk.get("detected"))
        _normalize_pass_or_repair_status(out)
        return out

    @property
    def requires_repair(self) -> bool:
        return self.status == "repair_needed" or any(item.requires_repair for item in self.findings)


class BlueprintRepairResponse(BaseModel):
    blueprint: StrategyBlueprint
    repair_summary: str = ""
    resolved_findings: list[str] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)


class StrategyCodeResponse(BaseModel):
    code: str
    summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if not out.get("code"):
            for key in ("python", "strategy_code", "body", "content"):
                value = out.get(key)
                if isinstance(value, str) and value.strip():
                    out["code"] = value
                    break
        return out


class StaticScanResult(BaseModel):
    passed: bool = False
    findings: list[StructuredFinding] = Field(default_factory=list)
    allowlisted_imports: list[str] = Field(default_factory=list)
    denied_imports: list[str] = Field(default_factory=list)
    summary: str = ""


class GeneratedStrategyArtifact(BaseModel):
    version: int
    path: str
    summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    static_scan: StaticScanResult | None = None
    source_model: str = ""
    prompt_role: str = ""


class StrategyExecutionResult(BaseModel):
    code_path: str
    weights_path: str
    gross_returns_path: str
    net_returns_path: str | None = None
    turnover_path: str | None = None
    diagnostics_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    sanity_checks: list[StructuredFinding] = Field(default_factory=list)
    passed: bool = False
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    data_plan: DataPlan | None = None
    benchmark_used: str = ""


class MetricBundle(BaseModel):
    gross: MetricBlock
    net: MetricBlock
    robustness: RobustnessBlock
    data_plan: DataPlan
    friction_note: str = ""
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    benchmark_used: str = ""
    start: date
    end: date


class NarrativeDraft(BaseModel):
    summary: str
    methodology: str = ""
    performance_summary: str = ""
    robustness_summary: str = ""
    key_risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if not (out.get("summary") or "").strip():
            narrative = out.get("narrative")
            if isinstance(narrative, dict):
                pieces: list[str] = []
                for key in ("title", "headline", "summary", "overview"):
                    value = narrative.get(key)
                    if isinstance(value, str) and value.strip():
                        pieces.append(value.strip())
                body = narrative.get("body") or narrative.get("text") or narrative.get("content")
                if isinstance(body, str) and body.strip():
                    pieces.append(body.strip())
                if pieces:
                    out["summary"] = "\n\n".join(pieces)
            elif isinstance(narrative, str) and narrative.strip():
                out["summary"] = narrative.strip()
        if not (out.get("summary") or "").strip():
            out["summary"] = "No analyst narrative returned."
        return out


class NarrativeReview(BaseModel):
    status: Literal["pass", "repair_needed"]
    findings: list[StructuredFinding] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        _normalize_pass_or_repair_status(out)
        return out

    @property
    def requires_repair(self) -> bool:
        return self.status == "repair_needed" or any(item.requires_repair for item in self.findings)


class FinalTearSheet(BaseModel):
    run_id: str
    paper_id: str
    title: str = ""
    paper_category: PaperCategory = PaperCategory.unknown
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    asset_universe_notes: str = ""
    regime_claim: RegimeClaim = RegimeClaim.unknown
    blueprint: StrategyBlueprint
    strategy_artifact: GeneratedStrategyArtifact
    execution: StrategyExecutionResult
    metrics: MetricBundle
    narrative: NarrativeDraft
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BacktestObservation(BaseModel):
    """Legacy calculator observation retained for legacy engine mode."""

    data_plan: DataPlan
    frictions: FrictionSpec
    gross: MetricBlock
    net: MetricBlock
    robustness: RobustnessBlock
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    benchmark_used: str = ""
    start: date
    end: date


class Observation(BaseModel):
    ok: bool = True
    source: str
    summary: str
    errors: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    state_updates: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


class HarnessAction(str, Enum):
    invoke_tool = "invoke_tool"
    stop = "stop"


class ToolRequest(BaseModel):
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class HarnessDecision(BaseModel):
    action: HarnessAction
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    subagent: str | None = None
    subagent_payload: dict[str, Any] = Field(default_factory=dict)
    stop: bool = False
    terminal_status: Literal["success", "failed", "aborted"] | None = None
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_action_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        raw_action = out.get("action")
        if isinstance(raw_action, str):
            normalized = raw_action.strip().lower().replace(" ", "_")
            if normalized in {"invoke_tool", "call_tool", "tool", "invoke"}:
                out["action"] = HarnessAction.invoke_tool.value
            elif normalized == "stop":
                out["action"] = HarnessAction.stop.value
            else:
                if not out.get("tool_name"):
                    out["tool_name"] = raw_action
                out["action"] = HarnessAction.invoke_tool.value
        return out

    @model_validator(mode="after")
    def _validate_shape(self) -> "HarnessDecision":
        if self.action == HarnessAction.invoke_tool and not self.tool_name:
            raise ValueError("tool_name is required for invoke_tool")
        if self.action == HarnessAction.stop:
            self.stop = True
            if self.terminal_status is None:
                raise ValueError("terminal_status is required when stopping")
        return self


class ToolManifestEntry(BaseModel):
    name: str
    description: str
    args_schema: dict[str, Any]


class AgentState(BaseModel):
    run_id: str
    paper_id: str
    title: str = ""
    status: TerminalStatus = TerminalStatus.running
    turn_count: int = 0
    tool_calls: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    max_turns: int = 16
    max_tool_calls: int = 32
    max_wall_clock_seconds: int = 900
    max_blueprint_attempts: int = 4
    max_codegen_attempts: int = 5
    max_execute_attempts: int = 5
    max_analyst_turns: int = 3
    blueprint_attempts: int = 0
    codegen_attempts: int = 0
    execute_attempts: int = 0
    analyst_turns: int = 0
    current_blueprint_version: int = 0
    current_strategy_version: int = 0
    current_blueprint: StrategyBlueprint | None = None
    verification: BlueprintVerificationResponse | None = None
    critique: BlueprintCritiqueResponse | None = None
    current_strategy: GeneratedStrategyArtifact | None = None
    execution: StrategyExecutionResult | None = None
    metrics: MetricBundle | None = None
    narrative: NarrativeDraft | None = None
    narrative_review: NarrativeReview | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    last_error: str = ""
    notes: list[str] = Field(default_factory=list)
    last_decision: dict[str, Any] = Field(default_factory=dict)
    last_observation: dict[str, Any] = Field(default_factory=dict)
    finalized: bool = False


class StateSnapshot(BaseModel):
    run_id: str
    paper_id: str
    title: str = ""
    status: TerminalStatus = TerminalStatus.running
    turn_count: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    budgets: dict[str, int | float] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)
    current_blueprint: StrategyBlueprint | None = None
    verification: BlueprintVerificationResponse | None = None
    critique: BlueprintCritiqueResponse | None = None
    current_strategy: GeneratedStrategyArtifact | None = None
    execution: StrategyExecutionResult | None = None
    metrics: MetricBundle | None = None
    narrative: NarrativeDraft | None = None
    narrative_review: NarrativeReview | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    last_error: str = ""
    last_decision: dict[str, Any] = Field(default_factory=dict)
    last_observation: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[ToolManifestEntry] = Field(default_factory=list)


class HarnessTraceEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    turn: int
    decision: dict[str, Any]
    tool_invoked: str | None = None
    observation_summary: str = ""
    observation_ok: bool = True
    errors: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    model: str | None = None
    latency_seconds: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class FailureArtifact(BaseModel):
    run_id: str
    paper_id: str
    terminal_status: TerminalStatus = TerminalStatus.failed
    message: str
    errors: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    trace_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgenticRunFailed(RuntimeError):
    def __init__(self, failure: FailureArtifact) -> None:
        super().__init__(failure.message)
        self.failure = failure


class TerminalRecord(BaseModel):
    run_id: str
    paper_id: str
    terminal_status: Literal["success", "failed", "aborted"]
    message: str = ""
    output_dir: str
    trace_path: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    state: AgentState
    last_decision: dict[str, Any] = Field(default_factory=dict)
