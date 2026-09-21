"""Typed contracts for Agent 1 ingest rows and the Agent 2 backtesting pipeline."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SCHEMA_VERSION = "1.1.0"


class PaperCategory(str, Enum):
    momentum = "momentum"
    mean_reversion = "mean_reversion"
    value = "value"
    carry = "carry"
    volatility = "volatility"
    trend = "trend"
    other = "other"
    unknown = "unknown"


class AssetUniverseType(str, Enum):
    equities = "equities"
    etfs = "etfs"
    multi_asset = "multi_asset"
    bonds = "bonds"
    futures = "futures"
    fx = "fx"
    crypto = "crypto"
    unknown = "unknown"


class RegimeClaim(str, Enum):
    bull = "bull"
    bear = "bear"
    sideways = "sideways"
    high_volatility = "high_volatility"
    low_volatility = "low_volatility"
    mixed = "mixed"
    not_specified = "not_specified"
    unknown = "unknown"


class Agent1Sections(BaseModel):
    """Normalized text sections supplied by the reader/ingestion layer."""

    abstract: str = ""
    methodology: str = ""
    introduction: str = ""
    conclusion: str = ""
    notes: str = ""


class Agent1ToAgent2Input(BaseModel):
    """Stable contract between the reader and the backtesting agent."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Version of the row contract.")
    paper_id: str = Field(..., description="Stable paper slug or database id.")
    title: str = ""
    source_url: str | None = None
    source_path: str | None = None
    parsed_at: datetime | None = None
    sections: Agent1Sections = Field(default_factory=Agent1Sections)
    extracted_tickers: list[str] = Field(default_factory=list)
    paper_category: PaperCategory = PaperCategory.unknown
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    asset_universe_notes: str = ""
    regime_claim: RegimeClaim = RegimeClaim.unknown
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Rebalance(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class StrategyKind(str, Enum):
    dual_moving_average = "dual_moving_average"
    momentum_rank = "momentum_rank"


class UniverseSpec(BaseModel):
    """Tradable universe for the backtest."""

    tickers: list[str] = Field(..., min_length=1)
    benchmark: str | None = "SPY"
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown


class DualMAParams(BaseModel):
    fast_window: int = Field(20, ge=2)
    slow_window: int = Field(50, ge=3)
    price_field: Literal["close", "adj_close"] = "adj_close"

    @model_validator(mode="after")
    def _validate_order(self) -> "DualMAParams":
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        return self


class MomentumParams(BaseModel):
    lookback_days: int = Field(126, ge=5)
    top_n: int = Field(3, ge=1)
    rebalance: Rebalance = Rebalance.monthly
    price_field: Literal["close", "adj_close"] = "adj_close"


class LogicSpec(BaseModel):
    """Validated, deterministic strategy intent."""

    paper_id: str = ""
    strategy: StrategyKind
    universe: UniverseSpec
    dual_ma: DualMAParams | None = None
    momentum: MomentumParams | None = None
    position_sizing: Literal["full_equity", "equal_weight"] = "full_equity"
    strategy_category_hint: PaperCategory = PaperCategory.unknown
    regime_hints: list[RegimeClaim] = Field(default_factory=list)
    source_method: Literal["preset", "embedded", "heuristic", "openrouter"] = "heuristic"
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_strategy_for_engine(cls, data: Any) -> Any:
        """Map common LLM strategy labels into the two supported StrategyKind values."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        raw = out.get("strategy")
        if isinstance(raw, StrategyKind):
            return out
        if isinstance(raw, str):
            key = raw.strip().lower().replace("-", "_").replace(" ", "_")
            allowed = {e.value for e in StrategyKind}
            if key in allowed:
                out["strategy"] = key
            elif key in ("mean_reversion", "meanreversion", "reversal", "contrarian", "short_term_reversal"):
                out["strategy"] = StrategyKind.momentum_rank.value
            elif any(
                part in key
                for part in (
                    "moving_average",
                    "ma_cross",
                    "crossover",
                    "dual_ma",
                    "trend_follow",
                    "timing",
                )
            ):
                out["strategy"] = StrategyKind.dual_moving_average.value
            else:
                out["strategy"] = StrategyKind.momentum_rank.value
            note = out.get("notes") or ""
            out["notes"] = (
                f"{note} [Engine mapping: strategy label {raw!r} → {out['strategy']!r}]"
            ).strip()
        strat = out.get("strategy")
        if strat == StrategyKind.dual_moving_average.value and not out.get("dual_ma"):
            out["dual_ma"] = {"fast_window": 20, "slow_window": 50, "price_field": "adj_close"}
        if strat == StrategyKind.momentum_rank.value and not out.get("momentum"):
            out["momentum"] = {
                "lookback_days": 126,
                "top_n": 3,
                "rebalance": "monthly",
                "price_field": "adj_close",
            }
        return out

    @model_validator(mode="after")
    def _validate_variant(self) -> "LogicSpec":
        if self.strategy == StrategyKind.dual_moving_average and self.dual_ma is None:
            raise ValueError("dual_ma params are required for dual_moving_average")
        if self.strategy == StrategyKind.momentum_rank and self.momentum is None:
            raise ValueError("momentum params are required for momentum_rank")
        return self


class DataAssumptions(BaseModel):
    """Point-in-time and data quality assumptions documented alongside every run."""

    price_adjustment: Literal["adjusted_close_yfinance"] = "adjusted_close_yfinance"
    signal_timing: str = (
        "Signals are computed on close at time t and applied using weights shifted by one bar "
        "to returns from t to t+1."
    )
    missing_data: str = (
        "Forward-fill interior gaps, keep leading NaNs until sufficient history exists, and "
        "drop days where all assets are missing."
    )
    survivorship_note: str = (
        "Hackathon assumption: static ticker lists are used, so survivorship bias remains for "
        "broad universes. Production should use point-in-time membership."
    )
    point_in_time_note: str = (
        "Only historical prices available up to each timestamp are used; no future returns or "
        "same-bar execution are allowed."
    )


class DataPlan(BaseModel):
    """Concrete data request consumed by the deterministic engine."""

    tickers: list[str] = Field(default_factory=list)
    benchmark: str | None = "SPY"
    required_series: list[Literal["ohlcv"]] = Field(default_factory=lambda: ["ohlcv"])
    source: Literal["yfinance"] = "yfinance"
    frequency: Literal["1d"] = "1d"
    start: date
    end: date
    signal_delay_bars: int = Field(default=1, ge=1)
    assumptions: DataAssumptions = Field(default_factory=DataAssumptions)
    extra: dict[str, Any] = Field(default_factory=dict)


class FrictionSpec(BaseModel):
    """Trading cost assumptions."""

    commission_bps: float = Field(0.0, ge=0)
    slippage_bps: float = Field(0.0, ge=0)
    apply_on: Literal["turnover"] = "turnover"


class MetricBlock(BaseModel):
    cagr: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    volatility_ann: float | None = None
    calmar: float | None = None
    total_return: float | None = None
    turnover_mean: float | None = None
    exposure_mean: float | None = None
    win_rate: float | None = None


class SensitivityPoint(BaseModel):
    label: str
    sharpe: float | None = None
    total_return: float | None = None


class RobustnessBlock(BaseModel):
    train_sharpe: float | None = None
    test_sharpe: float | None = None
    train_cagr: float | None = None
    test_cagr: float | None = None
    split_date: date | None = None
    walk_forward_windows: int = 1
    parameter_sensitivity: dict[str, list[SensitivityPoint]] = Field(default_factory=dict)
    suspect_flags: list[str] = Field(default_factory=list)


class OpenRouterCallMetadata(BaseModel):
    model: str
    latency_seconds: float
    usage: dict[str, Any] = Field(default_factory=dict)
    repair_attempts: int = 0


class PaperClassification(BaseModel):
    paper_category: PaperCategory = PaperCategory.unknown
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    asset_universe_notes: str = ""
    regime_claim: RegimeClaim = RegimeClaim.unknown
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: Literal["heuristic", "openrouter"] = "heuristic"
    reasoning: str = ""


class TearSheet(BaseModel):
    run_id: str
    paper_id: str
    title: str = ""
    paper_category: PaperCategory = PaperCategory.unknown
    asset_universe_type: AssetUniverseType = AssetUniverseType.unknown
    asset_universe_notes: str = ""
    regime_claim: RegimeClaim = RegimeClaim.unknown
    logic: LogicSpec
    data_plan: DataPlan
    frictions: FrictionSpec
    gross: MetricBlock
    net: MetricBlock
    robustness: RobustnessBlock
    analysis_summary: str = ""
    analysis_risks: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineArtifacts(BaseModel):
    agent1: Agent1ToAgent2Input
    logic: LogicSpec | None = None
    data_plan: DataPlan | None = None
    frictions: FrictionSpec | None = None
    tear_sheet: TearSheet | None = None
