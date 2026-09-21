"""Structured strategy specification.

This is the contract between the LLM (which reads a paper) and the deterministic
engine (which runs the backtest). The LLM never writes code — it emits one of a
fixed set of *templates* with validated parameters. That makes every run
reproducible and testable.

Adding a new strategy family = add a ``StrategyTemplate`` member + implement it
in ``engine.templates``. Nothing else in the system needs to change.
"""

from __future__ import annotations

import enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SpecValidationError(ValueError):
    """Raised when an LLM-produced spec cannot be coerced into a valid StrategySpec."""


class StrategyTemplate(str, enum.Enum):
    """The deterministic strategy families the engine can execute."""

    BUY_AND_HOLD = "buy_and_hold"
    # Hold each asset only while it is above its own trailing moving average; else cash.
    TIME_SERIES_MOMENTUM = "time_series_momentum"
    # Rank the universe by trailing return; hold the top N, equal-weight, rebalanced.
    CROSS_SECTIONAL_MOMENTUM = "cross_sectional_momentum"
    # Antonacci-style: pick best of relative momentum, but only if it beats cash (absolute).
    DUAL_MOMENTUM = "dual_momentum"
    # Per-asset mean reversion via RSI thresholds.
    MEAN_REVERSION_RSI = "mean_reversion_rsi"
    # Momentum rotation across a (sector) ETF set with a cash/safe fallback in downtrends.
    SECTOR_ROTATION = "sector_rotation"


class RebalanceFrequency(str, enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @property
    def pandas_rule(self) -> str:
        return {
            "daily": "D",
            "weekly": "W-FRI",
            "monthly": "ME",
            "quarterly": "QE",
        }[self.value]


# Default parameter grids per template (used when the paper hides the values).
# Keys must match the parameter names each template understands.
_TEMPLATE_PARAM_KEYS = {
    StrategyTemplate.BUY_AND_HOLD: set(),
    StrategyTemplate.TIME_SERIES_MOMENTUM: {"lookback", "sma_window"},
    StrategyTemplate.CROSS_SECTIONAL_MOMENTUM: {"lookback", "top_n"},
    StrategyTemplate.DUAL_MOMENTUM: {"lookback"},
    StrategyTemplate.MEAN_REVERSION_RSI: {"rsi_window", "oversold", "overbought"},
    StrategyTemplate.SECTOR_ROTATION: {"lookback", "top_n", "trend_sma"},
}


class StrategySpec(BaseModel):
    """A validated, executable description of a trading strategy."""

    strategy_name: str = Field(..., min_length=1, max_length=200)
    template: StrategyTemplate
    universe: List[str] = Field(..., min_length=1, max_length=100)
    benchmark: str = "SPY"
    rebalance: RebalanceFrequency = RebalanceFrequency.MONTHLY

    # The single best-guess parameter set.
    parameters: Dict[str, float] = Field(default_factory=dict)
    # Optional ranges for a (capped) grid search; maps param -> candidate values.
    parameter_ranges: Dict[str, List[float]] = Field(default_factory=dict)

    # Optional safe/cash asset for rotation strategies when trend is negative.
    safe_asset: Optional[str] = Field(default="BIL")

    # Provenance
    source_paper: str = "Unknown"
    strategy_type: str = "other"
    notes: str = ""

    model_config = {"use_enum_values": False, "extra": "ignore"}

    # -- field-level cleaning ------------------------------------------------

    @field_validator("universe", mode="before")
    @classmethod
    def _clean_universe(cls, v):
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            raise SpecValidationError(f"universe must be a list, got {type(v).__name__}")
        cleaned = []
        seen = set()
        for t in v:
            s = str(t).strip().upper()
            if s and s not in seen:
                seen.add(s)
                cleaned.append(s)
        if not cleaned:
            raise SpecValidationError("universe is empty after cleaning")
        return cleaned

    @field_validator("benchmark", "safe_asset", mode="before")
    @classmethod
    def _clean_ticker(cls, v):
        if v is None:
            return v
        return str(v).strip().upper() or None

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_params(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise SpecValidationError(f"parameters must be a dict, got {type(v).__name__}")
        out = {}
        for k, val in v.items():
            try:
                out[str(k)] = float(val)
            except (TypeError, ValueError):
                # skip non-numeric params rather than blowing up the whole spec
                continue
        return out

    @field_validator("parameter_ranges", mode="before")
    @classmethod
    def _coerce_ranges(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise SpecValidationError(f"parameter_ranges must be a dict, got {type(v).__name__}")
        out = {}
        for k, vals in v.items():
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            nums = []
            for x in vals:
                try:
                    nums.append(float(x))
                except (TypeError, ValueError):
                    continue
            if nums:
                out[str(k)] = sorted(set(nums))
        return out

    # -- cross-field validation ---------------------------------------------

    @model_validator(mode="after")
    def _validate_template_params(self):
        allowed = _TEMPLATE_PARAM_KEYS[self.template]
        # Fill in sane defaults for any missing required params.
        defaults = default_parameters(self.template)
        merged = dict(defaults)
        merged.update({k: v for k, v in self.parameters.items() if k in allowed})
        object.__setattr__(self, "parameters", merged)

        # Drop range keys the template doesn't understand.
        object.__setattr__(
            self,
            "parameter_ranges",
            {k: v for k, v in self.parameter_ranges.items() if k in allowed},
        )

        # top_n cannot exceed the universe size.
        if "top_n" in self.parameters:
            self.parameters["top_n"] = float(
                max(1, min(int(self.parameters["top_n"]), len(self.universe)))
            )
        return self

    # -- convenience ---------------------------------------------------------

    @classmethod
    def from_llm_dict(cls, data: dict) -> "StrategySpec":
        """Build a spec from a loose LLM dict, raising SpecValidationError on failure."""
        if not isinstance(data, dict):
            raise SpecValidationError(f"expected dict, got {type(data).__name__}")
        try:
            return cls.model_validate(data)
        except SpecValidationError:
            raise
        except Exception as exc:  # pydantic ValidationError, etc.
            raise SpecValidationError(str(exc)) from exc

    def all_tickers(self) -> List[str]:
        """Universe + benchmark + safe asset (deduped, upper)."""
        out = list(self.universe)
        for extra in (self.benchmark, self.safe_asset):
            if extra and extra not in out:
                out.append(extra)
        return out

    def int_param(self, key: str, default: int = 0) -> int:
        return int(round(self.parameters.get(key, default)))


def default_parameters(template: StrategyTemplate) -> Dict[str, float]:
    """Reasonable defaults so a spec is always runnable even if the paper is vague."""
    return {
        StrategyTemplate.BUY_AND_HOLD: {},
        StrategyTemplate.TIME_SERIES_MOMENTUM: {"lookback": 126.0, "sma_window": 200.0},
        StrategyTemplate.CROSS_SECTIONAL_MOMENTUM: {"lookback": 126.0, "top_n": 3.0},
        StrategyTemplate.DUAL_MOMENTUM: {"lookback": 252.0},
        StrategyTemplate.MEAN_REVERSION_RSI: {
            "rsi_window": 14.0,
            "oversold": 30.0,
            "overbought": 70.0,
        },
        StrategyTemplate.SECTOR_ROTATION: {
            "lookback": 126.0,
            "top_n": 3.0,
            "trend_sma": 200.0,
        },
    }[template]
