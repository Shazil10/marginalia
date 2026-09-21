"""Strategy ranking and Kelly-based position sizing.

Pure, deterministic math operating on backtest outputs. Given several
``BacktestResult`` objects we can:

* rank them by a chosen metric, and
* size how much capital a given investor should allocate, using a (fractional,
  capped) Kelly criterion scaled by the investor's drawdown tolerance.

None of this touches the network or an LLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rank_results(results: Sequence, by: str = "sharpe", descending: bool = True) -> List:
    """Sort a list of BacktestResult-like objects by ``result.metrics[by]``.

    Missing metrics sort to the bottom. Ties broken deterministically by name.
    """
    def key(r):
        val = r.metrics.get(by, float("-inf") if descending else float("inf"))
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = float("-inf") if descending else float("inf")
        return (val, getattr(r.spec, "strategy_name", ""))

    return sorted(results, key=key, reverse=descending)


def kelly_fraction(
    daily_returns: pd.Series,
    *,
    rf_annual: float = 0.0,
    cap: float = 1.0,
) -> float:
    """Full-Kelly allocation fraction for a return stream.

    For a series with mean excess daily return ``m`` and daily variance ``v``,
    the growth-optimal fraction is ``f* = m / v``. Returns a value clamped to
    ``[0, cap]`` (no shorting, capped leverage). Degenerate inputs return 0.
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    rf_daily = rf_annual / TRADING_DAYS
    excess = r - rf_daily
    var = float(excess.var())
    if var <= 0 or np.isnan(var):
        return 0.0
    f = float(excess.mean()) / var
    if np.isnan(f) or np.isinf(f):
        return 0.0
    return float(min(max(f, 0.0), cap))


def _kelly_scale_for_risk(risk_class: Optional[str], drawdown_tolerance: Optional[float]) -> float:
    """How much of full Kelly to use, based on investor risk appetite.

    Full Kelly is famously too aggressive, so we always use a fraction. More
    conservative investors use a smaller fraction.
    """
    if drawdown_tolerance is not None:
        # Map a 0-50% drawdown tolerance onto a 0.15x - 0.6x Kelly fraction.
        dd = max(0.0, min(float(drawdown_tolerance), 0.5))
        return 0.15 + (dd / 0.5) * (0.60 - 0.15)
    table = {
        "conservative": 0.15,
        "moderate-conservative": 0.25,
        "moderate": 0.40,
        "aggressive": 0.60,
    }
    return table.get((risk_class or "moderate").lower(), 0.40)


def size_position(
    daily_returns: pd.Series,
    capital: float,
    *,
    risk_class: Optional[str] = None,
    drawdown_tolerance: Optional[float] = None,
    rf_annual: float = 0.0,
    max_fraction: float = 1.0,
) -> Dict[str, float]:
    """Recommend a dollar allocation for a strategy given an investor profile.

    Returns a dict with the raw Kelly fraction, the risk-scaled fraction actually
    used, and the resulting dollar allocation.
    """
    full = kelly_fraction(daily_returns, rf_annual=rf_annual, cap=1.0)
    scale = _kelly_scale_for_risk(risk_class, drawdown_tolerance)
    used = min(full * scale, max_fraction)
    used = max(0.0, used)
    return {
        "full_kelly_fraction": round(full, 4),
        "kelly_scale": round(scale, 4),
        "allocation_fraction": round(used, 4),
        "allocation_dollars": round(capital * used, 2),
        "cash_dollars": round(capital * (1.0 - used), 2),
    }
