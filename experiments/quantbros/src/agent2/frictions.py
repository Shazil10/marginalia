"""Transaction-cost modelling for deterministic backtests."""

from __future__ import annotations

import pandas as pd

from agent2.schemas import FrictionSpec


def combined_bps(spec: FrictionSpec) -> float:
    return spec.commission_bps + spec.slippage_bps


def apply_turnover_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    spec: FrictionSpec,
) -> pd.Series:
    """
    Apply linear costs on turnover.

    Turnover should already represent notional one-way turnover per period.
    """

    aligned_turnover = turnover.reindex(gross_returns.index).fillna(0.0).astype(float)
    cost_rate = aligned_turnover * (combined_bps(spec) / 10000.0)
    return (gross_returns.fillna(0.0) - cost_rate).astype(float)


def summarize_friction_drag(gross_returns: pd.Series, net_returns: pd.Series) -> dict[str, float]:
    gross_total = float((1.0 + gross_returns.fillna(0.0)).prod() - 1.0)
    net_total = float((1.0 + net_returns.fillna(0.0)).prod() - 1.0)
    return {
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "drag": gross_total - net_total,
    }
