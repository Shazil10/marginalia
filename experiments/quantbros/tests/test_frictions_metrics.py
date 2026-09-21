from __future__ import annotations

import pandas as pd

from agent2.frictions import apply_turnover_costs
from agent2.metrics import build_metric_block
from agent2.schemas import FrictionSpec


def test_turnover_costs_reduce_returns_and_metrics_are_stable() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    gross_returns = pd.Series([0.0, 0.01, -0.005, 0.02, -0.01], index=index)
    turnover = pd.Series([0.0, 1.0, 0.0, 0.5, 0.5], index=index)
    net_returns = apply_turnover_costs(
        gross_returns,
        turnover,
        FrictionSpec(commission_bps=10.0, slippage_bps=5.0),
    )
    equity = (1.0 + net_returns).cumprod()
    metrics = build_metric_block(
        returns=net_returns,
        equity=equity,
        turnover=turnover,
        exposure=pd.Series([0.0, 1.0, 1.0, 1.0, 0.5], index=index),
    )

    assert net_returns.iloc[1] == gross_returns.iloc[1] - 0.0015
    assert round(metrics.total_return or 0.0, 6) == round(float(equity.iloc[-1] - 1.0), 6)
    assert round(metrics.turnover_mean or 0.0, 6) == 0.4
    assert (metrics.exposure_mean or 0.0) > 0.5
