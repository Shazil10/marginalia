from __future__ import annotations

import pandas as pd

from agent2.engine import backtest_dual_ma, reference_dual_ma_returns
from agent2.schemas import (
    AssetUniverseType,
    DualMAParams,
    LogicSpec,
    PaperCategory,
    StrategyKind,
    UniverseSpec,
)


def test_dual_ma_matches_reference_returns() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 103.0, 104.0],
        },
        index=index,
    )
    logic = LogicSpec(
        paper_id="fixture",
        strategy=StrategyKind.dual_moving_average,
        universe=UniverseSpec(
            tickers=["SPY"],
            benchmark="SPY",
            asset_universe_type=AssetUniverseType.etfs,
        ),
        dual_ma=DualMAParams(fast_window=2, slow_window=3),
        strategy_category_hint=PaperCategory.trend,
    )

    result = backtest_dual_ma(prices, logic)
    expected = reference_dual_ma_returns(prices["SPY"], fast=2, slow=3)

    pd.testing.assert_series_equal(result.gross_returns, expected, check_names=False)
    assert result.turnover.sum() > 0
    assert result.exposure.max() <= 1.0
