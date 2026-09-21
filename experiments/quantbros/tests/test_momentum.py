from __future__ import annotations

import pandas as pd

from agent2.engine import backtest_momentum_rank
from agent2.schemas import (
    AssetUniverseType,
    LogicSpec,
    MomentumParams,
    PaperCategory,
    Rebalance,
    StrategyKind,
    UniverseSpec,
)


def test_momentum_rank_monthly_rebalance_runs_without_index_errors() -> None:
    index = pd.date_range("2024-01-01", periods=90, freq="B")
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.3 for i in range(len(index))],
            "QQQ": [100 + i * 0.4 for i in range(len(index))],
            "IWM": [100 + i * 0.1 for i in range(len(index))],
        },
        index=index,
    )
    logic = LogicSpec(
        paper_id="fixture-mom",
        strategy=StrategyKind.momentum_rank,
        universe=UniverseSpec(
            tickers=["SPY", "QQQ", "IWM"],
            benchmark="SPY",
            asset_universe_type=AssetUniverseType.etfs,
        ),
        momentum=MomentumParams(lookback_days=21, top_n=2, rebalance=Rebalance.monthly),
        position_sizing="equal_weight",
        strategy_category_hint=PaperCategory.momentum,
    )

    result = backtest_momentum_rank(prices, logic)

    assert len(result.gross_returns) == len(index)
    assert result.turnover.max() <= 1.0
    assert result.exposure.iloc[-1] > 0.0
