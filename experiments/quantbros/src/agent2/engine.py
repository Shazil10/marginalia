"""Deterministic vectorized backtests driven by validated LogicSpec objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from agent2.data import fetch_adj_close
from agent2.schemas import LogicSpec, StrategyKind


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    exposure: pd.Series
    weights: pd.Series | pd.DataFrame
    benchmark_returns: pd.Series | None
    benchmark_equity: pd.Series | None


def _require_price_columns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise KeyError(f"Missing ticker columns: {missing}")
    return prices[tickers].astype(float)


def backtest_dual_ma(prices: pd.DataFrame, logic: LogicSpec) -> BacktestResult:
    if logic.dual_ma is None:
        raise ValueError("dual_ma parameters are required")
    ticker = logic.universe.tickers[0]
    px = _require_price_columns(prices, [ticker])[ticker]
    fast_ma = px.rolling(logic.dual_ma.fast_window).mean()
    slow_ma = px.rolling(logic.dual_ma.slow_window).mean()
    signal = (fast_ma > slow_ma).astype(float)
    weights = signal.shift(1).fillna(0.0)
    asset_returns = px.pct_change().fillna(0.0)
    gross_returns = (weights * asset_returns).fillna(0.0)
    turnover = weights.diff().abs().fillna(0.0)
    equity = (1.0 + gross_returns).cumprod()

    benchmark_returns = None
    benchmark_equity = None
    benchmark = logic.universe.benchmark
    if benchmark and benchmark in prices.columns:
        benchmark_returns = prices[benchmark].astype(float).pct_change().fillna(0.0)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()

    return BacktestResult(
        gross_returns=gross_returns,
        equity=equity,
        turnover=turnover,
        exposure=weights.abs(),
        weights=weights,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity,
    )


def _rebalance_mask(index: pd.DatetimeIndex, rebalance: str) -> pd.Series:
    if rebalance == "daily":
        return pd.Series(index=index, data=True)
    if rebalance == "weekly":
        return pd.Series(index=index, data=index.weekday == 0)
    month_periods = index.to_period("M")
    month_changes = np.r_[True, month_periods[1:] != month_periods[:-1]]
    return pd.Series(index=index, data=month_changes)


def backtest_momentum_rank(prices: pd.DataFrame, logic: LogicSpec) -> BacktestResult:
    if logic.momentum is None:
        raise ValueError("momentum parameters are required")
    px = _require_price_columns(prices, logic.universe.tickers)
    lookback = logic.momentum.lookback_days
    scores = px / px.shift(lookback) - 1.0
    ranks = scores.rank(axis=1, ascending=False, method="first")
    leaders = (ranks <= logic.momentum.top_n).astype(float)
    target_weights = leaders.div(leaders.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    if logic.momentum.rebalance.value != "daily":
        mask = _rebalance_mask(px.index, logic.momentum.rebalance.value)
        mask_frame = pd.DataFrame(
            np.repeat(mask.values[:, None], target_weights.shape[1], axis=1),
            index=target_weights.index,
            columns=target_weights.columns,
        )
        target_weights = target_weights.where(mask_frame).ffill().fillna(0.0)

    weights = target_weights.shift(1).fillna(0.0)
    asset_returns = px.pct_change().fillna(0.0)
    gross_returns = (weights * asset_returns).sum(axis=1).fillna(0.0)
    turnover = target_weights.diff().abs().sum(axis=1).fillna(0.0) * 0.5
    exposure = weights.abs().sum(axis=1)
    equity = (1.0 + gross_returns).cumprod()

    benchmark_returns = None
    benchmark_equity = None
    benchmark = logic.universe.benchmark
    if benchmark and benchmark in prices.columns:
        benchmark_returns = prices[benchmark].astype(float).pct_change().fillna(0.0)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()

    return BacktestResult(
        gross_returns=gross_returns,
        equity=equity,
        turnover=turnover,
        exposure=exposure,
        weights=weights,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity,
    )


def run_backtest_from_logic(logic: LogicSpec, prices: pd.DataFrame) -> BacktestResult:
    if logic.strategy == StrategyKind.dual_moving_average:
        return backtest_dual_ma(prices, logic)
    if logic.strategy == StrategyKind.momentum_rank:
        return backtest_momentum_rank(prices, logic)
    raise NotImplementedError(f"Unsupported strategy kind: {logic.strategy}")


def fetch_and_backtest(
    logic: LogicSpec,
    *,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, BacktestResult]:
    tickers = list(dict.fromkeys(logic.universe.tickers + ([logic.universe.benchmark] if logic.universe.benchmark else [])))
    prices = fetch_adj_close(tickers, start=start, end=end)
    return prices, run_backtest_from_logic(logic, prices)


def reference_dual_ma_returns(price: pd.Series, fast: int, slow: int) -> pd.Series:
    """Reference single-asset dual-MA implementation for parity tests."""

    fast_ma = price.rolling(fast).mean()
    slow_ma = price.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    weights = signal.shift(1).fillna(0.0)
    returns = price.pct_change().fillna(0.0)
    return weights * returns
