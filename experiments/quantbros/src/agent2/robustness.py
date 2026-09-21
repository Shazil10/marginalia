"""Robustness checks for hackathon-scale deterministic backtests."""

from __future__ import annotations

import math

import pandas as pd

from agent2.engine import run_backtest_from_logic
from agent2.frictions import apply_turnover_costs
from agent2.metrics import build_metric_block
from agent2.schemas import FrictionSpec, LogicSpec, RobustnessBlock, SensitivityPoint, StrategyKind


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _parameter_variants(logic: LogicSpec) -> dict[str, list[LogicSpec]]:
    variants: dict[str, list[LogicSpec]] = {}
    if logic.strategy == StrategyKind.dual_moving_average and logic.dual_ma is not None:
        base = logic.dual_ma
        fast_candidates = sorted({max(2, base.fast_window - 1), base.fast_window + 1})
        slow_candidates = sorted({max(base.fast_window + 1, base.slow_window - 1), base.slow_window + 1})
        variants["fast_window"] = [
            logic.model_copy(update={"dual_ma": base.model_copy(update={"fast_window": value})})
            for value in fast_candidates
            if value < base.slow_window
        ]
        variants["slow_window"] = [
            logic.model_copy(update={"dual_ma": base.model_copy(update={"slow_window": value})})
            for value in slow_candidates
            if value > base.fast_window
        ]
        return variants

    if logic.strategy == StrategyKind.momentum_rank and logic.momentum is not None:
        base = logic.momentum
        top_candidates = sorted({max(1, base.top_n - 1), base.top_n + 1})
        lookback_candidates = sorted({max(5, base.lookback_days - 21), base.lookback_days + 21})
        variants["top_n"] = [
            logic.model_copy(update={"momentum": base.model_copy(update={"top_n": value})})
            for value in top_candidates
            if value <= len(logic.universe.tickers)
        ]
        variants["lookback_days"] = [
            logic.model_copy(update={"momentum": base.model_copy(update={"lookback_days": value})})
            for value in lookback_candidates
        ]
    return variants


def evaluate_robustness(
    *,
    prices: pd.DataFrame,
    logic: LogicSpec,
    frictions: FrictionSpec,
    net_returns: pd.Series,
    net_equity: pd.Series,
    exposure: pd.Series,
) -> RobustnessBlock:
    if prices.empty:
        return RobustnessBlock(suspect_flags=["empty_price_panel"])

    split_idx = max(2, int(len(prices) * 0.7))
    split_idx = min(split_idx, len(prices) - 1)
    split_date = prices.index[split_idx].date()

    train_prices = prices.iloc[: split_idx + 1]
    test_prices = prices.iloc[split_idx:]

    train_result = run_backtest_from_logic(logic, train_prices)
    test_result = run_backtest_from_logic(logic, test_prices)
    train_net = apply_turnover_costs(train_result.gross_returns, train_result.turnover, frictions)
    test_net = apply_turnover_costs(test_result.gross_returns, test_result.turnover, frictions)
    train_metrics = build_metric_block(
        returns=train_net,
        equity=_equity_from_returns(train_net),
        turnover=train_result.turnover,
        exposure=train_result.exposure,
    )
    test_metrics = build_metric_block(
        returns=test_net,
        equity=_equity_from_returns(test_net),
        turnover=test_result.turnover,
        exposure=test_result.exposure,
    )

    parameter_sensitivity: dict[str, list[SensitivityPoint]] = {}
    for name, variants in _parameter_variants(logic).items():
        points: list[SensitivityPoint] = []
        for variant in variants:
            variant_result = run_backtest_from_logic(variant, prices)
            variant_net = apply_turnover_costs(variant_result.gross_returns, variant_result.turnover, frictions)
            variant_equity = _equity_from_returns(variant_net)
            variant_metrics = build_metric_block(
                returns=variant_net,
                equity=variant_equity,
                turnover=variant_result.turnover,
                exposure=variant_result.exposure,
            )
            label = ""
            if name == "fast_window" and variant.dual_ma is not None:
                label = str(variant.dual_ma.fast_window)
            elif name == "slow_window" and variant.dual_ma is not None:
                label = str(variant.dual_ma.slow_window)
            elif name == "top_n" and variant.momentum is not None:
                label = str(variant.momentum.top_n)
            elif name == "lookback_days" and variant.momentum is not None:
                label = str(variant.momentum.lookback_days)
            points.append(
                SensitivityPoint(
                    label=label,
                    sharpe=variant_metrics.sharpe,
                    total_return=variant_metrics.total_return,
                )
            )
        parameter_sensitivity[name] = points

    flags: list[str] = []
    overall_metrics = build_metric_block(
        returns=net_returns,
        equity=net_equity,
        turnover=pd.Series(index=net_returns.index, data=0.0),
        exposure=exposure,
    )
    if len(prices) < 252:
        flags.append("short_history")
    if overall_metrics.sharpe is not None and not math.isnan(overall_metrics.sharpe) and overall_metrics.sharpe > 3.5:
        flags.append("absurd_sharpe")
    if (
        overall_metrics.max_drawdown is not None
        and overall_metrics.total_return is not None
        and overall_metrics.max_drawdown > -0.03
        and overall_metrics.total_return > 0.25
    ):
        flags.append("too_smooth_equity")
    if (
        train_metrics.sharpe is not None
        and test_metrics.sharpe is not None
        and not math.isnan(train_metrics.sharpe)
        and not math.isnan(test_metrics.sharpe)
        and abs(train_metrics.sharpe - test_metrics.sharpe) > 1.5
    ):
        flags.append("train_test_instability")
    if overall_metrics.exposure_mean is not None and overall_metrics.exposure_mean < 0.1:
        flags.append("low_market_exposure")

    return RobustnessBlock(
        train_sharpe=train_metrics.sharpe,
        test_sharpe=test_metrics.sharpe,
        train_cagr=train_metrics.cagr,
        test_cagr=test_metrics.cagr,
        split_date=split_date,
        walk_forward_windows=1,
        parameter_sensitivity=parameter_sensitivity,
        suspect_flags=flags,
    )
