"""Deterministic backtester.

Given a validated ``StrategySpec`` and a clean price frame, it:

1. Computes target weights from the strategy template (no look-ahead).
2. Samples those weights at the rebalance frequency and shifts them one day
   forward, so a decision made using data through day *t* is acted on day *t+1*.
3. Holds weights between rebalances, applies linear transaction costs on turnover.
4. Produces a daily portfolio return series and a standard metrics dict.
5. Optionally runs a (capped, deterministic) grid search over
   ``spec.parameter_ranges`` and keeps the parameter set with the best Sharpe.

No LLM, no network beyond the price fetch, fully reproducible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from marginalia.data.market import get_prices
from marginalia.engine.metrics import compute_metrics
from marginalia.engine.spec import RebalanceFrequency, StrategySpec
from marginalia.engine.templates import compute_weights

# Hard cap so a careless parameter grid can never explode runtime.
MAX_GRID_COMBOS = 12
DEFAULT_COST_BPS = 1.0  # round-trip transaction cost per unit turnover, in basis points


@dataclass
class BacktestResult:
    spec: StrategySpec
    metrics: Dict[str, float]
    optimized_parameters: Dict[str, float]
    daily_returns: pd.Series
    weights: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None
    grid_searched: int = 1
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable summary (daily returns as [date, value] pairs)."""
        return {
            "strategy_name": self.spec.strategy_name,
            "template": self.spec.template.value,
            "source_paper": self.spec.source_paper,
            "universe": self.spec.universe,
            "benchmark": self.spec.benchmark,
            "rebalance": self.spec.rebalance.value,
            "optimized_parameters": self.optimized_parameters,
            "metrics": self.metrics,
            "grid_searched": self.grid_searched,
            "warnings": self.warnings,
            "daily_returns": [
                [d.strftime("%Y-%m-%d"), round(float(v), 8)]
                for d, v in self.daily_returns.items()
            ],
        }


def _rebalance_dates(index: pd.DatetimeIndex, freq: RebalanceFrequency) -> pd.DatetimeIndex:
    if freq == RebalanceFrequency.DAILY:
        return index
    # Last available trading day in each period bucket.
    grouped = pd.Series(index, index=index).resample(freq.pandas_rule).last().dropna()
    dates = pd.DatetimeIndex(grouped.values)
    return dates.intersection(index)


def _portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cost_bps: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Convert target weights into a daily portfolio return series (no look-ahead)."""
    cols = list(weights.columns)
    asset_returns = asset_returns.reindex(columns=cols).fillna(0.0)

    # Effective weights: only change on rebalance dates, else hold previous.
    eff = weights.copy()
    mask = ~eff.index.isin(rebal_dates)
    eff.loc[mask, :] = np.nan
    eff = eff.ffill().fillna(0.0)

    # Shift one day forward: today's target is held starting tomorrow (no look-ahead).
    held = eff.shift(1).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)

    # Transaction costs: charge on the L1 turnover whenever weights change.
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs

    return net, held


def _run_single(
    spec: StrategySpec,
    prices: pd.DataFrame,
    params: Dict[str, float],
    cost_bps: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    weights = compute_weights(prices, spec, params)
    asset_returns = prices.pct_change()
    rebal = _rebalance_dates(prices.index, spec.rebalance)
    port_ret, held = _portfolio_returns(weights, asset_returns, rebal, cost_bps)
    # Drop the leading warm-up region where weights are all zero (no position yet).
    first_active = held.abs().sum(axis=1)
    active_idx = first_active[first_active > 0].index
    if len(active_idx):
        port_ret = port_ret.loc[active_idx[0]:]
    return port_ret.dropna(), held


def _grid_combos(spec: StrategySpec) -> List[Dict[str, float]]:
    """Build a capped list of parameter combinations to evaluate."""
    ranges = {k: v for k, v in spec.parameter_ranges.items() if v}
    if not ranges:
        return [dict(spec.parameters)]

    keys = list(ranges.keys())
    value_lists = [ranges[k] for k in keys]
    combos = []
    for combo in itertools.product(*value_lists):
        merged = dict(spec.parameters)
        merged.update({k: float(v) for k, v in zip(keys, combo)})
        combos.append(merged)

    if len(combos) > MAX_GRID_COMBOS:
        # Deterministic, evenly-spaced subsample (never random) so runs reproduce.
        step = len(combos) / MAX_GRID_COMBOS
        idxs = sorted({int(i * step) for i in range(MAX_GRID_COMBOS)})
        combos = [combos[i] for i in idxs]
    return combos


def run_backtest(
    spec: StrategySpec,
    *,
    start=None,
    end=None,
    prices: Optional[pd.DataFrame] = None,
    cost_bps: float = DEFAULT_COST_BPS,
    optimize: bool = True,
) -> BacktestResult:
    """Run ``spec`` and return a :class:`BacktestResult`.

    If ``prices`` is provided it is used directly (great for tests / offline runs);
    otherwise prices for ``spec.all_tickers()`` are fetched via the data adapter.
    """
    warnings: List[str] = []

    if prices is None:
        prices = get_prices(spec.all_tickers(), start, end)
    else:
        prices = prices.copy()
        prices.columns = [str(c).upper() for c in prices.columns]
        if start is not None:
            prices = prices.loc[prices.index >= pd.Timestamp(start)]
        if end is not None:
            prices = prices.loc[prices.index <= pd.Timestamp(end)]

    missing = [t for t in spec.universe if t not in prices.columns]
    if missing:
        warnings.append(f"Missing price data for: {missing}; proceeding without them.")
    if not any(t in prices.columns for t in spec.universe):
        raise ValueError("None of the universe tickers have price data; cannot backtest.")

    combos = _grid_combos(spec) if optimize else [dict(spec.parameters)]

    best: Optional[Tuple[float, Dict[str, float], pd.Series, pd.DataFrame]] = None
    for params in combos:
        port_ret, held = _run_single(spec, prices, params, cost_bps)
        if len(port_ret) < 2:
            continue
        m = compute_metrics(port_ret)
        score = m["sharpe"]
        if best is None or score > best[0]:
            best = (score, params, port_ret, held)

    if best is None:
        raise ValueError("Backtest produced no usable return series for any parameter set.")

    _, best_params, best_ret, best_weights = best

    # Benchmark returns aligned to the strategy window.
    bench_ret = None
    bench = spec.benchmark
    if bench and bench in prices.columns:
        bench_ret = prices[bench].pct_change().reindex(best_ret.index).dropna()

    metrics = compute_metrics(best_ret, bench_ret)

    return BacktestResult(
        spec=spec,
        metrics=metrics,
        optimized_parameters=best_params,
        daily_returns=best_ret,
        weights=best_weights,
        benchmark_returns=bench_ret,
        grid_searched=len(combos),
        warnings=warnings,
    )
