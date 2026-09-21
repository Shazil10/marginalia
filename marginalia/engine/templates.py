"""Deterministic strategy templates.

Each template is a pure function:

    template(prices: DataFrame, spec: StrategySpec, params: dict) -> weights: DataFrame

``prices`` is a clean (dates x tickers) adjusted-close frame covering all of
``spec.all_tickers()``. ``weights`` is a (dates x tickers) frame of *target*
portfolio weights known at each date (no look-ahead — every value at date t is
computed only from data up to and including t). Weights for a row may sum to
less than 1; the remainder is treated as cash by the backtester.

The backtester is responsible for sampling these weights at the rebalance
frequency and shifting them one day forward so decisions are acted on the
following day.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from marginalia.engine import indicators as ind
from marginalia.engine.spec import StrategySpec, StrategyTemplate


def _empty_weights(prices: pd.DataFrame, columns) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=list(columns))


def buy_and_hold(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    uni = [t for t in spec.universe if t in prices.columns]
    w = _empty_weights(prices, uni)
    if uni:
        w.loc[:, uni] = 1.0 / len(uni)
    return w


def time_series_momentum(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    uni = [t for t in spec.universe if t in prices.columns]
    w = _empty_weights(prices, uni)
    if not uni:
        return w
    sma_window = int(params.get("sma_window", 200))
    lookback = int(params.get("lookback", 126))
    p = prices[uni]
    trend_on = ind.above_sma(p, sma_window)
    mom_on = ind.trailing_return(p, lookback) > 0
    on = (trend_on & mom_on).astype(float)
    # Each asset that is "on" gets an equal 1/N slice; the rest is cash.
    w.loc[:, uni] = on * (1.0 / len(uni))
    return w


def cross_sectional_momentum(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    uni = [t for t in spec.universe if t in prices.columns]
    w = _empty_weights(prices, uni)
    if not uni:
        return w
    lookback = int(params.get("lookback", 126))
    top_n = max(1, min(int(params.get("top_n", 3)), len(uni)))
    mom = ind.trailing_return(prices[uni], lookback)
    # Rank per row (descending). Highest momentum -> rank 1.
    ranks = mom.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= top_n) & mom.notna()
    w.loc[:, uni] = selected.astype(float) * (1.0 / top_n)
    return w


def dual_momentum(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    """Antonacci-style Global Equity Momentum (relative + absolute).

    Pick the single best-performing universe asset over ``lookback``; hold it only
    if its trailing return is positive (absolute momentum), otherwise hold the
    safe asset.
    """
    uni = [t for t in spec.universe if t in prices.columns]
    cols = list(uni)
    safe = spec.safe_asset if (spec.safe_asset and spec.safe_asset in prices.columns) else None
    if safe and safe not in cols:
        cols.append(safe)
    w = _empty_weights(prices, cols)
    if not uni:
        return w
    lookback = int(params.get("lookback", 252))
    mom = ind.trailing_return(prices[uni], lookback)

    best = mom.idxmax(axis=1)         # ticker with highest momentum each day
    best_val = mom.max(axis=1)        # its momentum value
    valid = best_val.notna()

    for date in prices.index[valid.values]:
        b = best.loc[date]
        if pd.isna(b):
            continue
        if best_val.loc[date] > 0:
            w.at[date, b] = 1.0
        elif safe is not None:
            w.at[date, safe] = 1.0
        # else: stay in cash (all zeros)
    return w


def mean_reversion_rsi(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    """Per-asset RSI mean reversion.

    Enter long when RSI falls below ``oversold``; exit to cash when RSI rises above
    ``overbought``. Position is held between thresholds (stateful, via forward-fill).
    """
    uni = [t for t in spec.universe if t in prices.columns]
    w = _empty_weights(prices, uni)
    if not uni:
        return w
    window = int(params.get("rsi_window", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    r = ind.rsi(prices[uni], window)

    # signal: 1 on oversold cross, 0 on overbought cross, NaN otherwise -> ffill
    signal = pd.DataFrame(np.nan, index=r.index, columns=uni)
    signal = signal.mask(r < oversold, 1.0)
    signal = signal.mask(r > overbought, 0.0)
    position = signal.ffill().fillna(0.0)
    w.loc[:, uni] = position * (1.0 / len(uni))
    return w


def sector_rotation(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    """Momentum rotation across (sector) ETFs with a trend filter and safe fallback.

    Select the top_n assets by trailing return, but only hold those that are also
    above their long-term trend SMA. Capital that fails the trend filter rotates
    into the safe asset (or cash if none).
    """
    uni = [t for t in spec.universe if t in prices.columns]
    cols = list(uni)
    safe = spec.safe_asset if (spec.safe_asset and spec.safe_asset in prices.columns) else None
    if safe and safe not in cols:
        cols.append(safe)
    w = _empty_weights(prices, cols)
    if not uni:
        return w
    lookback = int(params.get("lookback", 126))
    trend_sma = int(params.get("trend_sma", 200))
    top_n = max(1, min(int(params.get("top_n", 3)), len(uni)))

    mom = ind.trailing_return(prices[uni], lookback)
    ranks = mom.rank(axis=1, ascending=False, method="first")
    in_top = (ranks <= top_n) & mom.notna()
    in_trend = ind.above_sma(prices[uni], trend_sma)

    held = in_top & in_trend
    slice_w = 1.0 / top_n
    w.loc[:, uni] = held.astype(float) * slice_w

    if safe is not None:
        # Any of the top_n slots failing the trend filter park in the safe asset.
        failed_slots = (in_top & ~in_trend).sum(axis=1).astype(float)
        w.loc[:, safe] = w.loc[:, safe] + failed_slots * slice_w
    return w


TEMPLATE_FUNCS = {
    StrategyTemplate.BUY_AND_HOLD: buy_and_hold,
    StrategyTemplate.TIME_SERIES_MOMENTUM: time_series_momentum,
    StrategyTemplate.CROSS_SECTIONAL_MOMENTUM: cross_sectional_momentum,
    StrategyTemplate.DUAL_MOMENTUM: dual_momentum,
    StrategyTemplate.MEAN_REVERSION_RSI: mean_reversion_rsi,
    StrategyTemplate.SECTOR_ROTATION: sector_rotation,
}


def compute_weights(prices: pd.DataFrame, spec: StrategySpec, params: Dict[str, float]) -> pd.DataFrame:
    """Dispatch to the template implementation for ``spec.template``."""
    func = TEMPLATE_FUNCS.get(spec.template)
    if func is None:
        raise ValueError(f"No implementation for template {spec.template}")
    return func(prices, spec, params)
