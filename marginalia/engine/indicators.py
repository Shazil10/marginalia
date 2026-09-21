"""Deterministic technical indicators.

Pure pandas/numpy. No LLM-generated math. Every function takes a price Series or
DataFrame (indexed by date) and returns the same shape, so they compose cleanly
inside the engine templates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average."""
    window = max(1, int(window))
    return prices.rolling(window=window, min_periods=window).mean()


def trailing_return(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Total return over the trailing ``lookback`` periods: P_t / P_{t-lookback} - 1."""
    lookback = max(1, int(lookback))
    return prices / prices.shift(lookback) - 1.0


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change()


def rolling_volatility(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling volatility of daily returns."""
    window = max(2, int(window))
    rets = prices.pct_change()
    return rets.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's Relative Strength Index, vectorized via EWM.

    Returns values in [0, 100]. Works on a Series or DataFrame.
    """
    window = max(2, int(window))
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (only gains), RSI is 100. When both zero (flat), neutral 50.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return out


def above_sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Boolean mask: True where price is at/above its SMA(window)."""
    ma = sma(prices, window)
    return prices >= ma
