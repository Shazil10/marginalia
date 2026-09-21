"""Shared test fixtures and synthetic data helpers.

All engine/data tests run fully offline using deterministic synthetic prices, so
the suite never touches the network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable so `import marginalia` works.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def business_days(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def geometric_series(n: int, daily_drift: float, start_price: float = 100.0,
                     start: str = "2015-01-01") -> pd.Series:
    """A smooth (zero-noise) geometric price path with constant daily drift."""
    idx = business_days(n, start)
    prices = start_price * np.power(1.0 + daily_drift, np.arange(n))
    return pd.Series(prices, index=idx, name="ASSET")


def noisy_series(n: int, daily_drift: float, vol: float, seed: int = 0,
                 start_price: float = 100.0, start: str = "2015-01-01") -> pd.Series:
    """A geometric path with reproducible gaussian noise."""
    rng = np.random.default_rng(seed)
    idx = business_days(n, start)
    shocks = rng.normal(daily_drift, vol, size=n)
    prices = start_price * np.cumprod(1.0 + shocks)
    return pd.Series(prices, index=idx)


@pytest.fixture
def uptrend_prices():
    """Two assets trending up at different rates + a flat safe asset."""
    n = 600
    idx = business_days(n)
    df = pd.DataFrame(index=idx)
    df["AAA"] = geometric_series(n, 0.0010).values   # strong uptrend
    df["BBB"] = geometric_series(n, 0.0003).values   # mild uptrend
    df["BIL"] = geometric_series(n, 0.00005).values  # ~flat safe asset
    df["SPY"] = geometric_series(n, 0.0004).values   # benchmark
    df.index.name = "date"
    return df


@pytest.fixture
def updown_prices():
    """One asset that rises for the first half then falls in the second half."""
    n = 500
    idx = business_days(n)
    up = geometric_series(n // 2, 0.002, start_price=100.0).values
    down = geometric_series(n - n // 2, -0.002, start_price=float(up[-1])).values
    series = np.concatenate([up, down])
    df = pd.DataFrame({"AAA": series}, index=idx)
    df["BIL"] = geometric_series(n, 0.00005).values
    df["SPY"] = geometric_series(n, 0.0001).values
    df.index.name = "date"
    return df
