import numpy as np
import pandas as pd

from marginalia.engine import indicators as ind
from tests.conftest import geometric_series, business_days


def test_sma_basic():
    s = pd.DataFrame({"X": [1, 2, 3, 4, 5]}, index=business_days(5))
    out = ind.sma(s, 2)
    assert np.isnan(out["X"].iloc[0])
    assert out["X"].iloc[1] == 1.5
    assert out["X"].iloc[4] == 4.5


def test_trailing_return():
    s = pd.DataFrame({"X": [100.0, 110.0, 121.0]}, index=business_days(3))
    out = ind.trailing_return(s, 1)
    assert np.isnan(out["X"].iloc[0])
    assert abs(out["X"].iloc[1] - 0.10) < 1e-9
    assert abs(out["X"].iloc[2] - 0.10) < 1e-9


def test_rsi_range_and_extremes():
    n = 100
    up = geometric_series(n, 0.01).to_frame("X")
    down = geometric_series(n, -0.01).to_frame("X")
    flat = pd.DataFrame({"X": [100.0] * n}, index=business_days(n))

    rsi_up = ind.rsi(up, 14)["X"].dropna()
    rsi_down = ind.rsi(down, 14)["X"].dropna()
    rsi_flat = ind.rsi(flat, 14)["X"].dropna()

    # All within [0, 100]
    assert rsi_up.between(0, 100).all()
    assert rsi_down.between(0, 100).all()

    # Pure uptrend -> RSI 100; pure downtrend -> RSI ~0; flat -> 50
    assert rsi_up.iloc[-1] > 99
    assert rsi_down.iloc[-1] < 1
    assert abs(rsi_flat.iloc[-1] - 50.0) < 1e-6


def test_above_sma():
    s = geometric_series(300, 0.001).to_frame("X")
    mask = ind.above_sma(s, 200)
    # A steadily rising series should be above its SMA once warmed up.
    assert mask["X"].iloc[-1]
    assert mask.dtypes["X"] == bool
