import numpy as np
import pandas as pd

from marginalia.engine.metrics import compute_metrics
from tests.conftest import business_days


def test_empty_series_returns_zeros():
    m = compute_metrics(pd.Series([], dtype=float))
    assert m["sharpe"] == 0.0
    assert m["cagr"] == 0.0
    assert m["n_days"] == 0


def test_constant_zero_returns():
    r = pd.Series([0.0] * 100, index=business_days(100))
    m = compute_metrics(r)
    assert m["total_return"] == 0.0
    assert m["sharpe"] == 0.0          # zero vol -> guarded to 0
    assert m["max_drawdown"] == 0.0
    assert np.isfinite(m["calmar"])


def test_positive_drift_metrics():
    r = pd.Series([0.001] * 252, index=business_days(252))
    m = compute_metrics(r)
    assert m["total_return"] > 0
    assert m["cagr"] > 0
    assert m["max_drawdown"] == 0.0    # monotonic up -> no drawdown
    assert m["win_rate"] == 1.0


def test_drawdown_is_negative_when_losses():
    vals = [0.01] * 50 + [-0.02] * 50
    r = pd.Series(vals, index=business_days(100))
    m = compute_metrics(r)
    assert m["max_drawdown"] < 0


def test_benchmark_metrics_present():
    rng = np.random.default_rng(1)
    idx = business_days(300)
    strat = pd.Series(rng.normal(0.0008, 0.01, 300), index=idx)
    bench = pd.Series(rng.normal(0.0004, 0.01, 300), index=idx)
    m = compute_metrics(strat, bench)
    assert "alpha" in m and "beta" in m and "excess_cagr" in m
    assert np.isfinite(m["beta"])
