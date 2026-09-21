import numpy as np
import pandas as pd
import pytest

from marginalia.portfolio.ranking import rank_results, kelly_fraction, size_position
from tests.conftest import business_days


class _FakeResult:
    def __init__(self, name, sharpe):
        self.metrics = {"sharpe": sharpe}
        self.spec = type("S", (), {"strategy_name": name})()


def test_rank_results_by_sharpe():
    a = _FakeResult("A", 0.5)
    b = _FakeResult("B", 1.5)
    c = _FakeResult("C", 1.0)
    ranked = rank_results([a, b, c], by="sharpe")
    assert [r.spec.strategy_name for r in ranked] == ["B", "C", "A"]


def test_rank_handles_missing_metric():
    a = _FakeResult("A", 1.0)
    b = _FakeResult("B", None)
    ranked = rank_results([a, b], by="sharpe")
    assert ranked[0].spec.strategy_name == "A"


def test_kelly_positive_for_good_strategy():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000), index=business_days(1000))
    f = kelly_fraction(r)
    assert 0.0 < f <= 1.0


def test_kelly_zero_for_negative_strategy():
    r = pd.Series([-0.001] * 500, index=business_days(500))
    # constant negative -> zero variance guard -> 0
    assert kelly_fraction(r) == 0.0


def test_kelly_zero_for_empty():
    assert kelly_fraction(pd.Series([], dtype=float)) == 0.0


def test_kelly_capped():
    rng = np.random.default_rng(1)
    # huge mean, tiny vol -> Kelly would blow up, must be capped
    r = pd.Series(rng.normal(0.05, 0.001, 500), index=business_days(500))
    assert kelly_fraction(r, cap=1.0) == 1.0


def test_size_position_scales_with_risk():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 1000), index=business_days(1000))
    cons = size_position(r, 10000, risk_class="conservative")
    aggr = size_position(r, 10000, risk_class="aggressive")
    assert aggr["allocation_fraction"] >= cons["allocation_fraction"]
    assert cons["allocation_dollars"] + cons["cash_dollars"] == pytest.approx(10000, abs=1e-6)


def test_size_position_drawdown_tolerance():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 1000), index=business_days(1000))
    low = size_position(r, 10000, drawdown_tolerance=0.05)
    high = size_position(r, 10000, drawdown_tolerance=0.45)
    assert high["allocation_fraction"] >= low["allocation_fraction"]
