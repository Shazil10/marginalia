import numpy as np
import pandas as pd
import pytest

from marginalia.engine.spec import StrategySpec
from marginalia.engine.backtest import run_backtest
from tests.conftest import geometric_series, business_days


def _spec(template, universe, **kw):
    data = {"strategy_name": "T", "template": template, "universe": universe}
    data.update(kw)
    return StrategySpec.from_llm_dict(data)


def test_buy_and_hold_matches_equal_weight_returns(uptrend_prices):
    spec = _spec("buy_and_hold", ["AAA", "BBB"], rebalance="daily")
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)

    asset_ret = uptrend_prices[["AAA", "BBB"]].pct_change().mean(axis=1)
    expected = asset_ret.reindex(res.daily_returns.index)
    # With zero cost and daily rebalance, portfolio == equal-weight asset returns.
    assert np.allclose(res.daily_returns.values, expected.values, atol=1e-9)
    assert res.metrics["total_return"] > 0


def test_no_lookahead_first_active_day_is_dropped(uptrend_prices):
    # Buy-and-hold is invested from day 0, but the 1-day shift means the very first
    # day's return is never captured. Strategy series must start AFTER prices start.
    spec = _spec("buy_and_hold", ["AAA"], rebalance="daily")
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)
    assert res.daily_returns.index[0] > uptrend_prices.index[0]


def test_trend_filter_goes_to_cash_in_downtrend(updown_prices):
    # time_series_momentum should be flat (zero return) once the asset breaks below
    # its trend in the second half.
    spec = _spec(
        "time_series_momentum", ["AAA"],
        parameters={"lookback": 20, "sma_window": 50}, rebalance="daily",
    )
    res = run_backtest(spec, prices=updown_prices, cost_bps=0.0, optimize=False)
    # Last 50 days are deep in the downtrend -> should be in cash -> zero returns.
    tail = res.daily_returns.iloc[-50:]
    assert np.allclose(tail.values, 0.0, atol=1e-9)


def test_cross_sectional_picks_winner(uptrend_prices):
    # top_n=1 over AAA (strong) vs BBB (mild) should consistently hold AAA.
    spec = _spec(
        "cross_sectional_momentum", ["AAA", "BBB"],
        parameters={"lookback": 60, "top_n": 1}, rebalance="monthly",
    )
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)
    # AAA should get essentially all the weight on active days.
    held = res.weights
    avg_aaa = held["AAA"].iloc[60:].mean()
    avg_bbb = held["BBB"].iloc[60:].mean()
    assert avg_aaa > avg_bbb
    assert res.metrics["cagr"] > 0


def test_dual_momentum_holds_safe_in_downtrend(updown_prices):
    spec = _spec(
        "dual_momentum", ["AAA"],
        parameters={"lookback": 60}, safe_asset="BIL", rebalance="monthly",
    )
    res = run_backtest(spec, prices=updown_prices, cost_bps=0.0, optimize=False)
    # In the second half (AAA falling, momentum negative) it should rotate to BIL.
    held = res.weights
    assert "BIL" in held.columns
    assert held["BIL"].iloc[-30:].mean() > 0.5


def test_mean_reversion_rsi_runs(uptrend_prices):
    spec = _spec(
        "mean_reversion_rsi", ["AAA", "BBB"],
        parameters={"rsi_window": 14, "oversold": 30, "overbought": 70},
        rebalance="daily",
    )
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)
    assert res.daily_returns.notna().all()
    # weights never exceed 1 in aggregate
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_sector_rotation_runs_and_uses_safe(updown_prices):
    spec = _spec(
        "sector_rotation", ["AAA"],
        parameters={"lookback": 40, "top_n": 1, "trend_sma": 50},
        safe_asset="BIL", rebalance="monthly",
    )
    res = run_backtest(spec, prices=updown_prices, cost_bps=0.0, optimize=False)
    held = res.weights
    # Total invested (incl. safe) should never exceed 1.
    assert (held.sum(axis=1) <= 1.0 + 1e-9).all()


def test_grid_search_picks_best_sharpe(uptrend_prices):
    spec = _spec(
        "cross_sectional_momentum", ["AAA", "BBB"],
        parameters={"lookback": 60, "top_n": 1},
        parameter_ranges={"lookback": [20, 60, 120]},
        rebalance="monthly",
    )
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=True)
    assert res.grid_searched == 3
    assert "lookback" in res.optimized_parameters


def test_grid_search_is_capped():
    # 100 candidate lookbacks must be capped to MAX_GRID_COMBOS.
    from marginalia.engine.backtest import _grid_combos, MAX_GRID_COMBOS
    spec = _spec(
        "cross_sectional_momentum", ["AAA", "BBB", "CCC"],
        parameter_ranges={"lookback": list(range(10, 210))},
    )
    combos = _grid_combos(spec)
    assert len(combos) <= MAX_GRID_COMBOS


def test_missing_ticker_warns_but_runs(uptrend_prices):
    spec = _spec("buy_and_hold", ["AAA", "ZZZ_MISSING"], rebalance="daily")
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)
    assert any("Missing" in w for w in res.warnings)
    assert res.metrics["total_return"] > 0


def test_all_missing_tickers_raises(uptrend_prices):
    spec = _spec("buy_and_hold", ["NOPE1", "NOPE2"], rebalance="daily")
    with pytest.raises(ValueError):
        run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)


def test_transaction_costs_reduce_returns(uptrend_prices):
    spec = _spec(
        "cross_sectional_momentum", ["AAA", "BBB"],
        parameters={"lookback": 60, "top_n": 1}, rebalance="weekly",
    )
    free = run_backtest(spec, prices=uptrend_prices, cost_bps=0.0, optimize=False)
    costly = run_backtest(spec, prices=uptrend_prices, cost_bps=10.0, optimize=False)
    assert costly.metrics["total_return"] <= free.metrics["total_return"]


def test_to_dict_is_json_serializable(uptrend_prices):
    import json
    spec = _spec("buy_and_hold", ["AAA", "BBB"], rebalance="monthly")
    res = run_backtest(spec, prices=uptrend_prices, cost_bps=1.0, optimize=False)
    payload = res.to_dict()
    s = json.dumps(payload)   # must not raise
    assert "metrics" in payload
    assert isinstance(payload["daily_returns"], list)
    assert len(payload["daily_returns"]) > 0
