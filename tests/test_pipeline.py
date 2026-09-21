import json

import pytest

from marginalia.pipeline import analyze_text, backtest_spec_dict


def _fake_chat_factory(spec_dict):
    def _chat(_system, _user):
        return json.dumps(spec_dict)
    return _chat


def test_analyze_text_end_to_end(uptrend_prices):
    spec_dict = {
        "strategy_name": "Cross Sectional Momo",
        "template": "cross_sectional_momentum",
        "universe": ["AAA", "BBB"],
        "benchmark": "SPY",
        "rebalance": "monthly",
        "parameters": {"lookback": 60, "top_n": 1},
    }
    out = analyze_text(
        "Paper text " * 50,
        prices=uptrend_prices,
        optimize=False,
        chat_fn=_fake_chat_factory(spec_dict),
    )
    assert out.spec.template.value == "cross_sectional_momentum"
    s = out.summary()
    assert s["cagr"] is not None
    assert s["sharpe"] is not None
    assert "AAA" in out.spec.universe


def test_backtest_spec_dict_no_llm(uptrend_prices):
    res = backtest_spec_dict(
        {
            "strategy_name": "BH",
            "template": "buy_and_hold",
            "universe": ["AAA", "BBB"],
            "rebalance": "monthly",
        },
        prices=uptrend_prices,
        cost_bps=0.0,
    )
    assert res.metrics["total_return"] > 0
    payload = res.to_dict()
    json.dumps(payload)  # serializable


def test_analyze_text_propagates_invalid_spec(uptrend_prices):
    bad = {"strategy_name": "X", "template": "not_real", "universe": ["AAA"]}
    from marginalia.agents.extraction import ExtractionError
    with pytest.raises(ExtractionError):
        analyze_text(
            "Paper text " * 50,
            prices=uptrend_prices,
            chat_fn=_fake_chat_factory(bad),
        )
