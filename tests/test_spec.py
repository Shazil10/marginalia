import pytest

from marginalia.engine.spec import (
    StrategySpec,
    StrategyTemplate,
    RebalanceFrequency,
    SpecValidationError,
    default_parameters,
)


def test_minimal_spec_gets_defaults():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "Test Momentum",
        "template": "cross_sectional_momentum",
        "universe": ["spy", "qqq", "iwm"],
    })
    assert spec.universe == ["SPY", "QQQ", "IWM"]
    assert spec.rebalance == RebalanceFrequency.MONTHLY
    # defaults filled in
    assert spec.parameters["lookback"] == 126.0
    assert spec.parameters["top_n"] == 3.0


def test_universe_dedup_and_clean():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "X",
        "template": "buy_and_hold",
        "universe": ["aaa", "AAA", " bbb ", ""],
    })
    assert spec.universe == ["AAA", "BBB"]


def test_top_n_clamped_to_universe_size():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "X",
        "template": "cross_sectional_momentum",
        "universe": ["AAA", "BBB"],
        "parameters": {"top_n": 10},
    })
    assert spec.parameters["top_n"] == 2.0


def test_invalid_template_raises():
    with pytest.raises(SpecValidationError):
        StrategySpec.from_llm_dict({
            "strategy_name": "X",
            "template": "not_a_real_template",
            "universe": ["AAA"],
        })


def test_empty_universe_raises():
    with pytest.raises(SpecValidationError):
        StrategySpec.from_llm_dict({
            "strategy_name": "X",
            "template": "buy_and_hold",
            "universe": [],
        })


def test_non_dict_raises():
    with pytest.raises(SpecValidationError):
        StrategySpec.from_llm_dict("not a dict")


def test_unknown_params_dropped_known_kept():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "X",
        "template": "time_series_momentum",
        "universe": ["AAA"],
        "parameters": {"lookback": 50, "garbage_param": 999},
        "parameter_ranges": {"sma_window": [100, 150], "junk": [1, 2]},
    })
    assert spec.parameters["lookback"] == 50.0
    assert "garbage_param" not in spec.parameters
    assert "sma_window" in spec.parameter_ranges
    assert "junk" not in spec.parameter_ranges


def test_parameter_ranges_coerced_sorted_deduped():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "X",
        "template": "cross_sectional_momentum",
        "universe": ["AAA", "BBB", "CCC", "DDD"],
        "parameter_ranges": {"lookback": ["60", 20, 20, 120]},
    })
    assert spec.parameter_ranges["lookback"] == [20.0, 60.0, 120.0]


def test_all_tickers_includes_benchmark_and_safe():
    spec = StrategySpec.from_llm_dict({
        "strategy_name": "X",
        "template": "dual_momentum",
        "universe": ["AAA"],
        "benchmark": "SPY",
        "safe_asset": "BIL",
    })
    assert set(spec.all_tickers()) == {"AAA", "SPY", "BIL"}


def test_default_parameters_cover_all_templates():
    for t in StrategyTemplate:
        # Should not raise and should be a dict
        assert isinstance(default_parameters(t), dict)
