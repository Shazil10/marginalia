import json

import pytest

from marginalia.agents.intake import parse_risk_profile, IntakeError


def _fake(profile):
    def _chat(_system, _user):
        return json.dumps(profile)
    return _chat


def test_parse_fills_defaults():
    prof = parse_risk_profile("I want growth", chat_fn=_fake({"risk_class": "aggressive"}))
    assert prof["risk_class"] == "aggressive"
    # defaults filled
    assert prof["benchmark"] == "SPY"
    assert prof["capital"] == 10000.0


def test_parse_respects_provided_values():
    prof = parse_risk_profile(
        "conservative, $50k",
        chat_fn=_fake({"risk_class": "conservative", "capital": 50000, "max_drawdown_tolerance": 0.10}),
    )
    assert prof["capital"] == 50000
    assert prof["max_drawdown_tolerance"] == 0.10


def test_empty_input_raises():
    with pytest.raises(IntakeError):
        parse_risk_profile("   ", chat_fn=_fake({}))


def test_ignores_unknown_keys():
    prof = parse_risk_profile("x", chat_fn=_fake({"risk_class": "moderate", "junk": 1}))
    assert "junk" not in prof
