import json

import pytest

from marginalia.agents.extraction import extract_spec, ExtractionError, _extract_json
from marginalia.engine.spec import StrategyTemplate


VALID_SPEC_JSON = json.dumps({
    "strategy_name": "Sector Momentum Rotation",
    "template": "sector_rotation",
    "universe": ["XLK", "XLV", "XLF", "XLE"],
    "benchmark": "SPY",
    "safe_asset": "BIL",
    "rebalance": "monthly",
    "parameters": {"lookback": 126, "top_n": 2, "trend_sma": 200},
    "parameter_ranges": {"lookback": [63, 126, 252]},
    "source_paper": "Giordano 2018",
    "strategy_type": "momentum",
    "notes": "Rotate into top sector ETFs above trend.",
})


def _fake_chat(_system, _user):
    return VALID_SPEC_JSON


def _fake_chat_with_markdown(_system, _user):
    return f"Here is the spec:\n```json\n{VALID_SPEC_JSON}\n```\nDone."


def _fake_chat_garbage(_system, _user):
    return "I think this paper is about momentum but I won't give JSON."


def test_extract_spec_clean_json():
    spec = extract_spec("A" * 200, chat_fn=_fake_chat)
    assert spec.template == StrategyTemplate.SECTOR_ROTATION
    assert spec.universe == ["XLK", "XLV", "XLF", "XLE"]
    assert spec.parameters["top_n"] == 2.0
    assert spec.parameter_ranges["lookback"] == [63.0, 126.0, 252.0]


def test_extract_spec_from_markdown_fence():
    spec = extract_spec("A" * 200, chat_fn=_fake_chat_with_markdown)
    assert spec.template == StrategyTemplate.SECTOR_ROTATION


def test_extract_spec_garbage_raises():
    with pytest.raises(ExtractionError):
        extract_spec("A" * 200, chat_fn=_fake_chat_garbage)


def test_extract_spec_short_text_raises():
    with pytest.raises(ExtractionError):
        extract_spec("tiny", chat_fn=_fake_chat)


def test_extract_json_handles_surrounding_prose():
    data = _extract_json('blah blah {"a": 1, "b": [2,3]} trailing')
    assert data == {"a": 1, "b": [2, 3]}
