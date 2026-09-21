import json

import pytest

from marginalia.graph.pipeline import run_pipeline, build_graph


SECTOR_SPEC = {
    "strategy_name": "Sector Momentum Rotation",
    "template": "sector_rotation",
    "universe": ["AAA", "BBB"],
    "benchmark": "SPY",
    "safe_asset": "BIL",
    "rebalance": "monthly",
    "parameters": {"lookback": 40, "top_n": 1, "trend_sma": 50},
}

ROUTE_OK = {
    "matched_template": "sector_rotation",
    "confidence": 0.85,
    "category": "momentum",
    "reason": "rotation",
    "needs_new_template": False,
}


def _router_then_extractor(route_payload, spec_payload, intake_payload=None):
    """Fake chat_fn that returns the right JSON based on the system prompt content."""
    def _chat(system, _user):
        s = system.lower()
        if "classifier" in s or "matched_template" in s:
            return json.dumps(route_payload)
        if "risk profile" in s or "intake" in s:
            return json.dumps(intake_payload or {})
        # extraction
        return json.dumps(spec_payload)
    return _chat


def test_full_pipeline_confident_match(updown_prices):
    chat = _router_then_extractor(ROUTE_OK, SECTOR_SPEC)
    out = run_pipeline(
        "Paper text " * 50,
        prices=updown_prices,
        optimize=False,
        chat_fn=chat,
    )
    assert out["status"] == "ok"
    assert out["route"]["matched_template"] == "sector_rotation"
    assert out["spec"]["template"] == "sector_rotation"
    assert out["backtest"] is not None
    assert "metrics" in out["backtest"]


def test_pipeline_stops_on_needs_new_template(updown_prices):
    route_bad = {
        "matched_template": None,
        "confidence": 0.1,
        "needs_new_template": True,
        "proposed_template": {"id": "options_carry", "category": "volatility",
                              "display_name": "Options Carry", "description": "vol selling"},
    }
    chat = _router_then_extractor(route_bad, SECTOR_SPEC)
    out = run_pipeline("Paper text " * 50, prices=updown_prices, optimize=False, chat_fn=chat)
    assert out["status"] == "needs_new_template"
    # extraction/backtest should NOT have run
    assert out["spec"] is None
    assert out["backtest"] is None
    assert out["route"]["proposed_template"]["id"] == "options_carry"


def test_pipeline_with_intake(updown_prices):
    intake = {"risk_class": "conservative", "capital": 50000, "max_drawdown_tolerance": 0.10}
    chat = _router_then_extractor(ROUTE_OK, SECTOR_SPEC, intake_payload=intake)
    out = run_pipeline(
        "Paper text " * 50,
        user_input="I'm risk averse, $50k",
        prices=updown_prices,
        optimize=False,
        chat_fn=chat,
    )
    assert out["risk_profile"]["risk_class"] == "conservative"
    assert out["sizing"] is not None
    assert out["sizing"]["allocation_dollars"] <= 50000


def test_pipeline_sizing_present_on_success(updown_prices):
    chat = _router_then_extractor(ROUTE_OK, SECTOR_SPEC)
    out = run_pipeline("Paper text " * 50, prices=updown_prices, optimize=False, chat_fn=chat)
    assert out["sizing"] is not None
    assert "allocation_fraction" in out["sizing"]


def test_pipeline_result_json_serializable(updown_prices):
    chat = _router_then_extractor(ROUTE_OK, SECTOR_SPEC)
    out = run_pipeline("Paper text " * 50, prices=updown_prices, optimize=False, chat_fn=chat)
    json.dumps(out)  # must not raise


def test_router_extraction_template_disagreement(updown_prices):
    # Router confidently says sector_rotation; extraction picks buy_and_hold.
    # Pipeline should defer to the router's confident match.
    bh_spec = dict(SECTOR_SPEC)
    bh_spec["template"] = "buy_and_hold"
    chat = _router_then_extractor(ROUTE_OK, bh_spec)
    out = run_pipeline("Paper text " * 50, prices=updown_prices, optimize=False, chat_fn=chat)
    assert out["spec"]["template"] == "sector_rotation"
    assert any("router match" in w for w in out["warnings"])


def test_graph_compiles():
    app = build_graph(chat_fn=lambda s, u: "{}")
    assert app is not None
