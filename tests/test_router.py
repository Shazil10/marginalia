import json

import pytest

from marginalia.agents.router import route_paper, RouterError, RouteDecision, MIN_CONFIDENCE
from marginalia.engine.registry import get_registry


def _chat(payload):
    def _inner(_system, _user):
        return json.dumps(payload)
    return _inner


def test_confident_match():
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({
            "matched_template": "sector_rotation",
            "confidence": 0.88,
            "category": "momentum",
            "reason": "Rotates sector ETFs by momentum with trend filter.",
            "alternatives": [{"template": "cross_sectional_momentum", "confidence": 0.6}],
            "needs_new_template": False,
        }),
    )
    assert decision.matched_template == "sector_rotation"
    assert decision.is_confident_match
    assert decision.alternatives[0]["template"] == "cross_sectional_momentum"


def test_hallucinated_template_id_rejected():
    # Model returns a template id that doesn't exist -> treated as no match / needs new.
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({
            "matched_template": "quantum_flux_capacitor",
            "confidence": 0.95,
            "needs_new_template": False,
        }),
    )
    assert decision.matched_template is None
    assert decision.needs_new_template
    assert not decision.is_confident_match


def test_low_confidence_not_confident_match():
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({
            "matched_template": "buy_and_hold",
            "confidence": 0.2,
            "needs_new_template": False,
        }),
    )
    assert decision.matched_template == "buy_and_hold"
    assert decision.confidence < MIN_CONFIDENCE
    assert not decision.is_confident_match


def test_needs_new_template_with_proposal():
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({
            "matched_template": None,
            "confidence": 0.1,
            "reason": "Options volatility selling, no template fits.",
            "needs_new_template": True,
            "proposed_template": {
                "id": "options_carry",
                "category": "volatility",
                "display_name": "Options Carry",
                "description": "Sell options to harvest vol premium.",
            },
        }),
    )
    assert decision.matched_template is None
    assert decision.needs_new_template
    assert decision.proposed_template["id"] == "options_carry"
    assert not decision.is_confident_match


def test_alternatives_filtered_to_real_templates():
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({
            "matched_template": "cross_sectional_momentum",
            "confidence": 0.7,
            "alternatives": [
                {"template": "made_up_thing", "confidence": 0.5},
                {"template": "dual_momentum", "confidence": 0.4},
            ],
            "needs_new_template": False,
        }),
    )
    ids = [a["template"] for a in decision.alternatives]
    assert "made_up_thing" not in ids
    assert "dual_momentum" in ids


def test_short_text_raises():
    with pytest.raises(RouterError):
        route_paper("tiny", chat_fn=_chat({}))


def test_garbage_output_raises():
    def _bad(_s, _u):
        return "I have opinions but no JSON."
    with pytest.raises(RouterError):
        route_paper("A" * 200, chat_fn=_bad)


def test_to_dict_serializable():
    decision = route_paper(
        "A" * 200,
        chat_fn=_chat({"matched_template": "buy_and_hold", "confidence": 0.6, "needs_new_template": False}),
    )
    json.dumps(decision.to_dict())
