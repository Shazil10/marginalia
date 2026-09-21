"""Intake agent: natural-language investment goals -> structured risk profile.

Ported to the Nebius client. Pure structured extraction at low temperature.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from marginalia import llm

SYSTEM_PROMPT = """You convert a person's freeform investment goals into a structured risk profile JSON.

Extract:
- max_drawdown_tolerance: float 0.0-1.0
- investment_horizon_years: integer
- capital: float (USD)
- target_annual_return: float 0.0-1.0
- risk_class: one of "conservative", "moderate-conservative", "moderate", "aggressive"
- excluded_sectors: list of strings
- benchmark: string (default "SPY")
- clarification_needed: string or null

Mapping:
- fear of loss/crash -> conservative (drawdown 0.10, target 0.07)
- "some risk is okay" -> moderate-conservative (0.20, 0.10)
- growth, tolerates swings -> moderate (0.30, 0.13)
- max returns regardless of loss -> aggressive (0.50, 0.18)

Defaults: capital 10000, horizon 10. Infer target/drawdown from risk_class if unstated.
Return ONLY valid JSON, no markdown, no prose."""


class IntakeError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise IntakeError("No JSON object in intake output")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise IntakeError(f"Invalid JSON: {exc}") from exc


_DEFAULTS = {
    "capital": 10000.0,
    "investment_horizon_years": 10,
    "risk_class": "moderate",
    "max_drawdown_tolerance": 0.30,
    "target_annual_return": 0.13,
    "excluded_sectors": [],
    "benchmark": "SPY",
    "clarification_needed": None,
}


def parse_risk_profile(user_input: str, *, chat_fn: Optional[Callable] = None) -> dict:
    """Return a risk-profile dict, filling sane defaults for any missing keys."""
    if not user_input or not user_input.strip():
        raise IntakeError("Empty user input")

    if chat_fn is None:
        raw = llm.chat(SYSTEM_PROMPT, user_input, tier=llm.ModelTier.FAST, temperature=0.1)
    else:
        raw = chat_fn(SYSTEM_PROMPT, user_input)

    data = _extract_json(raw)
    profile = dict(_DEFAULTS)
    profile.update({k: v for k, v in data.items() if k in _DEFAULTS})
    return profile
