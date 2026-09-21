"""Extraction agent: academic paper text -> validated StrategySpec.

This is the ONLY place an LLM touches the strategy pipeline. It reads the paper
and emits JSON describing which deterministic template to use and with what
parameters. The output is immediately validated against ``StrategySpec``; if the
cheap model produces something invalid, we escalate to the powerful model once.

The LLM never writes executable code — it only fills a constrained schema.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from marginalia.engine.spec import StrategySpec, StrategyTemplate, SpecValidationError
from marginalia import llm


class ExtractionError(RuntimeError):
    pass


_TEMPLATE_GUIDE = "\n".join(f"  - {t.value}" for t in StrategyTemplate)

SYSTEM_PROMPT = f"""You are a quantitative analyst. Read the academic trading-strategy paper text and map it to ONE deterministic strategy template that our engine can execute. You do NOT write code. You only output JSON.

Choose exactly one template from this list (use the value verbatim):
{_TEMPLATE_GUIDE}

Template meanings:
  - buy_and_hold: passively hold the universe, equal weight.
  - time_series_momentum: hold each asset only while it is above its own moving average and has positive trailing return; else cash. Params: lookback, sma_window.
  - cross_sectional_momentum: rank the universe by trailing return, hold the top N equally. Params: lookback, top_n.
  - dual_momentum: hold the single best asset by trailing return, but only if its return is positive; otherwise hold the safe asset. Params: lookback.
  - mean_reversion_rsi: buy an asset when its RSI is oversold, sell when overbought. Params: rsi_window, oversold, overbought.
  - sector_rotation: rotate into the top N momentum assets that are also above a long trend SMA; park the rest in the safe asset. Params: lookback, top_n, trend_sma.

Return ONLY valid JSON (no markdown, no prose) with these keys:
{{
  "strategy_name": "short descriptive name",
  "template": "<one template value>",
  "universe": ["TICKER", ...],            // real, tradable tickers (ETFs/stocks). If the paper is generic, choose representative liquid ETFs.
  "benchmark": "SPY",
  "safe_asset": "BIL",                     // cash-like asset for downtrends
  "rebalance": "daily|weekly|monthly|quarterly",
  "parameters": {{ "lookback": 126, ... }},      // best-guess values
  "parameter_ranges": {{ "lookback": [63,126,252] }},  // optional small grids to search
  "source_paper": "author + year",
  "strategy_type": "momentum|mean_reversion|factor|trend_following|other",
  "notes": "one sentence on the core logic"
}}

Rules:
- universe MUST contain real tickers. Never leave it empty or use placeholders.
- Only include parameter names relevant to the chosen template.
- Keep parameter_ranges small (<= 4 values each)."""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    text = text.strip()
    # strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # find the outermost {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ExtractionError("No JSON object found in model output")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Invalid JSON from model: {exc}") from exc


def _is_valid_spec_text(text: str) -> bool:
    """Validator used to decide whether to escalate to the powerful model."""
    try:
        data = _extract_json(text)
        StrategySpec.from_llm_dict(data)
        return True
    except (ExtractionError, SpecValidationError):
        return False


def extract_spec(
    paper_text: str,
    *,
    chat_fn: Optional[Callable] = None,
) -> StrategySpec:
    """Extract a validated :class:`StrategySpec` from paper text.

    ``chat_fn`` lets tests inject a fake LLM. By default it uses the FAST tier and
    escalates to POWER if the FAST output fails validation.
    """
    if not paper_text or len(paper_text.strip()) < 50:
        raise ExtractionError("Paper text is empty or too short to extract a strategy")

    if chat_fn is None:
        raw = llm.chat_with_escalation(
            SYSTEM_PROMPT,
            f"Paper text:\n{paper_text}",
            validate=_is_valid_spec_text,
            temperature=0.1,
            max_tokens=2048,
        )
    else:
        raw = chat_fn(SYSTEM_PROMPT, f"Paper text:\n{paper_text}")

    data = _extract_json(raw)
    try:
        return StrategySpec.from_llm_dict(data)
    except SpecValidationError as exc:
        raise ExtractionError(f"Model output failed spec validation: {exc}") from exc
