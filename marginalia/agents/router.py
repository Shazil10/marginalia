"""Template Router agent.

Before we extract parameters, we decide *which* strategy family a paper belongs
to. The router reads the paper text plus the (expandable) template catalog and
returns a structured decision:

    {
      "matched_template": "sector_rotation" | null,
      "confidence": 0.0-1.0,
      "category": "momentum",
      "reason": "...",
      "alternatives": [{"template": "...", "confidence": 0.x}, ...],
      "needs_new_template": bool,
      "proposed_template": {"id": "...", "category": "...", "display_name": "...",
                            "description": "..."} | null
    }

Key safety rule: the router may *propose* a new template (metadata only) when
nothing fits, but it can never invent executable code. A proposal is recorded for
human implementation; it does not become runnable automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from marginalia import llm
from marginalia.engine.registry import TemplateRegistry, get_registry

# Below this confidence we do not trust the match enough to backtest it.
MIN_CONFIDENCE = 0.45


class RouterError(RuntimeError):
    pass


@dataclass
class RouteDecision:
    matched_template: Optional[str]
    confidence: float
    category: str = "other"
    reason: str = ""
    alternatives: List[dict] = field(default_factory=list)
    needs_new_template: bool = False
    proposed_template: Optional[dict] = None

    @property
    def is_confident_match(self) -> bool:
        return (
            self.matched_template is not None
            and not self.needs_new_template
            and self.confidence >= MIN_CONFIDENCE
        )

    def to_dict(self) -> dict:
        return {
            "matched_template": self.matched_template,
            "confidence": round(float(self.confidence), 3),
            "category": self.category,
            "reason": self.reason,
            "alternatives": self.alternatives,
            "needs_new_template": self.needs_new_template,
            "proposed_template": self.proposed_template,
        }


def _system_prompt(registry: TemplateRegistry) -> str:
    return f"""You are a quantitative strategy classifier. Given the text of a trading-strategy paper, decide which ONE of our existing strategy templates best matches the paper's core logic.

Available templates (id [category] — description):
{registry.catalog_text(runnable_only=True)}

Return ONLY valid JSON (no markdown, no prose):
{{
  "matched_template": "<one template id from the list, or null if none fit>",
  "confidence": 0.0,                 // 0-1, how well the best template fits the paper
  "category": "<category of the matched template, or your best guess>",
  "reason": "one sentence justifying the choice",
  "alternatives": [                  // up to 2 other plausible templates
    {{"template": "<id>", "confidence": 0.0}}
  ],
  "needs_new_template": false,       // true ONLY if no existing template fits the strategy
  "proposed_template": null          // if needs_new_template, propose metadata only:
                                     // {{"id": "snake_case_id", "category": "...", "display_name": "...", "description": "..."}}
}}

Rules:
- matched_template MUST be one of the listed ids, or null.
- Set needs_new_template true only when the paper's mechanism genuinely cannot be expressed by any listed template (e.g. options selling, intraday, ML signals).
- Be honest with confidence: a loose thematic match is ~0.5, a strong mechanical match is >0.8.
- Output JSON only."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RouterError("No JSON object found in router output")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RouterError(f"Invalid JSON from router: {exc}") from exc


def _coerce_decision(data: dict, registry: TemplateRegistry) -> RouteDecision:
    if not isinstance(data, dict):
        raise RouterError(f"expected dict, got {type(data).__name__}")

    matched = data.get("matched_template")
    if isinstance(matched, str):
        matched = matched.strip() or None
    # Reject hallucinated ids that aren't actually runnable.
    if matched is not None and not registry.is_runnable(matched):
        matched = None

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    needs_new = bool(data.get("needs_new_template", False))
    # If the model gave no valid match, that implies a new template is needed.
    if matched is None:
        needs_new = True
        confidence = min(confidence, MIN_CONFIDENCE - 0.01) if confidence >= MIN_CONFIDENCE else confidence

    alternatives = []
    for alt in data.get("alternatives", []) or []:
        if not isinstance(alt, dict):
            continue
        tid = alt.get("template")
        if isinstance(tid, str) and registry.is_runnable(tid):
            try:
                ac = max(0.0, min(float(alt.get("confidence", 0.0)), 1.0))
            except (TypeError, ValueError):
                ac = 0.0
            alternatives.append({"template": tid, "confidence": round(ac, 3)})

    proposed = data.get("proposed_template")
    if not isinstance(proposed, dict):
        proposed = None

    category = data.get("category")
    if not isinstance(category, str) or not category.strip():
        info = registry.get(matched) if matched else None
        category = info.category if info else "other"

    return RouteDecision(
        matched_template=matched,
        confidence=confidence,
        category=category,
        reason=str(data.get("reason", ""))[:500],
        alternatives=alternatives[:2],
        needs_new_template=needs_new,
        proposed_template=proposed,
    )


def route_paper(
    paper_text: str,
    *,
    registry: Optional[TemplateRegistry] = None,
    chat_fn: Optional[Callable] = None,
) -> RouteDecision:
    """Classify ``paper_text`` against the template catalog.

    ``chat_fn`` lets tests inject a fake LLM. By default uses the FAST tier with
    escalation to POWER if the FAST output is unparseable.
    """
    if not paper_text or len(paper_text.strip()) < 50:
        raise RouterError("Paper text is empty or too short to classify")

    registry = registry or get_registry()
    system = _system_prompt(registry)

    def _validate(text: str) -> bool:
        try:
            _extract_json(text)
            return True
        except RouterError:
            return False

    if chat_fn is None:
        raw = llm.chat_with_escalation(
            system, f"Paper text:\n{paper_text}",
            validate=_validate, temperature=0.1, max_tokens=1024,
        )
    else:
        raw = chat_fn(system, f"Paper text:\n{paper_text}")

    data = _extract_json(raw)
    return _coerce_decision(data, registry)
