"""LLM-backed agents that emit validated, structured outputs."""

from marginalia.agents.extraction import extract_spec, ExtractionError
from marginalia.agents.intake import parse_risk_profile
from marginalia.agents.router import route_paper, RouteDecision, RouterError

__all__ = [
    "extract_spec",
    "ExtractionError",
    "parse_risk_profile",
    "route_paper",
    "RouteDecision",
    "RouterError",
]
