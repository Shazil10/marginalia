"""LangGraph StateGraph wiring all Marginalia agents together.

Flow:

    intake (optional) -> router -> [confident match?]
        yes -> extraction -> backtest -> sizing -> END
        no  -> END (status="needs_new_template")

The router gate is the key agentic decision: we only spend compute on extraction
+ backtest when a paper confidently matches a *runnable* template. Papers that
don't fit are surfaced (with the router's proposal) rather than silently forced
into a wrong template.

LLM access is injected via ``chat_fn`` so the whole graph runs offline in tests.
``prices`` can be injected to backtest without hitting the network.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from marginalia.agents.extraction import extract_spec, ExtractionError
from marginalia.agents.intake import parse_risk_profile, IntakeError
from marginalia.agents.router import route_paper, RouterError
from marginalia.engine.backtest import run_backtest
from marginalia.engine.registry import TemplateRegistry, get_registry
from marginalia.engine.spec import StrategySpec
from marginalia.graph.state import AgentState, new_state
from marginalia.portfolio.ranking import size_position


def _make_intake_node(chat_fn):
    def intake_node(state: AgentState) -> dict:
        user_input = state.get("user_input")
        if not user_input:
            return {}  # nothing to do; skip
        try:
            profile = parse_risk_profile(user_input, chat_fn=chat_fn)
            return {"risk_profile": profile}
        except IntakeError as exc:
            return {"warnings": state.get("warnings", []) + [f"intake: {exc}"]}
    return intake_node


def _make_router_node(chat_fn, registry: TemplateRegistry):
    def router_node(state: AgentState) -> dict:
        try:
            decision = route_paper(
                state["paper_text"], registry=registry, chat_fn=chat_fn
            )
        except RouterError as exc:
            return {"status": "error", "errors": state.get("errors", []) + [f"router: {exc}"]}
        update = {"route": decision.to_dict()}
        if not decision.is_confident_match:
            update["status"] = "needs_new_template"
        return update
    return router_node


def _make_extraction_node(chat_fn, registry: TemplateRegistry):
    def extraction_node(state: AgentState) -> dict:
        try:
            spec = extract_spec(state["paper_text"], chat_fn=chat_fn)
        except ExtractionError as exc:
            return {"status": "error", "errors": state.get("errors", []) + [f"extraction: {exc}"]}

        # Honor the router's confident match if extraction disagreed on template.
        route = state.get("route") or {}
        matched = route.get("matched_template")
        warnings = list(state.get("warnings", []))
        if matched and matched != spec.template.value and registry.is_runnable(matched):
            warnings.append(
                f"extraction chose '{spec.template.value}' but router matched "
                f"'{matched}'; using router match."
            )
            data = spec.model_dump()
            data["template"] = matched
            try:
                spec = StrategySpec.from_llm_dict(data)
            except Exception:
                pass  # keep extraction's spec if re-coercion fails
        return {"spec": spec.model_dump(mode="json"), "warnings": warnings}
    return extraction_node


def _make_backtest_node(prices: Optional[pd.DataFrame]):
    def backtest_node(state: AgentState) -> dict:
        spec_dict = state.get("spec")
        if not spec_dict:
            return {"status": "error", "errors": state.get("errors", []) + ["backtest: no spec"]}
        try:
            spec = StrategySpec.from_llm_dict(spec_dict)
            result = run_backtest(
                spec,
                start=state.get("start"),
                end=state.get("end"),
                prices=prices,
                cost_bps=state.get("cost_bps", 1.0),
                optimize=state.get("optimize", True),
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "errors": state.get("errors", []) + [f"backtest: {exc}"]}
        return {
            "backtest": result.to_dict(),
            "warnings": state.get("warnings", []) + result.warnings,
        }
    return backtest_node


def _returns_from_backtest(bt: dict) -> Optional[pd.Series]:
    """Reconstruct a daily-return Series from the serialized backtest dict."""
    pairs = bt.get("daily_returns") or []
    if len(pairs) < 2:
        return None
    idx = pd.to_datetime([p[0] for p in pairs])
    vals = [float(p[1]) for p in pairs]
    return pd.Series(vals, index=idx)


def _sizing_node(state: AgentState) -> dict:
    bt = state.get("backtest")
    if not bt:
        return {}
    daily = _returns_from_backtest(bt)
    if daily is None:
        return {}
    profile = state.get("risk_profile") or {}
    sizing = size_position(
        daily,
        capital=float(profile.get("capital", 10000.0)),
        risk_class=profile.get("risk_class"),
        drawdown_tolerance=profile.get("max_drawdown_tolerance"),
    )
    return {"sizing": sizing}


def _route_gate(state: AgentState) -> str:
    """Conditional edge after the router."""
    if state.get("status") in ("needs_new_template", "error"):
        return "stop"
    return "continue"


def build_graph(
    *,
    chat_fn: Optional[Callable] = None,
    prices: Optional[pd.DataFrame] = None,
    registry: Optional[TemplateRegistry] = None,
):
    """Compile and return the LangGraph app.

    ``chat_fn`` injects the LLM (None -> live Nebius). ``prices`` injects market
    data (None -> live fetch). ``registry`` defaults to the on-disk catalog.
    """
    from langgraph.graph import StateGraph, END

    registry = registry or get_registry()

    g = StateGraph(AgentState)
    g.add_node("intake", _make_intake_node(chat_fn))
    g.add_node("router", _make_router_node(chat_fn, registry))
    g.add_node("extraction", _make_extraction_node(chat_fn, registry))
    g.add_node("backtest", _make_backtest_node(prices))
    g.add_node("sizing", _sizing_node)

    g.set_entry_point("intake")
    g.add_edge("intake", "router")
    g.add_conditional_edges(
        "router", _route_gate, {"continue": "extraction", "stop": END}
    )
    g.add_edge("extraction", "backtest")
    g.add_edge("backtest", "sizing")
    g.add_edge("sizing", END)

    return g.compile()


def run_pipeline(
    paper_text: str,
    *,
    user_input: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cost_bps: float = 1.0,
    optimize: bool = True,
    chat_fn: Optional[Callable] = None,
    prices: Optional[pd.DataFrame] = None,
    registry: Optional[TemplateRegistry] = None,
) -> dict:
    """Run the full graph and return a clean, JSON-serializable result dict."""
    app = build_graph(chat_fn=chat_fn, prices=prices, registry=registry)
    init = new_state(
        paper_text,
        user_input=user_input,
        start=start,
        end=end,
        cost_bps=cost_bps,
        optimize=optimize,
    )
    final = app.invoke(init)
    return dict(final)
