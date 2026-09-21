"""Shared state object threaded through the LangGraph pipeline.

Every node reads from and writes to this single ``AgentState`` dict. No node uses
global or class-level mutable state, so the graph is deterministic given its
inputs (the only nondeterminism is the LLM, which is injected and mockable).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # -- inputs --------------------------------------------------------------
    user_input: Optional[str]          # freeform investor goals (optional)
    paper_text: str                    # raw text of the strategy paper
    start: Optional[str]               # backtest window start (YYYY-MM-DD)
    end: Optional[str]                 # backtest window end
    cost_bps: float                    # transaction cost in basis points
    optimize: bool                     # run grid search?

    # -- intermediate products ----------------------------------------------
    risk_profile: Optional[Dict[str, Any]]   # from intake agent
    route: Optional[Dict[str, Any]]          # RouteDecision.to_dict()
    spec: Optional[Dict[str, Any]]           # StrategySpec (model_dump)
    backtest: Optional[Dict[str, Any]]       # BacktestResult.to_dict()
    sizing: Optional[Dict[str, Any]]         # Kelly position sizing

    # -- control / diagnostics ----------------------------------------------
    status: str                        # "ok" | "needs_new_template" | "error"
    errors: List[str]
    warnings: List[str]


def new_state(
    paper_text: str,
    *,
    user_input: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cost_bps: float = 1.0,
    optimize: bool = True,
) -> AgentState:
    """Construct an initial state with sane defaults."""
    return AgentState(
        user_input=user_input,
        paper_text=paper_text,
        start=start,
        end=end,
        cost_bps=cost_bps,
        optimize=optimize,
        risk_profile=None,
        route=None,
        spec=None,
        backtest=None,
        sizing=None,
        status="ok",
        errors=[],
        warnings=[],
    )
