"""FastAPI application exposing the Marginalia engine.

Endpoints
---------
GET  /health                 liveness + registry summary
GET  /templates              the (runnable) template catalog
GET  /ping                   Nebius connectivity check (uses LLM budget)
POST /backtest-spec          deterministic backtest of a provided spec (no LLM)
POST /intake                 natural language -> risk profile (LLM)
POST /route                  classify a paper against the template catalog (LLM)
POST /analyze-paper          full pipeline: route -> extract -> backtest -> size (LLM)

Deterministic endpoints (/backtest-spec, /templates, /health) need no API budget
and are safe to hit in CI/tests. LLM endpoints call Nebius.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from marginalia.engine.registry import get_registry
from marginalia.engine.spec import SpecValidationError, StrategySpec
from marginalia.engine.backtest import run_backtest


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class BacktestSpecRequest(BaseModel):
    spec: Dict[str, Any] = Field(..., description="A StrategySpec-shaped dict")
    start: Optional[str] = None
    end: Optional[str] = None
    cost_bps: float = 1.0
    optimize: bool = True


class IntakeRequest(BaseModel):
    user_input: str


class PaperRequest(BaseModel):
    paper_text: str
    user_input: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    cost_bps: float = 1.0
    optimize: bool = True


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Marginalia API",
        version="0.1.0",
        description="Turn academic trading-strategy papers into validated, backtested strategies.",
    )

    @app.get("/health")
    def health() -> dict:
        reg = get_registry()
        return {
            "status": "ok",
            "registry_version": reg.version,
            "runnable_templates": reg.runnable_ids(),
            "n_templates": len(reg.all()),
        }

    @app.get("/templates")
    def templates(runnable_only: bool = True) -> dict:
        reg = get_registry()
        return {"templates": reg.catalog(runnable_only=runnable_only)}

    @app.get("/ping")
    def ping() -> dict:
        from marginalia.llm import ping as llm_ping, ModelTier, LLMError
        out = {}
        for tier in (ModelTier.FAST, ModelTier.POWER):
            try:
                out[tier.value] = llm_ping(tier)
            except LLMError as exc:
                out[tier.value] = f"ERROR: {exc}"
        return out

    @app.post("/backtest-spec")
    def backtest_spec(req: BacktestSpecRequest) -> dict:
        try:
            spec = StrategySpec.from_llm_dict(req.spec)
        except SpecValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid spec: {exc}")
        try:
            result = run_backtest(
                spec, start=req.start, end=req.end,
                cost_bps=req.cost_bps, optimize=req.optimize,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Backtest failed: {exc}")
        return result.to_dict()

    @app.post("/intake")
    def intake(req: IntakeRequest) -> dict:
        from marginalia.agents.intake import parse_risk_profile, IntakeError
        from marginalia.llm import LLMError
        try:
            return parse_risk_profile(req.user_input)
        except (IntakeError, LLMError) as exc:
            raise HTTPException(status_code=502, detail=f"Intake failed: {exc}")

    @app.post("/route")
    def route(req: PaperRequest) -> dict:
        from marginalia.agents.router import route_paper, RouterError
        from marginalia.llm import LLMError
        try:
            decision = route_paper(req.paper_text)
        except (RouterError, LLMError) as exc:
            raise HTTPException(status_code=502, detail=f"Routing failed: {exc}")
        return decision.to_dict()

    @app.post("/analyze-paper")
    def analyze_paper(req: PaperRequest) -> dict:
        from marginalia.graph.pipeline import run_pipeline
        from marginalia.llm import LLMError
        try:
            result = run_pipeline(
                req.paper_text,
                user_input=req.user_input,
                start=req.start,
                end=req.end,
                cost_bps=req.cost_bps,
                optimize=req.optimize,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}")
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail={"errors": result.get("errors", [])})
        return result

    return app


app = create_app()
