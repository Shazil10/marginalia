"""End-to-end strategy pipeline.

    PDF / paper text  ->  extract_spec (LLM)  ->  StrategySpec (validated)
                      ->  run_backtest (deterministic)  ->  BacktestResult

The LLM step is isolated and optional-to-mock, so the whole flow is testable
offline. The numeric backtest is fully deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from marginalia.agents.extraction import extract_spec
from marginalia.engine.backtest import BacktestResult, run_backtest
from marginalia.engine.spec import StrategySpec
from marginalia.ingest.pdf import extract_text_from_pdf


@dataclass
class PipelineOutput:
    spec: StrategySpec
    backtest: BacktestResult
    paper_text_chars: int

    def summary(self) -> dict:
        m = self.backtest.metrics
        return {
            "strategy_name": self.spec.strategy_name,
            "template": self.spec.template.value,
            "source_paper": self.spec.source_paper,
            "universe": self.spec.universe,
            "rebalance": self.spec.rebalance.value,
            "optimized_parameters": self.backtest.optimized_parameters,
            "cagr": m.get("cagr"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "total_return": m.get("total_return"),
            "win_rate": m.get("win_rate"),
            "years": m.get("years"),
            "warnings": self.backtest.warnings,
        }


def analyze_text(
    paper_text: str,
    *,
    start=None,
    end=None,
    prices=None,
    cost_bps: float = 1.0,
    optimize: bool = True,
    chat_fn: Optional[Callable] = None,
) -> PipelineOutput:
    """Run the pipeline from raw paper text."""
    spec = extract_spec(paper_text, chat_fn=chat_fn)
    result = run_backtest(
        spec, start=start, end=end, prices=prices, cost_bps=cost_bps, optimize=optimize
    )
    return PipelineOutput(spec=spec, backtest=result, paper_text_chars=len(paper_text))


def analyze_paper(
    pdf_path: str,
    *,
    start=None,
    end=None,
    prices=None,
    cost_bps: float = 1.0,
    optimize: bool = True,
    chat_fn: Optional[Callable] = None,
    max_chars: int = 8000,
) -> PipelineOutput:
    """Run the pipeline from a PDF file path."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)
    text = extract_text_from_pdf(pdf_path, max_chars=max_chars)
    if not text:
        raise ValueError(f"Could not extract any text from {pdf_path}")
    return analyze_text(
        text, start=start, end=end, prices=prices,
        cost_bps=cost_bps, optimize=optimize, chat_fn=chat_fn,
    )


def backtest_spec_dict(
    spec_dict: dict,
    *,
    start=None,
    end=None,
    prices=None,
    cost_bps: float = 1.0,
    optimize: bool = True,
) -> BacktestResult:
    """Convenience: validate a raw spec dict and backtest it (no LLM involved)."""
    spec = StrategySpec.from_llm_dict(spec_dict)
    return run_backtest(
        spec, start=start, end=end, prices=prices, cost_bps=cost_bps, optimize=optimize
    )
