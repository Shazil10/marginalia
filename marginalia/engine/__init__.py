"""Deterministic backtesting engine."""

from marginalia.engine.spec import (
    StrategySpec,
    StrategyTemplate,
    RebalanceFrequency,
    SpecValidationError,
)
from marginalia.engine.backtest import BacktestResult, run_backtest

__all__ = [
    "StrategySpec",
    "StrategyTemplate",
    "RebalanceFrequency",
    "SpecValidationError",
    "BacktestResult",
    "run_backtest",
]
