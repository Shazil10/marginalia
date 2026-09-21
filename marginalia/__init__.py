"""Marginalia — deterministic quant research engine.

This package replaces the fragile "LLM writes a fresh backtest script per paper"
approach with a deterministic, tested pipeline:

    PDF -> structured StrategySpec (LLM) -> deterministic backtest engine -> metrics

The LLM is only used to read papers and emit a validated spec. All numerical
backtesting runs in vetted, deterministic Python.
"""

__version__ = "0.1.0"
