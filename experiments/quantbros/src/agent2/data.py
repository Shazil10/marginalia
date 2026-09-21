"""Market data planning and fetch utilities."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from agent2.agents.models import StrategyBlueprint
from agent2.schemas import Agent1ToAgent2Input, DataPlan, LogicSpec

try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover
    yf = None
    _YF_ERR = e
else:
    _YF_ERR = None


DEFAULT_START = date(2010, 1, 1)


def require_yfinance() -> None:
    if yf is None:
        raise ImportError(
            "yfinance is required for historical market data. Install with: pip install -e '.[data]'"
        ) from _YF_ERR


def build_data_plan(
    *,
    agent1: Agent1ToAgent2Input,
    logic: LogicSpec,
    start: date,
    end: date,
    paper_category: str | None = None,
    asset_universe_notes: str | None = None,
    regime_claim: str | None = None,
) -> DataPlan:
    tickers = list(dict.fromkeys(logic.universe.tickers))
    extra = {
        "paper_category": paper_category or agent1.paper_category.value,
        "asset_universe_notes": asset_universe_notes if asset_universe_notes is not None else agent1.asset_universe_notes,
        "regime_claim": regime_claim or agent1.regime_claim.value,
    }
    return DataPlan(
        tickers=tickers,
        benchmark=logic.universe.benchmark,
        start=start,
        end=end,
        extra=extra,
    )


def build_data_plan_from_blueprint(
    *,
    agent1: Agent1ToAgent2Input,
    blueprint: StrategyBlueprint,
    start: date,
    end: date,
) -> DataPlan:
    tickers = list(dict.fromkeys(blueprint.universe.tickers))
    benchmark = blueprint.benchmark or blueprint.universe.benchmark
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)
    requirement = blueprint.data_requirements[0] if blueprint.data_requirements else None
    frequency = requirement.frequency if requirement is not None else "1d"
    required_series = ["ohlcv"]
    extra = {
        "reader_paper_category": agent1.paper_category.value,
        "reader_asset_universe_notes": agent1.asset_universe_notes,
        "reader_regime_claim": agent1.regime_claim.value,
        "signal_definition": blueprint.signal_definition,
        "rebalance_schedule": blueprint.rebalance_schedule,
        "holding_period": blueprint.holding_period,
        "timing_and_delay_rules": blueprint.timing_and_delay_rules,
        "required_parameters": blueprint.required_parameters,
        "open_questions": blueprint.open_questions,
    }
    return DataPlan(
        tickers=tickers,
        benchmark=benchmark,
        required_series=required_series,
        frequency=frequency,
        start=start,
        end=end,
        extra=extra,
    )


def fetch_adj_close(
    tickers: list[str],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Return a wide adjusted-close panel for the requested ticker universe.

    yfinance treats the `end` date as exclusive, so the request extends by one day.
    """

    require_yfinance()
    if not tickers:
        raise ValueError("At least one ticker is required")
    download_end = end + timedelta(days=1)
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=download_end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No price data for {tickers} between {start} and {end}")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data.iloc[:, : len(tickers)]
    else:
        prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise KeyError(f"Missing downloaded series for tickers: {missing}")
    return prices[tickers]
