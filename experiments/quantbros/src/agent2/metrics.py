"""Performance metrics and helpers for tear sheets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent2.schemas import MetricBlock


def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if equity.empty:
        return float("nan")
    years = len(equity) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 252.0, rf: float = 0.0) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    excess = clean - rf / periods_per_year
    sigma = float(excess.std(ddof=1))
    if sigma == 0.0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / sigma)


def volatility_annualized(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    cagr_value = cagr(equity, periods_per_year)
    mdd = max_drawdown(equity)
    if np.isnan(cagr_value) or np.isnan(mdd) or mdd >= 0.0:
        return float("nan")
    return float(cagr_value / abs(mdd))


def turnover_mean(turnover: pd.Series) -> float:
    clean = turnover.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def exposure_mean(exposure: pd.Series) -> float:
    clean = exposure.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def win_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float((clean > 0).mean())


def build_metric_block(
    *,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
) -> MetricBlock:
    return MetricBlock(
        cagr=cagr(equity),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(equity),
        volatility_ann=volatility_annualized(returns),
        calmar=calmar_ratio(equity),
        total_return=total_return(equity),
        turnover_mean=turnover_mean(turnover),
        exposure_mean=exposure_mean(exposure),
        win_rate=win_rate(returns),
    )
