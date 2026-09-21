"""Deterministic performance metrics from a daily return series.

All functions are defensive: short, empty, constant, or all-zero return series
return finite numbers (0.0) rather than NaN/inf, so downstream JSON and ranking
never break.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or b is None or np.isnan(b):
        return default
    out = a / b
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def compute_metrics(returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """Compute a standard metrics dict from a daily simple-return series."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return _empty_metrics()

    years = n / TRADING_DAYS
    cumul = (1.0 + r).cumprod()
    total_return = float(cumul.iloc[-1] - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and total_return > -1 else 0.0

    vol = float(r.std())
    ann_vol = vol * np.sqrt(TRADING_DAYS)
    sharpe = _safe_div(float(r.mean()), vol) * np.sqrt(TRADING_DAYS)

    downside = r[r < 0]
    down_vol = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = _safe_div(float(r.mean()) * TRADING_DAYS, down_vol * np.sqrt(TRADING_DAYS))

    running_max = cumul.cummax()
    drawdown = (cumul - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    calmar = _safe_div(cagr, abs(max_dd))
    win_rate = float((r > 0).sum()) / n

    out = {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "n_days": n,
        "years": round(years, 2),
        "start_date": r.index[0].strftime("%Y-%m-%d"),
        "end_date": r.index[-1].strftime("%Y-%m-%d"),
    }

    if benchmark is not None:
        out.update(_benchmark_metrics(r, benchmark))
    return out


def _benchmark_metrics(r: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    ab = pd.DataFrame({"s": r, "b": pd.Series(benchmark)}).dropna()
    if len(ab) < 30:
        return {}
    # A flat (zero-variance) strategy or benchmark makes beta/correlation undefined.
    if ab["b"].std() == 0 or ab["s"].std() == 0:
        return {}
    beta, alpha = np.polyfit(ab["b"], ab["s"], 1)
    bench_total = float((1.0 + ab["b"]).prod() - 1.0)
    bench_years = len(ab) / TRADING_DAYS
    bench_cagr = (
        (1.0 + bench_total) ** (1.0 / bench_years) - 1.0
        if bench_years > 0 and bench_total > -1
        else 0.0
    )
    strat_total = float((1.0 + ab["s"]).prod() - 1.0)
    strat_cagr = (
        (1.0 + strat_total) ** (1.0 / bench_years) - 1.0
        if bench_years > 0 and strat_total > -1
        else 0.0
    )
    return {
        "alpha": round(float(alpha) * TRADING_DAYS, 6),
        "beta": round(float(beta), 4),
        "correlation": round(float(ab["s"].corr(ab["b"])), 4),
        "benchmark_cagr": round(bench_cagr, 6),
        "benchmark_total_return": round(bench_total, 6),
        "excess_cagr": round(strat_cagr - bench_cagr, 6),
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "ann_vol": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "n_days": 0,
        "years": 0.0,
        "start_date": None,
        "end_date": None,
    }
