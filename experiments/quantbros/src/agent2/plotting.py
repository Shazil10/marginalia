"""Plot helpers for artifact generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_curve(
    *,
    path: str | Path,
    gross_equity: pd.Series,
    net_equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "Agent 2 Backtest Equity Curve",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    gross_equity.plot(ax=ax, label="Gross")
    net_equity.plot(ax=ax, label="Net")
    if benchmark_equity is not None:
        benchmark_equity.plot(ax=ax, label="Benchmark", linestyle="--", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
