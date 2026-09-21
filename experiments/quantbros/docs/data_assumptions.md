# Data Assumptions

This document captures the key simplifications used by Agent 2 so the backtests are explainable during the hackathon.

Agentic mode is the default execution path. Strategy interpretation now comes from OpenRouter-backed blueprint extraction and paper-specific code generation, while the deterministic layer below handles data fetch, metrics, plotting, and artifact management.

## Point-in-Time Rules

- Signals are computed using information available at close on day `t`.
- Portfolio weights are shifted by one bar.
- Returns applied to the strategy are from `t` to `t+1`.
- Same-bar execution is not allowed.

## Market Data

- Historical prices come from `yfinance`.
- Prices are fetched as adjusted close via `auto_adjust=True`.
- Benchmarks are only used when they are present in the paper metadata or blueprint.

## Missing Data

- Interior missing values are forward-filled.
- Leading NaNs remain until enough history exists for the strategy.
- Days with all assets missing are dropped.

## Universe Construction

- The current hackathon build uses static ticker lists.
- This means survivorship bias remains for broad universes.
- Asset universes inferred from paper text are approximations or ETF proxies, not point-in-time constituent databases.

## Frictions

- The preferred default is that generated strategy code models frictions directly.
- The deterministic layer records and evaluates whatever return series the generated code emits.
- If generated code omits `net_returns`, the metrics layer falls back to `gross_returns` and records a warning in the artifact bundle.

## Robustness Checks

- Single train/test split on executed return series
- Suspect flags for implausible Sharpe, high turnover, and excessive exposure
- Additional parameter sensitivity is only available if the generated strategy emits it in diagnostics

## PDF Extraction

- Primary extractor: `pypdf`
- Fallbacks:
  - `.txt` sidecar if present
  - `pdftotext`
  - OCR via `pdftoppm` + `tesseract` for image-heavy PDFs

## What This Is Not Yet

- No point-in-time fundamentals database
- No intraday execution simulation
- No borrow fees, financing costs, or market impact model
- No production-grade universe membership history
- No guarantee that LLM-generated strategy code perfectly captures every paper on the first try
- No live trading connectivity in Agent 2
