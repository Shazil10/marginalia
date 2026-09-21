# Marginalia

**V1: research papers → cited strategy specifications → independently checked
backtests → verification report → human-reviewed strategy registry.**

Build trustworthy research verification first. Evaluate LEAN as primary engine
and a separate bounded reference implementation; own data integrity, fidelity,
comparison tests and audit trail. These integrations are planned, not built.

Portfolio construction and Alpaca paper trading follow V1. Paper account remains
eventual operational target; real money is a separate future decision.

Start with these two guides:

- [Revised plan, current progress and next steps](docs/AI_NATIVE_HEDGE_FUND.md)
- [Shazil and Vlad: parallel workstreams and handoff](docs/WORKSTREAMS.md)

## Current state

`marginalia/` contains deterministic backtesting prototype, six strategy templates,
validated JSON extraction, PDF text/OCR fallbacks, market-data cache, metrics,
basic sizing, LangGraph orchestration, FastAPI endpoints and CLI. Offline tests
cover these components. Full-paper fidelity, general accounting, independent
validation, portfolio risk and Alpaca order execution still need development.

Current backtest optimizes and reports on same history by default. Use
`--no-optimize` for baseline runs; even then, results are preliminary research.
Read engine gaps in system guide before interpreting performance.

First milestone: three manually reviewed cases, then one complete independent
replication, then frozen twenty-case evaluation including failures and blockers.
Two engines agreeing does not itself prove faithful extraction or alpha.

## Run current engine

Python 3.12 recommended. Core install avoids optional sentiment/legacy dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.lock
python -m pytest tests
python -m marginalia.cli --help
```

Backtest example (downloads market data; no LLM required):

```bash
python -m marginalia.cli backtest --spec examples/sector_rotation.json --no-optimize
```

API:

```bash
python -m uvicorn marginalia.api.app:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`.

Paper extraction requires `NEBIUS_API_KEY` in local `.env` and funded provider
access. Set `NEBIUS_FAST_MODEL` / `NEBIUS_POWER_MODEL` to available models;
defaults in code may not match account access. OCR fallback additionally needs
its Python packages and Tesseract system installation.

```bash
python -m marginalia.cli paper --pdf agents/strategy_extraction/quantmentals.pdf --no-optimize
```

`requirements.txt` retains optional original agent dependencies. Original agents
may also use `OPENROUTER_API_KEY` and `TAVILY_API_KEY`. Keep keys in ignored `.env`.
Alpaca integration is planned; no brokerage orders are placed by above commands.

## Repository map

| Path | Purpose |
| --- | --- |
| `marginalia/` | Current engine, data, extraction, graph, API and sizing. |
| `tests/`, `examples/` | Offline core tests and example strategy. |
| `agents/strategy_extraction/` | Original PDF/code-generation prototype and source papers. |
| `experiments/quantbros/` | Separate research snapshot: blueprints, critic, verifier and generated-code trials. |
| `agents/sentiment/` | Optional news/social collection and FinBERT. |
| `frontend/`, `web/` | Historical Flask intake and static product prototypes; not operating console. |
| `docs/` | Current roadmap, ownership and supporting documentation. |
| `spec.md` | Historical build proposal; current guides take precedence. |

Generated reports, downloaded sentiment and caches are local outputs, not source.
Original run examples remain recoverable from Git history. See
[cleanup record](docs/CLEANUP.md) for preserved experiments and removed duplication.

## Team workflow

Shazil works on `Shazil`; Vlad works on `Vlad`. Both open pull requests into
`main`. Shazil (`@Shazil10`) controls main merges. CI runs offline core tests.
See [work plan](docs/WORKSTREAMS.md#branches-and-merge-workflow).
