# QuantBros Agent 2

> Preserved research snapshot, not main Marginalia engine. Source lives here so
> Vlad can reuse blueprint, verifier and critic work. Original nested Git repo,
> credentials, database, PDF copies and generated runs remain in local recovery
> copy. Run this project from its own directory/environment. For ingestion,
> pass `--papers-dir ../../agents/strategy_extraction`; default `Papers/` is not
> included. Supply your own ignored `.env` (no `.env.example` in snapshot).
> Its tests are separate from main CI. Generated returns and subprocess scans
> do not establish trustworthy accounting or secure isolation.

Agent 2 is an LLM-harnessed backtesting system where the default path is:

`paper -> StrategyBlueprint -> generated strategy.py -> sandbox execution -> deterministic metrics -> narrative loop -> publish`

## Agentic Mode

In the default runtime:

- one `HarnesserAgent` chooses exactly one tool per turn
- there is no fixed step graph
- the model must explicitly loop through blueprint extraction, code generation, execution, metrics, narrative, and publish
- the generated `strategy.py` is the default strategy executor
- deterministic Python outside the generated code is limited to data fetch, sandboxing, metrics, plotting, artifact IO, and budgets
- failures are loud and structured; there is no silent fallback to `StrategyKind`, presets, or heuristic logic

Every turn writes `artifacts/<run_id>/trace.jsonl` with the harness decision, tool invoked, observation summary, stderr/stdout excerpts, and OpenRouter usage.

## Default Tool Surface

The harness can choose among:

- `extract_blueprint`
- `verify_blueprint`
- `critique_blueprint`
- `repair_blueprint`
- `draft_strategy_code`
- `static_scan_strategy_code`
- `execute_strategy`
- `revise_strategy_code`
- `compute_metrics`
- `compute_robustness`
- `analyst_turn`
- `narrative_verify`
- `publish_final_artifacts`
- `read_artifact`
- `list_artifacts`
- `write_note`

## Strategy Contract

Generated strategy code must implement:

```python
def run_strategy(prices: pd.DataFrame, cfg: dict) -> dict:
    ...
```

Required return keys:

- `weights`
- `gross_returns`

Optional return keys:

- `net_returns`
- `turnover`
- `diagnostics`

The sandbox expects `prices` to be a wide dataframe of ticker columns. Generated code may only use allowlisted imports such as `pandas`, `numpy`, `math`, and `statistics`.

## Install

```bash
python -m pip install ".[data,dev]"
```

## Environment

Copy `.env.example` to `.env` and set at least:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

If OpenRouter is unavailable, runs fail loudly by design.

## Usage

Ingest papers:

```bash
python scripts/ingest_papers.py
```

Run one paper through the default harness:

```bash
PYTHONPATH=src python -m agent2.cli run \
  --paper-id tactical \
  --start 2010-01-01 \
  --end 2024-12-31 \
  --max-harness-turns 16 \
  --max-tool-calls 32 \
  --max-blueprint-attempts 4 \
  --max-codegen-attempts 5 \
  --max-execute-attempts 5 \
  --max-analyst-turns 3
```

Run a local fixture:

```bash
PYTHONPATH=src python -m agent2.cli run \
  --fixture fixtures/spy_dual_ma.json \
  --start 2020-01-01 \
  --end 2024-12-31
```

Run all papers from SQLite:

```bash
PYTHONPATH=src python -m agent2.cli run-all \
  --start 2010-01-01 \
  --end 2024-12-31
```

Deprecated legacy engine path:

```bash
PYTHONPATH=src python -m agent2.cli run \
  --paper-id tactical \
  --legacy-engine
```

## Output Artifacts

A successful run publishes:

- `strategy_blueprint.json`
- `strategy.py`
- `tear_sheet.json`
- `equity_curve.csv`
- `equity_curve.png`
- `trace.jsonl`
- `state.json`

Intermediate versioned artifacts include:

- `strategy_blueprint_v{k}.json`
- `strategy_artifact_v{k}.json`
- `generated/strategy_v{k}.py`
- `execution_v{k}.json`
- `verification_v{k}.json`
- `critique_v{k}.json`
- `repair_instructions_v{k}.json`
- `narrative_v{k}.json`
- `narrative_review_v{k}.json`
- preview data and metric artifacts

Failed runs write `failure.json` and keep the intermediate artifacts that were produced before termination.

## Safety Notes

- Generated strategy code runs in a subprocess with restricted imports, no secret env vars, and explicit timeouts.
- No network is available to the generated strategy by default.
- The harness never silently swaps in deterministic strategy templates when the model fails.
- Metrics are deterministic and computed from executed outputs, not invented by the model.

## Testing

Run the suite with:

```bash
pytest -q
```

Current coverage includes:

- legacy engine parity tests
- friction and metric calculations
- static scan and sandbox execution validation
- harness routing, budget enforcement, and publish behavior
- no-network mocked OpenRouter flows for blueprint, codegen, and narrative loops
