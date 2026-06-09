# Strategy Extraction Sub-Pipeline

This directory contains the temporary MVP orchestrator (`mvp.py`) for the strategy extraction pipeline. As you build out the complete LangGraph system detailed in `spec.md`, the logic housed here has been refactored into modular, global components that are ready to be wired into your final `AgentState` graph.

## How the Pipeline Works

The strategy extraction process takes a raw academic quantitative finance PDF and autonomously converts it into an optimized, executable trading backtest wrapper. It executes in three distinct agentic steps:

### 1. Extraction Agent (`agents/strategy_extraction/extraction.py`)
- **Input**: Raw text from an academic PDF.
- **Action**: Uses deep LLM reasoning to perform qualitative NLP extraction. It reads the paper, identifies the core mathematical logic (Entry Signals, Exit Signals, Asset Universe, Rebalance Frequency), and extracts implied parameters. It is specifically prompted to deduce `parameter_ranges` to combat academic publication bias (where authors hide the parameter combinations that didn't work).
- **Output**: A highly structured JSON specification of the trading strategy.

### 2. Code Generation Agent (`agents/strategy_extraction/codegen.py`)
- **Input**: The JSON specification from Step 1 (or the failed Python code + stacktrace from Step 3).
- **Action**: Acts as a Quantitative Python Developer. It translates the JSON text into a robust `pandas` and `yfinance` testing script. Crucially, it incorporates a dynamic **Randomized Grid Search** that will automatically backtest the `parameter_ranges` against historical ticker data to locate the weights/lookbacks that generate the highest safe alpha (Sharpe Ratio). 
- **Output**: Pure, raw Python code containing a zero-argument `run_backtest()` function.

### 3. Backtest Execution Agent (`agents/strategy_extraction/backtest.py`)
- **Input**: The Python script generated in Step 2.
- **Action**: Executes the python script securely in an isolated Subprocess. By manipulating the execution module, it intercepts the generated math and prints the financial performance logic. If the Code Generation agent hallucinated syntax or trapped a pandas indexing error (MultiIndex `Adj Close` errors), this boundary catches the `stderr`, halts the crash, and orchestrates a **Self-Healing Loop**—passing the stack trace directly back into Step 2 for an automated rewrite.
- **Output**: A final nested JSON dictionary detailing CAGR, Max Drawdown, Sharpe Ratio, Win Rate, and the exact Parameter combination that achieved it.

## Architecture Migration
We've set this up so the transition into LangGraph is seamless. 
- All API client initializations have been moved to `utils/llm_client.py`
- Global agents: `agents/strategy_extraction/extraction.py`, `agents/strategy_extraction/codegen.py`, `agents/strategy_extraction/backtest.py`.
- `agents/strategy_extraction/mvp.py` acts purely as a local synchronous orchestrator. When `pipeline/graph.py` is ready, simply point the corresponding graph nodes to the functions exported from these structural modules.

## Validation Results

We ran the pipeline using `anthropic/claude-3.5-sonnet`. It successfully parsed the Giordano ETF logic, generated the grid-search logic, recovered from its own bugs, and ultimately yielded this return after discovering proper allocation weights on a ~5-year YFinance backtest window:

```json
{
  "cagr": 0.01585895644352897,
  "sharpe": 0.9368876055240922,
  "max_drawdown": -0.0570709477799164,
  "calmar": 0.2778814275993123,
  "win_rate": 0.5271629778672032,
  "total_return": 0.1161573689370421,
  "optimized_parameters": {
    "momentum_lookback_months": 2,
    "atr_period": 42,
    "upper_band_highest_close_period": 63,
    "lower_band_highest_low_period": 105,
    "rank_momentum_weight": 1.0,
    "rank_volatility_weight": 1.0,
    "rank_correlation_weight": 1.5,
    "top_n_assets": 4
  }
}
```

The extracted metrics successfully match the exact dictionary structure requested in `spec.md`, meaning the output JSON from this sub-branch is fully ready to be picked up by the Debate Agent downstream!
