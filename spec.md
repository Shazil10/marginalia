# QuantAgent — Full Build Specification
> Use this document as a prompt for AI coding agents (Cline, Cursor, GitHub Copilot, etc.) to build the full QuantAgent pipeline locally. Follow sections in order. Each section is a self-contained build task.

---

## Project Overview

QuantAgent is an autonomous multi-agent system that:
1. Takes a user's investment goals and risk tolerance in natural language
2. Searches a curated vector store of academic trading strategy papers
3. Extracts the algorithmic logic from matching papers
4. Generates and executes backtesting code against historical market data
5. Runs a two-LLM debate to stress-test each strategy
6. Recommends the best-fit strategy with Kelly Criterion position sizing
7. Runs a safety gate before executing real (paper) trades via Alpaca

The end user never needs to understand finance. They type a sentence. They get a trade executed.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM Provider | Nebius Token Factory (OpenAI-compatible API) |
| Web Search / Sentiment | Tavily Python SDK |
| Agent Orchestration | LangGraph |
| Vector Store | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | PyPDF2 |
| Market Data | yfinance |
| Backtesting Execution | Python subprocess (sandboxed) |
| Safety Evaluation | Toloka OpenClaw API |
| Trade Execution | Alpaca Trade API (paper trading mode) |
| Frontend | Streamlit |
| AI Coding Assistant | Cline (use actively for code generation agent task) |

---

## Environment Variables

Create a `.env` file in the project root with the following keys. All are required before running.

```
NEBIUS_API_KEY=
TAVILY_API_KEY=
OPENROUTER_API_KEY=
TOLOKA_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Load all env vars at the top of every agent file using `python-dotenv`.

---

## Project Structure

Build exactly this directory structure. Do not deviate.

```
quant-agent/
├── agents/
│   ├── __init__.py
│   ├── intake.py
│   ├── retrieval.py
│   ├── extraction.py
│   ├── codegen.py
│   ├── backtest.py
│   ├── debate.py
│   ├── recommend.py
│   ├── openclaw_gate.py
│   └── executor.py
├── pipeline/
│   ├── __init__.py
│   ├── graph.py
│   └── state.py
├── vector_store/
│   ├── __init__.py
│   ├── build_index.py
│   └── query.py
├── data/
│   └── papers/          ← Drop all PDF papers here before running
├── frontend/
│   └── app.py
├── utils/
│   ├── __init__.py
│   ├── nebius_client.py
│   └── kelly.py
├── .env
├── requirements.txt
└── README.md
```

---

## Dependencies

The `requirements.txt` must include exactly these packages. Pin no versions unless a conflict arises.

```
langgraph
langchain
langchain-community
tavily-python
faiss-cpu
sentence-transformers
yfinance
pandas
numpy
alpaca-trade-api
PyPDF2
openai
streamlit
python-dotenv
requests
```

Install with: `pip install -r requirements.txt`

---

## Step 0 — Shared State Definition

**File:** `pipeline/state.py`

Define a single `AgentState` TypedDict that is passed through every node in the LangGraph pipeline. Every agent reads from and writes to this state object. No agent should use global variables or class-level state.

The state must contain fields for every piece of data that flows between agents:

- `user_input` — raw string from the user
- `risk_profile` — structured dict produced by the Intake Agent
- `candidate_papers` — list of dicts from the vector store search
- `market_sentiment` — string of live context from Tavily
- `strategy_specs` — list of structured strategy dicts from Extraction Agent
- `strategy_code` — list of dicts containing generated Python code per strategy
- `codegen_errors` — list of error strings from failed backtest executions
- `codegen_retries` — int counter for retry loop (max 2)
- `backtest_results` — list of dicts with performance metrics per strategy
- `debate_summaries` — list of dicts containing prosecution, defense, and verdict per strategy
- `recommendations` — list of ranked recommendation dicts
- `selected_strategy` — single dict, the strategy the user chose to execute
- `openclaw_cleared` — boolean, whether the safety gate passed
- `safety_issues` — list of strings describing any safety gate failures
- `trade_spec` — dict of specific trade actions to execute
- `execution_result` — dict containing trade confirmation details

All fields must have appropriate types. Optional fields that may not be populated early in the pipeline should be typed as `Optional`.

---

## Step 1 — Nebius Client Utility

**File:** `utils/nebius_client.py`

Create a shared utility that returns a configured OpenAI client pointed at the Nebius Token Factory base URL. Nebius Token Factory is fully OpenAI SDK compatible, so use the standard `openai.OpenAI` class with:
- `api_key` set from the `NEBIUS_API_KEY` environment variable
- `base_url` set to `https://api.studio.nebius.ai/v1/`

All agents import this utility rather than instantiating their own clients. This ensures a single place to swap models or keys.

The default model to use across all agents unless otherwise specified is `meta-llama/Meta-Llama-3.1-70B-Instruct`.

---

## Step 2 — Vector Store Build

**File:** `vector_store/build_index.py`

This script is run once before the pipeline starts. It must:

1. Walk the `data/papers/` directory and load every `.pdf` file
2. Extract text from each PDF using PyPDF2. Take the first 8,000 characters of each paper — this covers the abstract and methodology sections which contain the strategy logic
3. For each paper, parse basic metadata from the filename. The naming convention for papers is: `strategy_type_author_year.pdf` (e.g., `momentum_antonacci_2014.pdf`). Extract strategy type, author, and year from the filename
4. Embed all paper texts using the `sentence-transformers` model `all-MiniLM-L6-v2`. This model is fast and sufficient for semantic similarity on financial text
5. Build a FAISS `IndexFlatL2` index from the embeddings
6. Save the FAISS index to `vector_store/papers.index`
7. Save the paper metadata (filename, text, parsed metadata) to `vector_store/papers_metadata.pkl` using pickle

Print a confirmation message showing how many papers were indexed successfully.

**File:** `vector_store/query.py`

Create a `query_index(query_text, k=5)` function that:
1. Loads the FAISS index and metadata from disk
2. Encodes the query text using the same sentence-transformer model
3. Runs a nearest-neighbor search returning the top `k` results
4. Returns a list of paper dicts with their text and metadata

The query function should be importable and callable from any agent.

---

## Step 3 — LangGraph Pipeline Orchestration

**File:** `pipeline/graph.py`

Build the full LangGraph `StateGraph` that wires all agents together. Use the `AgentState` TypedDict as the graph's state type.

**Nodes** (one per agent, in order):
- `intake` → calls `intake_agent`
- `retrieval` → calls `retrieval_agent`
- `extraction` → calls `extraction_agent`
- `codegen` → calls `codegen_agent`
- `backtest` → calls `backtest_agent`
- `debate` → calls `debate_agent`
- `recommend` → calls `recommend_agent`
- `openclaw_gate` → calls `openclaw_agent`
- `executor` → calls `executor_agent`

**Edges:**
- Linear: `intake → retrieval → extraction → codegen → backtest`
- **Conditional after `backtest`**: if `codegen_errors` is non-empty AND `codegen_retries < 2`, route back to `codegen`. Otherwise route to `debate`. This creates the self-healing retry loop.
- Linear: `debate → recommend`
- **Conditional after `recommend`**: if `selected_strategy` is populated in state, route to `openclaw_gate`. Otherwise route to `END`. This gate only fires after user confirmation.
- **Conditional after `openclaw_gate`**: if `openclaw_cleared` is `True`, route to `executor`. Otherwise route to `END`.
- Linear: `executor → END`

Set `intake` as the entry point. Compile the graph and export it as `app` so the frontend can call `app.invoke({"user_input": "..."})`.

---

## Step 4 — Agent 1: Intake Agent

**File:** `agents/intake.py`

**Purpose:** Convert freeform natural language into a structured risk profile JSON.

**System prompt instructions for the LLM:**
- Act as a financial intake specialist
- Extract the following fields from the user's message: max drawdown tolerance (as a float 0–1), investment horizon in years, capital amount in dollars, target annual return (as a float 0–1), risk class (one of: conservative, moderate-conservative, moderate, aggressive), excluded sectors (list of strings), and benchmark (default to "SPY")
- Apply this risk class mapping: language suggesting fear of loss or crash maps to "conservative" with max drawdown 0.10; "some risk is okay" maps to "moderate-conservative" at 0.20; wanting growth and tolerating swings maps to "moderate" at 0.30; wanting maximum returns regardless of loss maps to "aggressive" at 0.50
- Default capital to 10,000 if not mentioned
- Default horizon to 10 years if not mentioned
- Infer target return from risk class if not stated: conservative → 0.07, moderate-conservative → 0.10, moderate → 0.13, aggressive → 0.18
- If a critical field is completely ambiguous, set a `clarification_needed` key with a single clarifying question. Otherwise set it to null.
- Return **only** valid JSON. No preamble, no explanation, no markdown fences.

**Temperature:** 0.1 (low — this is structured extraction)

**State update:** Write the parsed JSON to `state["risk_profile"]`

---

## Step 5 — Agent 2: Retrieval Agent

**File:** `agents/retrieval.py`

**Purpose:** Find the most relevant strategy papers from the vector store AND pull live market sentiment from Tavily.

**Layer 1 — Vector store retrieval:**
1. Build a semantic query string from the risk profile: combine risk class, horizon, max drawdown tolerance, and target return into a natural language description
2. Call `query_index(query, k=5)` from `vector_store/query.py`
3. Apply a hard filter: exclude any paper whose metadata `typical_drawdown` field exceeds the user's `max_drawdown` by more than 20%. If `typical_drawdown` is not in metadata, do not exclude the paper.
4. Take the top 3 papers after filtering and pass them forward

**Layer 2 — Tavily sentiment (run in parallel using threading or asyncio):**
- Fire two Tavily searches simultaneously:
  - Query 1: Current macro market outlook relevant to the user's risk class and 2026 market conditions
  - Query 2: Recent performance of momentum/factor strategies in current market environment
- Use `search_depth="basic"` and `max_results=3` for each
- Concatenate all result `content` fields into a single `market_sentiment` string

**State update:** Write `candidate_papers` (list of 3 paper dicts) and `market_sentiment` (string) to state.

---

## Step 6 — Agent 3: Extraction Agent

**File:** `agents/extraction.py`

**Purpose:** Read each candidate paper and produce a machine-executable strategy specification JSON.

**For each paper in `state["candidate_papers"]`**, make a separate LLM call with:

**Input to LLM:** The full paper text (first 8,000 chars) plus the market sentiment string appended as "CURRENT MARKET CONTEXT."

**System prompt instructions:**
- Act as a quantitative analyst extracting trading strategies from academic papers
- Extract these fields and return only valid JSON:
  - `strategy_name`: descriptive name
  - `strategy_type`: one of momentum, mean_reversion, factor, trend_following, other
  - `entry_signal`: specific, implementable description of when to buy
  - `exit_signal`: specific, implementable description of when to sell
  - `rebalance_frequency`: one of daily, weekly, monthly, quarterly
  - `asset_universe`: list of tickers or description (e.g. "S&P 500 constituents")
  - `parameters`: dict of all tunable numeric parameters (e.g. lookback windows, thresholds)
  - `hidden_params`: list of parameters the paper deliberately obscures or does not specify
  - `suggested_defaults`: conservative default values for all hidden params
  - `source_paper`: author + year string
  - `source_journal`: journal name or null
  - `estimated_sharpe`: float from paper or null
  - `estimated_drawdown`: float 0–1 from paper or null
  - `python_indicators_needed`: list of technical indicators needed (e.g. moving_average, RSI, momentum)
- For hidden parameters, always suggest the most conservative reasonable default
- If the entry signal is ambiguous, describe the best interpretation and add the key `entry_signal_inferred: true`
- Return only valid JSON, no markdown, no explanation

**Temperature:** 0.1

**State update:** Append each strategy spec dict (with `paper_filename` added) to `state["strategy_specs"]`

---

## Step 7 — Agent 4: Code Generation Agent

**File:** `agents/codegen.py`

**Purpose:** Translate each strategy specification JSON into a complete, runnable Python backtesting script.

This agent is the primary place to use Cline actively. Have Cline open in VS Code to autocomplete and validate generated code in real time.

**For each spec in `state["strategy_specs"]`**, make a separate LLM call with:

**Input to LLM:** The strategy spec JSON. If `state["codegen_errors"]` is non-empty (meaning this is a retry), append the error messages to the input with a header "PREVIOUS ERRORS TO FIX."

**System prompt requirements for the generated code:**
- Use `yfinance` to download historical price data from 2004-01-01 to 2024-12-31
- Use `pandas` for all signal calculations and portfolio logic
- Implement the exact entry and exit signals from the spec
- Wrap everything in a single function called `run_backtest()` that takes no arguments
- The function must return a dict with exactly these keys:
  - `cagr`: compound annual growth rate as a float
  - `sharpe`: annualized Sharpe ratio as a float
  - `max_drawdown`: maximum drawdown as a negative float (e.g. -0.34)
  - `calmar`: CAGR divided by absolute max drawdown
  - `win_rate`: fraction of periods with positive returns
  - `total_return`: total return over full period
  - `crisis_2008`: portfolio return during 2008-10-01 to 2009-03-31
  - `crisis_2020`: portfolio return during 2020-02-01 to 2020-04-30
  - `spy_cagr`: SPY CAGR over the same period as benchmark
  - `annual_returns`: dict of year (string) to annual return (float)
- Handle all missing data with try/except — the function must never raise an unhandled exception
- All imports must be at the top of the script
- The script must be completely self-contained — no external dependencies other than yfinance, pandas, numpy

**Temperature:** 0.2

**Output format:** Return only the raw Python code. No markdown fences. No explanation. No comments that weren't in the original spec.

**State update:**
- Append each `{strategy_name, code, spec}` dict to `state["strategy_code"]`
- Increment `state["codegen_retries"]` by 1
- Reset `state["codegen_errors"]` to empty list

---

## Step 8 — Agent 5: Backtesting Agent

**File:** `agents/backtest.py`

**Purpose:** Execute each generated backtesting script in a sandboxed subprocess and collect results.

**For each strategy in `state["strategy_code"]`:**
1. Write the code to a temporary `.py` file using `tempfile.NamedTemporaryFile`
2. Append a `__main__` block to the temp file that calls `run_backtest()` and prints the result as JSON
3. Execute the temp file using `subprocess.run` with `capture_output=True`, `text=True`, and a `timeout=60`
4. If the subprocess exits with a non-zero return code, capture `stderr` and append a descriptive error string to `state["codegen_errors"]`. Skip this strategy.
5. If the subprocess times out, append a timeout error. Skip this strategy.
6. If the stdout cannot be parsed as JSON, append a parse error. Skip this strategy.
7. On success, parse the JSON output and append a result dict to `state["backtest_results"]` containing: `strategy_name`, `source` (from spec), `metrics` (the parsed JSON), `spec`
8. Always delete the temp file in a `finally` block

**State update:** Write `backtest_results` and `codegen_errors` to state.

---

## Step 9 — Agent 6: Debate Agent

**File:** `agents/debate.py`

**Purpose:** Run a two-LLM adversarial debate for each backtested strategy. This is the primary originality differentiator — make it clearly visible in the demo.

**For each result in `state["backtest_results"]`**, run two sequential LLM calls:

**Call 1 — Prosecution (skeptical risk officer):**
- System prompt: You are a skeptical risk officer at a hedge fund. Your job is to find 3 specific reasons why this trading strategy should NOT be recommended to a retail investor. Be specific — reference the actual metrics provided. Cover at least these risk dimensions: (1) overfitting or data snooping risk, (2) regime sensitivity or when the strategy fails, (3) practical implementation concerns for a retail investor. Return only valid JSON with three critique objects, each containing: `issue` (short label), `severity` (high/medium/low), `detail` (2-sentence explanation).
- Temperature: 0.7 (higher — generate creative, diverse critiques)

**Call 2 — Defense (portfolio manager):**
- System prompt: You are the portfolio manager who developed this strategy. You have just received three critiques from a risk officer. Respond to each critique directly and honestly. Concede where the critique is valid. Defend where it is overstated or wrong. End with a single sentence final verdict for a retail investor. Return only valid JSON with three response objects (each containing `concede` boolean and `response` string) and a `final_verdict` string.
- Input: The original strategy context plus the prosecution's critiques
- Temperature: 0.3 (lower — measured, reasoned responses)

**State update:** Append a `debate_summaries` dict per strategy containing: `strategy_name`, `critiques`, `responses`, `verdict` (the final_verdict string), `backtest_result` (the full result dict).

---

## Step 10 — Kelly Criterion Utility

**File:** `utils/kelly.py`

Implement a `kelly_criterion(win_rate, avg_win, avg_loss)` function that:
- Calculates the Kelly fraction: `(win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win`
- Caps the result at 0.25 (quarter-Kelly maximum for safety)
- Returns 0 if avg_loss is 0 or the result is negative
- Also returns a `half_kelly` value which is the Kelly fraction divided by 2, representing the conservative recommended allocation

The function should return a dict with `kelly_fraction`, `half_kelly`, and `suggested_allocation_pct`.

---

## Step 11 — Agent 7: Recommendation Agent

**File:** `agents/recommend.py`

**Purpose:** Rank all backtested strategies against the user's risk profile and produce final user-facing recommendation cards with position sizing.

**For each debate summary in `state["debate_summaries"]`:**

1. **Risk alignment scoring** — compute a composite score (0–1):
   - Drawdown alignment: how close is the strategy's max drawdown to the user's stated tolerance? Penalize strategies that exceed tolerance.
   - Return alignment: ratio of strategy CAGR to user's target return, capped at 1.5
   - Weight: 60% drawdown alignment, 40% return alignment

2. **Kelly sizing** — call `kelly_criterion` from `utils/kelly.py` using the strategy's `win_rate` and CAGR as proxy values. Apply to the user's `capital` field to get a dollar suggestion.

3. **Plain English explanation** — make one additional LLM call per strategy with a simple prompt asking for a 2-sentence explanation of what the strategy does and why it performed well, written for a complete beginner. No jargon. Temperature 0.5.

4. **Build recommendation card** — produce a dict per strategy containing:
   - `rank`: integer (set after sorting)
   - `strategy_name`
   - `source`: paper citation
   - `journal`: journal name or "Academic Research"
   - `explanation`: the plain English 2-sentence description
   - `metrics`: formatted strings for CAGR, Sharpe, max drawdown, vs S&P500, crisis_2008 return, crisis_2020 return
   - `sizing`: dict with Kelly fraction, suggested dollar amount, and the note "Half-Kelly applied for safety"
   - `risk_summary`: the `verdict` string from the debate
   - `alignment_score`: the computed float

5. Sort all recommendation cards by `alignment_score` descending. Assign ranks starting from 1.

**State update:** Write sorted `recommendations` list to state.

---

## Step 12 — OpenClaw Safety Gate

**File:** `agents/openclaw_gate.py`

**Purpose:** Run an independent safety validation on the user's selected strategy before any trade is executed. This agent only runs after the user confirms their selection in the frontend.

**Run these checks sequentially:**

**Check 1 — Drawdown alignment:**
Compare the strategy's historical max drawdown against the user's stated `max_drawdown` tolerance. If the strategy's drawdown exceeds the user's tolerance by more than 30%, add a descriptive safety issue to the list.

**Check 2 — Capital concentration:**
Calculate what percentage of the user's total capital is being deployed. If the suggested allocation exceeds 50% of total capital, add a safety issue.

**Check 3 — Excluded sectors:**
If the user specified excluded sectors during intake, check whether any of the strategy's asset universe overlaps with those sectors. If overlap exists, add a safety issue.

**Check 4 — OpenClaw API call:**
Make a POST request to the Toloka OpenClaw security eval endpoint at `https://platform.toloka.ai/preset/openclaw-security-test` with the Authorization header set to `ApiKey {TOLOKA_API_KEY}`. Pass the full trade context as the request body. Parse the response — if the `flagged` field is true, add the `reason` from the response to safety issues. Wrap in try/except with a 10-second timeout — if the API is unreachable, log the failure but do not block execution.

**State update:**
- Set `state["openclaw_cleared"]` to `True` if `safety_issues` is empty, `False` otherwise
- Write `safety_issues` list to state

---

## Step 13 — Executor Agent

**File:** `agents/executor.py`

**Purpose:** Execute the approved trade strategy via Alpaca Paper Trading API.

**Setup:**
Initialize the Alpaca REST client using `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_BASE_URL` from environment variables. The base URL must point to `https://paper-api.alpaca.markets` — never live trading.

**For each action in `state["trade_spec"]["actions"]`:**
1. Fetch the current price of the symbol using Alpaca's `get_latest_bar`
2. Calculate quantity: divide the dollar allocation (capital × allocation_pct) by current price, truncate to integer
3. Skip if quantity is less than 1
4. Submit a market order via `api.submit_order` with `time_in_force='gtc'`
5. Immediately after each filled buy order, submit a stop-loss sell order at a price calculated as: `current_price × (1 - abs(max_drawdown))` where max_drawdown comes from the strategy metrics
6. Wrap each order in try/except and collect results whether success or failure — never abort the entire execution due to one failed order

**State update:** Write `execution_result` to state containing: list of `trades` (each with symbol, qty, side, order_id, status, estimated_cost or error), `total_deployed` dollar amount, and `portfolio_url` pointing to the Alpaca paper trading dashboard.

---

## Step 14 — Frontend

**File:** `frontend/app.py`

Build a Streamlit application. This is the only user-facing interface. It must be runnable with `streamlit run frontend/app.py` from the project root.

**Page config:** Wide layout. Title "QuantAgent". Subtitle "Your autonomous quant fund."

**Section 1 — User Input:**
- A `st.text_area` with placeholder text showing an example of a natural language investment goal
- A primary "Analyze My Profile →" button
- On click: show a progress bar that advances through 7 stages (one per agent), with a status text label for each stage. Run the full pipeline by calling `app.invoke({"user_input": user_input})` from `pipeline/graph.py`. Store the full result in `st.session_state`.

**Section 2 — Recommendation Cards:**
Display each recommendation in a `st.expander`. The first recommendation (rank 1) should be expanded by default. Each card must show:
- Header: rank, strategy name, CAGR, Sharpe ratio
- Two columns: left for the plain English explanation and source citation; right for key metrics displayed as `st.metric` widgets
- A `st.warning` block for the risk summary / debate verdict
- The Kelly sizing suggestion in plain text
- An "Execute This Strategy" button that stores the selected strategy in `st.session_state` and triggers a rerun

**Section 3 — Execution Confirmation (shown only when `selected_strategy` is in session state):**
Display a confirmation block showing in plain English exactly what trades will be placed and for how much. Two columns: a green "Confirm & Execute" primary button and a red "Cancel" button.

On confirm:
1. Show a spinner with text "Running safety check..."
2. Call the pipeline with the selected strategy and the existing state
3. If `openclaw_cleared` is True: show a green success message, display the trade results as JSON, and show a link button to the Alpaca paper trading dashboard
4. If `openclaw_cleared` is False: show a red error block listing each safety issue

On cancel: remove `selected_strategy` from session state and rerun.

---

## Step 15 — Paper Trading Account Setup

Before running the full pipeline, verify the Alpaca paper trading account is active:
1. Sign up at `https://alpaca.markets` (free)
2. Switch to paper trading mode in the dashboard
3. Note the paper trading API key and secret (separate from live trading keys)
4. Confirm the base URL is `https://paper-api.alpaca.markets` in `.env`
5. Paper accounts start with $100,000 in simulated cash — no real money is used

---

## Step 16 — Run Order

Execute steps in this exact order when setting up locally for the first time:

1. `pip install -r requirements.txt`
2. Fill in all keys in `.env`
3. Drop all PDF papers into `data/papers/`
4. `python vector_store/build_index.py` — builds FAISS index from papers
5. `streamlit run frontend/app.py` — starts the full application

The full pipeline runs on demand from the frontend. No separate backend server needed.

---

## Paper Strategy Archetypes to Source

Download 5–7 papers per archetype from SSRN (ssrn.com) or arXiv (arxiv.org/list/q-fin.PM/recent) and save to `data/papers/` before running.

Use this naming convention strictly: `strategytype_author_year.pdf`

| Archetype | Search Terms | Target Papers |
|---|---|---|
| Momentum | "cross-sectional momentum equity", "dual momentum Antonacci" | 6 papers |
| Mean Reversion | "pairs trading cointegration", "statistical arbitrage equity" | 5 papers |
| Low Volatility | "low volatility anomaly equity", "minimum variance portfolio" | 5 papers |
| Trend Following | "time series momentum", "managed futures trend following" | 5 papers |
| Factor | "quality factor equity", "value investing systematic" | 4 papers |

---

## Demo Flow (for judges)

The demo should show these five moments in order:

1. **Input** — Type a natural language risk profile in the text area
2. **Pipeline visible** — Progress bar advances through all 7 agent stages with status labels
3. **Recommendations** — Three ranked strategy cards appear with real backtested metrics including 2008 and 2020 crisis performance
4. **Debate visible** — Expand the risk summary on any card to show the adversarial debate verdict
5. **Execution** — Click execute, show the OpenClaw safety check passing (green), then show the Alpaca paper trading dashboard with filled orders appearing in real time

Total demo time: 3 minutes. Do not show code. Do not show Figma. Do not show a presentation. Show the live product only.

---

## Key Pitch Points

When judges ask questions, lead with these:

- **On novelty:** "We have two LLMs arguing about every strategy before it reaches the user — a prosecution and a defense — so the user sees not just what to do but why it might be wrong."
- **On safety:** "Every trade passes through an independent safety gate that checks alignment with the user's stated risk tolerance before a single order is placed."
- **On accessibility:** "The user never sees a chart, a ticker, or a financial term. They type one sentence. They get a trade executed."
- **On Tavily:** "We use Tavily for two distinct signals — peer-reviewed strategy retrieval and live market sentiment — then reconcile them in the extraction layer."
- **On Cline:** "Cline was our pair programmer for the code generation agent — it went from strategy spec to running backtest code in under 10 minutes."
- **On Nebius:** "All LLM inference runs through Nebius Token Factory — we benchmarked it specifically for the multi-turn reasoning required in our debate agent."

---

## What NOT to Build

Do not spend time on:

- Live brokerage integration (Schwab, Robinhood, etc.) — Alpaca paper trading is sufficient
- Mobile app — Streamlit web is the demo surface
- User authentication or accounts — single-user local demo only
- Scraping papers at runtime — all papers are pre-loaded in the vector store
- Fine-tuning any model — use the base Nebius Token Factory models as-is
- Real money — paper trading only, every time, no exceptions