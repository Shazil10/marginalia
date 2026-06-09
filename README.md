# Marginalia

**The insight in the margins of quant research, made executable.**

Marginalia is a multi-agent quant research platform that turns published academic papers into backtested, sentiment-aware strategy intelligence — built for small funds, RIAs, family offices, indie quants, and fintech API partners.

> **Status: Work in progress.** Core agent modules are functional; the full LangGraph pipeline, production API, and live product UI are under active development. Expect breaking changes.

**Website (coming soon):** [marginalia.markets](https://marginalia.markets)

---

## What it does

Marginalia reads dense quant finance papers and extracts the actionable strategy buried inside them — entry/exit rules, parameters, asset universes — then generates Python backtest code, runs historical simulation, and layers live sentiment and risk analysis on top.

```
User goals → Intake → Paper retrieval → Strategy extraction → Codegen → Backtest
                                                              ↓
                                         Sentiment (FinBERT) + Risk + Ranking → Recommendations
```

---

## Current modules

| Module | Status | Description |
|--------|--------|-------------|
| **Intake** | ✅ Working | Natural language → structured risk profile (drawdown, horizon, capital) |
| **Strategy extraction** | ✅ Working | PDF → JSON spec → Python codegen → sandboxed backtest (self-healing retry loop) |
| **Sentiment** | ✅ Working | Tavily news + Reddit ingestion → FinBERT scoring per ticker |
| **Retrieval (FAISS)** | 🚧 Planned | Semantic search over curated academic paper library |
| **Debate & ranking** | 🚧 Planned | Dual-LLM stress-test + Kelly sizing against client mandates |
| **Execution** | 🚧 Planned | Gated Alpaca paper-trading with human approval |

See [`spec.md`](spec.md) for the full build specification.

---

## Quick start

### Prerequisites

- Python 3.11+
- API keys in a `.env` file at the project root (see [Environment variables](#environment-variables))

### Install

```bash
git clone https://github.com/Shazil10/marginalia.git
cd marginalia
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run agents

**Intake agent** — parse investment goals from natural language:

```bash
python agents/intake.py
```

**Strategy extraction** — PDF → backtest pipeline:

```bash
python agents/strategy_extraction/mvp.py
```

**Sentiment pipeline** — news + social → FinBERT scores:

```bash
python run_social_ingest.py --ticker TSLA
python build_sentiment_input.py --ticker TSLA --company Tesla
python run_finbert_sentiment.py --ticker TSLA
```

**Web demo UI** (static prototype):

```bash
cd web && npx serve .
```

---

## Environment variables

Create a `.env` file in the project root:

```env
NEBIUS_API_KEY=
TAVILY_API_KEY=
OPENROUTER_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## Project structure

```
marginalia/
├── agents/
│   ├── intake.py                 # Risk profile extraction
│   ├── sentiment/                # News + social → FinBERT
│   └── strategy_extraction/      # PDF → codegen → backtest
├── data/                         # Raw + processed sentiment data
├── docs/                         # Agent documentation
├── frontend/                     # Streamlit intake prototype
├── utils/                        # Shared LLM client
├── web/                          # Product demo UI
├── spec.md                       # Full pipeline specification
└── requirements.txt
```

---

## Tech stack

- **Orchestration:** LangGraph (planned)
- **LLMs:** Nebius / OpenRouter (OpenAI-compatible)
- **Vector store:** FAISS + sentence-transformers (planned)
- **Backtesting:** yfinance, pandas, quantstats
- **Sentiment:** FinBERT, Tavily
- **Execution:** Alpaca (paper trading, planned)
- **Frontend:** Vercel-ready static UI + Streamlit prototype

---

## Disclaimer

Marginalia is a **research and simulation tool**. It does not provide investment advice. Paper trading only. Not affiliated with any brokerage. Past backtest performance does not guarantee future results.

---

## Author

**Shazil Farukh** — [GitHub](https://github.com/Shazil10) · [Portfolio](https://shazilfarukh.com) · [LinkedIn](https://www.linkedin.com/in/shazil-farukh)

Built in San Francisco.
