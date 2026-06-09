# Sentiment Agent

This agent turns a stock identifier into a sentiment payload that downstream agents can consume.

## Purpose

Input:

```json
{
  "ticker": "SOFI"
}
```

Output:

```json
{
  "ticker": "SOFI",
  "sentiment_label": "neutral",
  "combined_sentiment_score": -0.07,
  "news_sentiment_score": -0.12,
  "social_sentiment_score": 0.0,
  "confidence": 0.42,
  "coverage": {
    "news_items": 13,
    "social_items": 3,
    "social_threads": 3
  },
  "warnings": [
    "Social coverage is thin; social sentiment has low confidence."
  ]
}
```

This output is intended for the rest of the pipeline, not for direct manual inspection of raw collection artifacts.

## Pipeline

1. Discovery via Tavily for finance news and public discussion sources.
2. Collection and normalization of news items plus Reddit discussion.
3. Filtering of weak results, duplicates, and low-signal social comments.
4. Evidence bundle creation in `data/processed/<TICKER>_sentiment_input.json`.
5. FINBERT scoring and aggregation in `data/processed/<TICKER>_sentiment_output.json`.
6. Confidence, coverage, and warning annotation for downstream consumers.

## Package Layout

- `agents/sentiment/config.py`
- `agents/sentiment/models.py`
- `agents/sentiment/io_utils.py`
- `agents/sentiment/collection.py`
- `agents/sentiment/sentiment.py`
- `agents/sentiment/pipeline.py`

Thin compatibility wrappers remain at the repo root:

- `run_social_ingest.py`
- `build_sentiment_input.py`
- `run_finbert_sentiment.py`
- `tavily_social_pipeline.py`
- `sentiment_pipeline.py`

## Integration Notes

Preferred agent input:

```json
{
  "ticker": "AAPL"
}
```

Optional override:

```json
{
  "ticker": "AAPL",
  "company": "Apple"
}
```

Recommended downstream behavior:

- weight high-confidence outputs more heavily
- inspect `warnings` before acting on thin coverage
- consider `news_sentiment_score` and `social_sentiment_score` separately before using the combined score
