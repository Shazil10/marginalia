from __future__ import annotations

import argparse
import json
from typing import Any

from .collection import collect_news, collect_social, get_client
from .config import DEFAULT_LOOKBACK_DAYS, DEFAULT_MAX_RESULTS_PER_QUERY, DEFAULT_MODE
from .io_utils import write_json
from .sentiment import write_sentiment_input, write_sentiment_output


def infer_company_name(ticker: str, company: str | None) -> str:
    return (company or ticker).strip()


def build_coverage(summary: dict[str, Any], sentiment_output: dict[str, Any]) -> dict[str, Any]:
    social = summary.get("social", {})
    news = summary.get("news", {})
    return {
        "news_items": sentiment_output.get("counts", {}).get("news_items_scored", 0),
        "social_items": sentiment_output.get("counts", {}).get("social_items_scored", 0),
        "social_threads": social.get("unique_discussion_pages", 0),
        "news_search_results": news.get("search_results", 0),
        "social_search_results": social.get("search_results", 0),
        "enough_social_data": social.get("enough_social_data", False),
    }


def estimate_confidence(summary: dict[str, Any], sentiment_output: dict[str, Any]) -> float:
    coverage = build_coverage(summary, sentiment_output)
    confidence = 0.25
    confidence += min(coverage["news_items"], 10) * 0.03
    confidence += min(coverage["social_threads"], 5) * 0.05
    if coverage["enough_social_data"]:
        confidence += 0.1
    return round(min(confidence, 0.95), 2)


def build_agent_warnings(summary: dict[str, Any], sentiment_output: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    coverage = build_coverage(summary, sentiment_output)
    if coverage["social_items"] == 0:
        warnings.append("No usable social comments were collected; social sentiment is weak or unavailable.")
    elif not coverage["enough_social_data"]:
        warnings.append("Social coverage is thin; social sentiment has low confidence.")
    if coverage["news_items"] < 4:
        warnings.append("News coverage is thin; news sentiment may be unstable.")
    if sentiment_output.get("social_sentiment_score") == 0.0 and coverage["social_items"] > 0:
        warnings.append("Social items scored near-neutral overall; verify they reflect stock sentiment rather than product or customer-service chatter.")
    return warnings


def run_pipeline(
    ticker: str,
    company: str,
    days: int,
    max_results_per_query: int,
    mode: str,
) -> dict[str, Any]:
    client = get_client()
    summary: dict[str, Any] = {
        "ticker": ticker,
        "company": company,
        "mode": mode,
    }

    if mode in {"all", "news"}:
        summary["news"] = collect_news(client, ticker, company, days, max_results_per_query)

    if mode in {"all", "social"}:
        summary["social"] = collect_social(client, ticker, company, days, max_results_per_query)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect stock news and social discussion for sentiment analysis.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, for example TSLA")
    parser.add_argument("--company", help="Optional company name override, for example Tesla")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="How far back discovery should search.")
    parser.add_argument("--max-results-per-query", type=int, default=DEFAULT_MAX_RESULTS_PER_QUERY, help="Max Tavily search results per query.")
    parser.add_argument("--mode", choices=["all", "news", "social"], default=DEFAULT_MODE, help="Which collector to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip().upper()
    company = infer_company_name(ticker, args.company)
    summary = run_pipeline(
        ticker=ticker,
        company=company,
        days=args.days,
        max_results_per_query=args.max_results_per_query,
        mode=args.mode,
    )

    sentiment_input, sentiment_input_file = write_sentiment_input(ticker, company)
    summary["sentiment_input"] = {
        "path": str(sentiment_input_file),
        "counts": sentiment_input["counts"],
    }

    try:
        sentiment_output, sentiment_output_file = write_sentiment_output(ticker, sentiment_input)
        warnings = build_agent_warnings(summary, sentiment_output)
        confidence = estimate_confidence(summary, sentiment_output)
        coverage = build_coverage(summary, sentiment_output)
        sentiment_output["warnings"] = warnings
        sentiment_output["confidence"] = confidence
        sentiment_output["coverage"] = coverage
        write_json(sentiment_output_file, sentiment_output)
        summary["sentiment_output"] = {
            "path": str(sentiment_output_file),
            "sentiment_label": sentiment_output["sentiment_label"],
            "combined_sentiment_score": sentiment_output["combined_sentiment_score"],
            "counts": sentiment_output["counts"],
            "confidence": confidence,
            "coverage": coverage,
            "warnings": warnings,
        }
    except Exception as exc:
        summary["sentiment_output"] = {
            "path": None,
            "error": str(exc),
        }

    print(json.dumps(summary, indent=2))
