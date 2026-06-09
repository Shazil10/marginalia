from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, write_json
from .models import SentimentNewsItem, SentimentSocialItem


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def sentiment_input_path(ticker: str) -> Path:
    return Path("data/processed") / f"{ticker}_sentiment_input.json"


def sentiment_output_path(ticker: str) -> Path:
    return Path("data/processed") / f"{ticker}_sentiment_output.json"


def load_news_items(ticker: str) -> list[SentimentNewsItem]:
    path = Path("data/processed") / f"{ticker}_news_items.jsonl"
    rows = read_jsonl(path)
    items: list[SentimentNewsItem] = []
    for row in rows:
        items.append(
            SentimentNewsItem(
                source=row.get("source_domain", ""),
                title=row.get("title", ""),
                summary=row.get("summary", ""),
                url=row.get("url", ""),
                published_date=row.get("published_date"),
                search_score=row.get("search_score"),
            )
        )
    return items


def social_item_from_row(row: dict[str, Any]) -> SentimentSocialItem:
    return SentimentSocialItem(
        platform=row.get("platform", "reddit"),
        subreddit=row.get("subreddit"),
        thread_title=row.get("thread_title", ""),
        thread_url=row.get("thread_url", ""),
        author=row.get("author", "[unknown]"),
        body=row.get("body", ""),
        score=row.get("score"),
        created_utc=row.get("created_utc"),
        discovery_score=row.get("discovery_score"),
        relevance_score=row.get("relevance_score"),
    )


def load_social_items(ticker: str) -> list[SentimentSocialItem]:
    processed_path = Path("data/processed") / f"{ticker}_social_comments.jsonl"
    if processed_path.exists():
        rows = read_jsonl(processed_path)
        return [social_item_from_row(row) for row in rows]

    raw_threads_path = Path("data/raw") / f"{ticker}_reddit_threads.json"
    threads = read_json(raw_threads_path) or []
    fallback_rows: list[SentimentSocialItem] = []
    for thread in threads:
        payload = thread.get("payload")
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        post_children = payload[0].get("data", {}).get("children", [])
        if not post_children:
            continue
        post = post_children[0].get("data", {})
        body = clean_text(post.get("selftext", ""))[:800]
        title = post.get("title", "")
        if not title and not body:
            continue
        fallback_rows.append(
            SentimentSocialItem(
                platform="reddit",
                subreddit=post.get("subreddit"),
                thread_title=title,
                thread_url=thread.get("url", ""),
                author=post.get("author", "[unknown]"),
                body=body or title,
                score=post.get("score"),
                created_utc=post.get("created_utc"),
                discovery_score=thread.get("discovery_score"),
                relevance_score=None,
            )
        )
    return fallback_rows


def build_sentiment_input(ticker: str, company: str) -> dict[str, Any]:
    news_items = load_news_items(ticker)
    social_items = load_social_items(ticker)
    return {
        "ticker": ticker,
        "company": company,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "news_items": [asdict(item) for item in news_items],
        "social_items": [asdict(item) for item in social_items],
        "counts": {
            "news_items": len(news_items),
            "social_items": len(social_items),
        },
    }


def write_sentiment_input(ticker: str, company: str) -> tuple[dict[str, Any], Path]:
    payload = build_sentiment_input(ticker, company)
    output_path = sentiment_input_path(ticker)
    write_json(output_path, payload)
    return payload, output_path


def chunk_text(text: str, limit: int = 450) -> str:
    return clean_text(text)[:limit]


def get_finbert_pipeline():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
    )


def label_to_score(label: str, confidence: float) -> float:
    normalized = label.lower()
    if normalized == "positive":
        return confidence
    if normalized == "negative":
        return -confidence
    return 0.0


def score_items_with_finbert(classifier, items: list[dict[str, Any]], item_type: str) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for item in items:
        if item_type == "news":
            text = chunk_text(f"{item.get('title', '')}. {item.get('summary', '')}")
        else:
            text = chunk_text(f"{item.get('thread_title', '')}. {item.get('body', '')}")

        if not text:
            continue

        result = classifier(text)[0]
        scored.append(
            {
                **item,
                "finbert_label": result["label"].lower(),
                "finbert_confidence": result["score"],
                "finbert_score": label_to_score(result["label"], result["score"]),
            }
        )
    return scored


def aggregate_scores(items: list[dict[str, Any]], weight_key: str | None = None) -> float | None:
    if not items:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for item in items:
        weight = 1.0
        if weight_key is not None:
            candidate = item.get(weight_key)
            if isinstance(candidate, (int, float)):
                weight += max(candidate, 0)
        weighted_sum += item["finbert_score"] * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def score_sentiment_bundle(input_payload: dict[str, Any]) -> dict[str, Any]:
    classifier = get_finbert_pipeline()

    news_items = score_items_with_finbert(classifier, input_payload.get("news_items", []), "news")
    social_items = score_items_with_finbert(classifier, input_payload.get("social_items", []), "social")

    news_score = aggregate_scores(news_items, "search_score")
    social_score = aggregate_scores(social_items, "relevance_score")

    combined_items = []
    combined_items.extend({"type": "news", **item} for item in news_items)
    combined_items.extend({"type": "social", **item} for item in social_items)

    combined_score = aggregate_scores(combined_items)
    label = "neutral"
    if combined_score is not None:
        if combined_score > 0.15:
            label = "bullish"
        elif combined_score < -0.15:
            label = "bearish"

    return {
        "ticker": input_payload["ticker"],
        "company": input_payload["company"],
        "as_of": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "news_items_scored": len(news_items),
            "social_items_scored": len(social_items),
        },
        "news_sentiment_score": news_score,
        "social_sentiment_score": social_score,
        "combined_sentiment_score": combined_score,
        "sentiment_label": label,
        "news_items": news_items,
        "social_items": social_items,
    }


def write_sentiment_output(ticker: str, input_payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    output = score_sentiment_bundle(input_payload)
    output_path = sentiment_output_path(ticker)
    write_json(output_path, output)
    return output, output_path


def build_sentiment_input_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a normalized sentiment input bundle from collected news/social outputs.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, for example ADBE")
    parser.add_argument("--company", required=True, help="Company name, for example Adobe")
    args = parser.parse_args()

    payload, output_path = write_sentiment_input(args.ticker.upper(), args.company)
    print(json.dumps({"sentiment_input_path": str(output_path), "counts": payload["counts"]}, indent=2))


def finbert_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score a normalized sentiment input bundle with FINBERT.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, for example ADBE")
    parser.add_argument("--input-path", help="Optional path to a prebuilt sentiment input JSON.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    input_path = Path(args.input_path) if args.input_path else sentiment_input_path(ticker)
    payload = read_json(input_path)
    if payload is None:
        raise RuntimeError(f"Sentiment input file not found: {input_path}")

    output, output_path = write_sentiment_output(ticker, payload)
    print(
        json.dumps(
            {
                "sentiment_output_path": str(output_path),
                "counts": output["counts"],
                "sentiment_label": output["sentiment_label"],
                "combined_sentiment_score": output["combined_sentiment_score"],
            },
            indent=2,
        )
    )
