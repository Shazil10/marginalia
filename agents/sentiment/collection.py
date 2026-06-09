from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tavily import TavilyClient

from .config import (
    BLOCKED_URL_PATTERNS,
    DEFAULT_MIN_DISCUSSION_PAGES,
    NEWS_DOMAINS,
    NEWS_QUERY_TEMPLATES,
    REDDIT_BLOCKED_SUBREDDITS,
    REDDIT_SUBREDDITS,
    SOCIAL_DOMAINS,
    SOCIAL_QUERY_TEMPLATES,
    SOCIAL_SIGNAL_TERMS,
)
from .io_utils import ensure_dirs, load_env_file, write_json, write_jsonl
from .models import NewsRecord, SocialCommentRecord


def get_client() -> TavilyClient:
    load_env_file()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")
    return TavilyClient(api_key=api_key)


def map_days_to_time_range(days: int) -> str:
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def build_queries(ticker: str, company: str, templates: list[str]) -> list[str]:
    company_slug = re.sub(r"[^a-z0-9]+", "", company.lower())
    return [
        template.format(
            ticker=ticker,
            company=company,
            company_slug=company_slug,
        )
        for template in templates
    ]


def run_tavily_searches(
    client: TavilyClient,
    queries: list[str],
    days: int,
    max_results_per_query: int,
    include_domains: list[str],
) -> dict[str, Any]:
    time_range = map_days_to_time_range(days)
    all_results: list[dict[str, Any]] = []

    for query in queries:
        response = client.search(
            query=query,
            topic="general",
            search_depth="advanced",
            time_range=time_range,
            include_domains=include_domains,
            include_raw_content=False,
            max_results=max_results_per_query,
        )
        for result in response.get("results", []):
            result["query"] = query
            all_results.append(result)

    return {
        "days": days,
        "time_range": time_range,
        "results": dedupe_results(all_results),
    }


def dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: -(item.get("score") or 0.0)):
        url = result.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(result)
    return unique


def is_low_quality_news_url(url: str) -> bool:
    lowered = url.lower()
    bad_patterns = [
        "/analystestimates",
        "/stock/",
        "stock-quote",
    ]
    return any(pattern in lowered for pattern in bad_patterns)


def is_low_quality_news_item(title: str, summary: str) -> bool:
    combined = f"{title} {summary}".lower()
    listicle_markers = [
        "prediction:",
        "top ",
        "best performing",
        "2 large-cap",
        "1 artificial intelligence",
        "worth more than",
        "in 5 years",
        "is this a collapse or a buying opportunity",
        "is it too late",
        "could be a great choice",
    ]
    return any(marker in combined for marker in listicle_markers)


def is_news_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    if any(pattern in lowered for pattern in BLOCKED_URL_PATTERNS):
        return False
    if is_low_quality_news_url(url):
        return False
    domain = urlparse(url).netloc.lower()
    if "reuters.com" in domain:
        return "/world/" in lowered or "/business/" in lowered or "/technology/" in lowered
    if "finance.yahoo.com" in domain:
        return "/news/" in lowered or "/video/" in lowered
    if "seekingalpha.com" in domain:
        return "/article/" in lowered
    if "marketwatch.com" in domain:
        return True
    return False


def is_social_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    domain = urlparse(url).netloc.lower()
    if "reddit.com" in domain:
        return "/comments/" in lowered
    if "stocktwits.com" in domain:
        return "/symbol/" in lowered or "/message/" in lowered
    return False


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.replace("[...]", " ")


def normalize_news_records(ticker: str, company: str, search_payload: dict[str, Any]) -> list[NewsRecord]:
    records: list[NewsRecord] = []
    for item in search_payload.get("results", []):
        url = item.get("url", "")
        if not is_news_url(url):
            continue
        title = item.get("title", "")
        summary = clean_text(item.get("content", ""))[:600]
        if is_low_quality_news_item(title, summary):
            continue
        records.append(
            NewsRecord(
                ticker=ticker,
                company=company,
                source_domain=urlparse(url).netloc.lower(),
                url=url,
                title=title,
                published_date=item.get("published_date"),
                summary=summary,
                query=item.get("query", ""),
                search_score=item.get("score"),
            )
        )
    return dedupe_news_records(records)


def dedupe_news_records(records: list[NewsRecord]) -> list[NewsRecord]:
    seen: set[tuple[str, str]] = set()
    unique: list[NewsRecord] = []
    for record in records:
        parsed = urlparse(record.url)
        key = (parsed.netloc.lower(), parsed.path.rstrip("/").lower())
        title_key = record.title.strip().lower()
        if key in seen or ("title", title_key) in seen:
            continue
        seen.add(key)
        seen.add(("title", title_key))
        unique.append(record)
    return unique


def reddit_json_url(thread_url: str) -> str:
    clean_url = thread_url.rstrip("/")
    if clean_url.endswith(".json"):
        return clean_url
    return f"{clean_url}.json?limit=100"


def fetch_json(url: str) -> Any:
    request = Request(
        url,
        headers={
            "User-Agent": "hackathon-sentiment-bot/0.1",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_reddit_comments(node: Any) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return comments

    kind = node.get("kind")
    data = node.get("data", {})
    if kind == "t1":
        body = clean_text(data.get("body", ""))
        if is_useful_comment(body):
            comments.append(
                {
                    "author": data.get("author") or "[unknown]",
                    "body": body,
                    "score": data.get("score"),
                    "created_utc": data.get("created_utc"),
                }
            )

        replies = data.get("replies")
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []):
                comments.extend(parse_reddit_comments(child))
        return comments

    for child in data.get("children", []):
        comments.extend(parse_reddit_comments(child))
    return comments


def is_useful_comment(text: str) -> bool:
    if len(text) < 30 or len(text) > 1500:
        return False
    lowered = text.lower()
    banned_markers = [
        "related answers section",
        "create your account",
        "anyone can view, post, and comment",
        "[deleted]",
        "[removed]",
    ]
    return not any(marker in lowered for marker in banned_markers)


def fetch_reddit_threads(search_payload: dict[str, Any]) -> list[dict[str, Any]]:
    thread_payloads: list[dict[str, Any]] = []
    for item in search_payload.get("results", []):
        url = item.get("url", "")
        if not is_social_url(url):
            continue
        if "reddit.com" not in urlparse(url).netloc.lower():
            continue

        try:
            payload = fetch_json(reddit_json_url(url))
        except Exception as exc:
            thread_payloads.append(
                {
                    "url": url,
                    "query": item.get("query", ""),
                    "discovery_score": item.get("score"),
                    "error": str(exc),
                }
            )
            continue

        thread_payloads.append(
            {
                "url": url,
                "query": item.get("query", ""),
                "discovery_score": item.get("score"),
                "payload": payload,
            }
        )
    return thread_payloads


def subreddit_allowed(subreddit: str | None) -> bool:
    if not subreddit:
        return False
    normalized = subreddit.lower()
    if normalized in REDDIT_BLOCKED_SUBREDDITS:
        return False
    return normalized in REDDIT_SUBREDDITS or len(normalized) <= 6


def build_company_terms(ticker: str, company: str) -> set[str]:
    normalized_company = company.lower()
    terms = {
        ticker.lower(),
        f"${ticker.lower()}",
        normalized_company,
    }
    for part in re.split(r"[\s&.,-]+", normalized_company):
        if len(part) >= 3:
            terms.add(part)
    return {term for term in terms if term}


def score_social_relevance(
    ticker: str,
    company: str,
    subreddit: str | None,
    thread_title: str,
    body: str,
    author: str,
) -> int:
    if author == "AutoModerator":
        return 0

    score = 0
    combined = f"{thread_title} {body}".lower()
    company_terms = build_company_terms(ticker, company)

    for term in company_terms:
        if term in combined:
            score += 2

    for term in SOCIAL_SIGNAL_TERMS:
        if term in combined:
            score += 1

    if subreddit and subreddit.lower() in REDDIT_SUBREDDITS:
        score += 1
    if subreddit and subreddit.lower() in {ticker.lower(), company.lower(), re.sub(r"[^a-z0-9]+", "", company.lower())}:
        score += 2

    if len(body) >= 80:
        score += 1

    low_signal_markers = [
        "discord invite",
        "hiring",
        "interview",
        "offer",
        "career",
        "layoffs",
        "team matching",
        "elementary school",
    ]
    for marker in low_signal_markers:
        if marker in combined:
            score -= 2

    return score


def dedupe_social_comments(records: list[SocialCommentRecord]) -> list[SocialCommentRecord]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[SocialCommentRecord] = []
    for record in records:
        key = (record.thread_url, record.author, record.body.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def normalize_reddit_comments(
    ticker: str,
    company: str,
    reddit_threads: list[dict[str, Any]],
) -> list[SocialCommentRecord]:
    records: list[SocialCommentRecord] = []

    for thread in reddit_threads:
        payload = thread.get("payload")
        if not isinstance(payload, list) or len(payload) < 2:
            continue

        post_children = payload[0].get("data", {}).get("children", [])
        post_data = post_children[0].get("data", {}) if post_children else {}
        thread_title = post_data.get("title", "")
        subreddit = post_data.get("subreddit")
        if not subreddit_allowed(subreddit):
            continue
        thread_url = thread.get("url", "")

        comment_listing = payload[1]
        for comment in parse_reddit_comments(comment_listing):
            relevance_score = score_social_relevance(
                ticker=ticker,
                company=company,
                subreddit=subreddit,
                thread_title=thread_title,
                body=comment["body"],
                author=comment["author"],
            )
            if relevance_score < 3:
                continue
            records.append(
                SocialCommentRecord(
                    ticker=ticker,
                    company=company,
                    platform="reddit",
                    subreddit=subreddit,
                    thread_url=thread_url,
                    thread_title=thread_title,
                    author=comment["author"],
                    body=comment["body"],
                    score=comment.get("score"),
                    created_utc=comment.get("created_utc"),
                    query=thread.get("query", ""),
                    discovery_score=thread.get("discovery_score"),
                    relevance_score=relevance_score,
                )
            )

    return dedupe_social_comments(records)


def count_unique_threads(records: list[SocialCommentRecord]) -> int:
    return len({record.thread_url for record in records})


def search_result_subreddit_allowed(url: str) -> bool:
    match = re.search(r"reddit\.com/r/([^/]+)/", url, re.IGNORECASE)
    if not match:
        return True
    return subreddit_allowed(match.group(1))


def collect_social_discovery(
    client: TavilyClient,
    ticker: str,
    company: str,
    days: int,
    max_results_per_query: int,
) -> dict[str, Any]:
    queries = build_queries(ticker, company, SOCIAL_QUERY_TEMPLATES)
    search_payload = run_tavily_searches(
        client=client,
        queries=queries,
        days=days,
        max_results_per_query=max_results_per_query,
        include_domains=SOCIAL_DOMAINS,
    )
    search_payload["results"] = [
        item
        for item in search_payload["results"]
        if is_social_url(item.get("url", "")) and search_result_subreddit_allowed(item.get("url", ""))
    ]
    return search_payload


def collect_news_discovery(
    client: TavilyClient,
    ticker: str,
    company: str,
    days: int,
    max_results_per_query: int,
) -> dict[str, Any]:
    queries = build_queries(ticker, company, NEWS_QUERY_TEMPLATES)
    search_payload = run_tavily_searches(
        client=client,
        queries=queries,
        days=days,
        max_results_per_query=max_results_per_query,
        include_domains=NEWS_DOMAINS,
    )
    search_payload["results"] = [item for item in search_payload["results"] if is_news_url(item.get("url", ""))]
    return search_payload


def collect_news(
    client: TavilyClient,
    ticker: str,
    company: str,
    days: int,
    max_results_per_query: int,
) -> dict[str, Any]:
    raw_dir, processed_dir = ensure_dirs()
    news_search = collect_news_discovery(client, ticker, company, days, max_results_per_query)
    news_records = normalize_news_records(ticker, company, news_search)
    news_search_path = raw_dir / f"{ticker}_news_search_results.json"
    news_items_path = processed_dir / f"{ticker}_news_items.jsonl"
    write_json(news_search_path, news_search)
    write_jsonl(news_items_path, [asdict(record) for record in news_records])
    return {
        "search_results": len(news_search.get("results", [])),
        "news_items": len(news_records),
        "search_path": str(news_search_path),
        "items_path": str(news_items_path),
    }


def collect_social(
    client: TavilyClient,
    ticker: str,
    company: str,
    days: int,
    max_results_per_query: int,
) -> dict[str, Any]:
    raw_dir, processed_dir = ensure_dirs()
    social_search = collect_social_discovery(client, ticker, company, days, max_results_per_query)
    reddit_threads = fetch_reddit_threads(social_search)
    social_comments = normalize_reddit_comments(ticker, company, reddit_threads)
    unique_threads = count_unique_threads(social_comments)

    social_search_path = raw_dir / f"{ticker}_social_search_results.json"
    reddit_threads_path = raw_dir / f"{ticker}_reddit_threads.json"
    social_comments_path = processed_dir / f"{ticker}_social_comments.jsonl"

    write_json(social_search_path, social_search)
    write_json(reddit_threads_path, reddit_threads)
    write_jsonl(social_comments_path, [asdict(record) for record in social_comments])

    return {
        "search_results": len(social_search.get("results", [])),
        "reddit_threads_fetched": len([item for item in reddit_threads if "payload" in item]),
        "social_comments": len(social_comments),
        "unique_discussion_pages": unique_threads,
        "subreddit_allowlist": REDDIT_SUBREDDITS,
        "subreddit_blocklist": REDDIT_BLOCKED_SUBREDDITS,
        "recommended_min_discussion_pages": DEFAULT_MIN_DISCUSSION_PAGES,
        "enough_social_data": unique_threads >= DEFAULT_MIN_DISCUSSION_PAGES,
        "search_path": str(social_search_path),
        "reddit_threads_path": str(reddit_threads_path),
        "comments_path": str(social_comments_path),
    }
