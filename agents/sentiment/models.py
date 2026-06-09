from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SocialCommentRecord:
    ticker: str
    company: str
    platform: str
    subreddit: str | None
    thread_url: str
    thread_title: str
    author: str
    body: str
    score: int | None
    created_utc: float | None
    query: str
    discovery_score: float | None
    relevance_score: int


@dataclass
class NewsRecord:
    ticker: str
    company: str
    source_domain: str
    url: str
    title: str
    published_date: str | None
    summary: str
    query: str
    search_score: float | None


@dataclass
class SentimentNewsItem:
    source: str
    title: str
    summary: str
    url: str
    published_date: str | None
    search_score: float | None


@dataclass
class SentimentSocialItem:
    platform: str
    subreddit: str | None
    thread_title: str
    thread_url: str
    author: str
    body: str
    score: int | None
    created_utc: float | None
    discovery_score: float | None
    relevance_score: int | None
