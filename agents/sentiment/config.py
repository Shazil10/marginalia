from __future__ import annotations

SOCIAL_DOMAINS = [
    "reddit.com",
    "stocktwits.com",
]

NEWS_DOMAINS = [
    "reuters.com",
    "finance.yahoo.com",
    "seekingalpha.com",
    "marketwatch.com",
]

BLOCKED_URL_PATTERNS = [
    "/quote/",
    "/profile/",
]

DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_MAX_RESULTS_PER_QUERY = 8
DEFAULT_MIN_DISCUSSION_PAGES = 3
DEFAULT_MODE = "all"

REDDIT_SUBREDDITS = [
    "stocks",
    "wallstreetbets",
    "investing",
    "stockmarket",
    "securityanalysis",
    "valueinvesting",
    "options",
    "theraceto10million",
]

REDDIT_BLOCKED_SUBREDDITS = [
    "financialcareers",
    "jpmorganchase",
    "careerguidance",
    "jobs",
    "recruitinghell",
    "economics",
    "gold",
    "cryptocurrency",
    "clarval",
]

SOCIAL_SIGNAL_TERMS = [
    "stock",
    "shares",
    "earnings",
    "revenue",
    "guidance",
    "valuation",
    "multiple",
    "buy",
    "sell",
    "bull",
    "bear",
    "long",
    "short",
    "calls",
    "puts",
    "position",
    "upside",
    "downside",
    "target",
    "price target",
    "estimate",
    "analyst",
    "profit",
    "margin",
    "dividend",
]

SOCIAL_QUERY_TEMPLATES = [
    '"${ticker}" "{company}" stock discussion site:reddit.com',
    '"{ticker}" "{company}" site:reddit.com/r/stocks OR site:reddit.com/r/wallstreetbets OR site:reddit.com/r/investing',
    '"{ticker}" "{company}" site:reddit.com/r/StockMarket OR site:reddit.com/r/SecurityAnalysis OR site:reddit.com/r/ValueInvesting',
    '"{ticker}" "{company}" site:reddit.com/r/{ticker}',
    '"{ticker}" "{company}" site:reddit.com/r/{company_slug}',
    '"{ticker}" "{company}" site:stocktwits.com',
]

NEWS_QUERY_TEMPLATES = [
    '"{ticker}" "{company}" stock news',
    '"{ticker}" "{company}" analyst news',
    '"{ticker}" "{company}" earnings outlook',
]
