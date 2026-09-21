"""Cached, schema-clean market data adapter.

Everything downstream (engine, backtests, reports) pulls prices ONLY through
``get_prices`` / ``get_returns``. This guarantees:

* A single, predictable schema: a DataFrame indexed by a tz-naive DatetimeIndex
  with one column per ticker holding adjusted close prices. Never a MultiIndex.
* On-disk caching so repeated runs / retries don't re-hit yfinance.
* Retries with backoff around the (flaky) network call.

The actual network fetch is isolated in ``_download_raw`` so tests can monkeypatch
it and exercise all caching / cleaning logic offline.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache", "prices")

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


class MarketDataError(RuntimeError):
    """Raised when usable price data cannot be obtained for any ticker."""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _normalize_tickers(tickers: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(tickers, str):
        raw = [tickers]
    else:
        raw = list(tickers)
    seen = {}
    for t in raw:
        if t is None:
            continue
        clean = str(t).strip().upper()
        if clean:
            seen[clean] = None  # dict preserves insertion order, dedupes
    return list(seen.keys())


def _to_timestamp(value, default: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return default
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{ticker}.csv")


def _read_cache(ticker: str, cache_dir: str) -> Optional[pd.Series]:
    path = _cache_path(ticker, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    if df.empty or df.shape[1] == 0:
        return None
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = ticker
    return s if len(s) else None


def _write_cache(ticker: str, series: pd.Series, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    out = series.copy()
    out.index.name = "Date"
    out.name = "close"
    out.to_csv(_cache_path(ticker, cache_dir))


# ----------------------------------------------------------------------------
# Network fetch (isolated for testability)
# ----------------------------------------------------------------------------

def _download_raw(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance.

    Returns a DataFrame indexed by date with one column per *successfully*
    downloaded ticker. Tickers with no data are simply absent from the result.
    This function is monkeypatched out in unit tests.
    """
    import yfinance as yf

    # yfinance 'end' is exclusive for daily bars; pad by a day so 'end' is included.
    fetch_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_start = start.strftime("%Y-%m-%d")

    last_err: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers,
                start=fetch_start,
                end=fetch_end,
                progress=False,
                auto_adjust=True,
                group_by="column",
                threads=True,
            )
            if raw is None or raw.empty:
                raise MarketDataError("yfinance returned an empty frame")
            return _extract_close(raw, tickers)
        except MarketDataError:
            raise
        except Exception as exc:  # network/parse errors are retried
            last_err = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise MarketDataError(f"yfinance download failed after {_MAX_RETRIES} attempts: {last_err}")


def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Pull just the Close prices out of yfinance's varied output shapes."""
    if isinstance(raw.columns, pd.MultiIndex):
        # Columns are (field, ticker) or (ticker, field) depending on group_by.
        level0 = set(raw.columns.get_level_values(0))
        if "Close" in level0:
            close = raw["Close"]
        else:
            # group_by='ticker' style -> select Close on the second level
            close = raw.xs("Close", axis=1, level=1)
        if isinstance(close, pd.Series):
            close = close.to_frame()
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
            # single ticker case -> name the column after the ticker
            if len(tickers) == 1:
                close.columns = [tickers[0]]
        else:
            close = raw.copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.columns = [str(c).upper() for c in close.columns]
    return close


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def get_prices(
    tickers: Union[str, Sequence[str]],
    start=None,
    end=None,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    min_rows: int = 2,
) -> pd.DataFrame:
    """Return adjusted close prices as a clean (dates x tickers) DataFrame.

    Parameters
    ----------
    tickers : str | sequence of str
        One or more tickers. Case-insensitive; deduped; order preserved.
    start, end : date-like or None
        Inclusive date bounds. ``start`` defaults to 2005-01-01, ``end`` to today.
    cache_dir : str
        Where per-ticker CSV caches live.
    use_cache : bool
        If False, always re-download (and refresh the cache).
    min_rows : int
        Minimum number of rows required for the result to be considered usable.

    Raises
    ------
    MarketDataError
        If no ticker yields usable data.
    """
    tickers = _normalize_tickers(tickers)
    if not tickers:
        raise MarketDataError("No valid tickers provided")

    start_ts = _to_timestamp(start, pd.Timestamp("2005-01-01"))
    end_ts = _to_timestamp(end, pd.Timestamp.today().normalize())
    if start_ts > end_ts:
        raise MarketDataError(f"start ({start_ts.date()}) is after end ({end_ts.date()})")

    series_by_ticker = {}
    to_download: List[str] = []

    for t in tickers:
        cached = _read_cache(t, cache_dir) if use_cache else None
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            series_by_ticker[t] = cached
        else:
            to_download.append(t)

    if to_download:
        # Download a generous window so the cache is useful for future calls.
        dl_start = min([start_ts] + [
            s.index.min() for t, s in series_by_ticker.items() if s is not None
        ])
        downloaded = _download_raw(to_download, dl_start, end_ts)
        for t in to_download:
            if t in downloaded.columns:
                s = pd.to_numeric(downloaded[t], errors="coerce").dropna()
                s = s[~s.index.duplicated(keep="last")].sort_index()
                s.name = t
                if len(s):
                    series_by_ticker[t] = s
                    if use_cache:
                        # merge with any existing cache to extend history
                        prior = _read_cache(t, cache_dir)
                        merged = s if prior is None else (
                            pd.concat([prior, s])
                            .pipe(lambda x: x[~x.index.duplicated(keep="last")])
                            .sort_index()
                        )
                        _write_cache(t, merged, cache_dir)

    if not series_by_ticker:
        raise MarketDataError(f"No usable price data for any of: {tickers}")

    frame = pd.DataFrame({t: series_by_ticker[t] for t in tickers if t in series_by_ticker})
    frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
    frame = frame.sort_index()
    frame = frame.dropna(how="all")

    if len(frame) < min_rows:
        raise MarketDataError(
            f"Insufficient price history ({len(frame)} rows) for {list(frame.columns)} "
            f"between {start_ts.date()} and {end_ts.date()}"
        )

    frame.index.name = "date"
    return frame


def get_returns(
    tickers: Union[str, Sequence[str]],
    start=None,
    end=None,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """Daily simple returns for the given tickers (same clean schema as prices)."""
    prices = get_prices(
        tickers, start, end, cache_dir=cache_dir, use_cache=use_cache
    )
    rets = prices.pct_change()
    if dropna:
        rets = rets.dropna(how="all")
    return rets


def clear_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> int:
    """Delete all cached price files. Returns the number of files removed."""
    if not os.path.isdir(cache_dir):
        return 0
    removed = 0
    for name in os.listdir(cache_dir):
        if name.endswith(".csv"):
            try:
                os.remove(os.path.join(cache_dir, name))
                removed += 1
            except OSError:
                pass
    return removed
