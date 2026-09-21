import os

import numpy as np
import pandas as pd
import pytest

from marginalia.data import market
from tests.conftest import business_days, geometric_series


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "prices")


def _fake_download(frame):
    """Return a stand-in for market._download_raw that yields `frame`."""
    def _inner(tickers, start, end):
        cols = [t for t in tickers if t in frame.columns]
        if not cols:
            raise market.MarketDataError("no data")
        sub = frame[cols]
        return sub.loc[(sub.index >= start) & (sub.index <= end)]
    return _inner


def test_get_prices_clean_schema(monkeypatch, tmp_cache):
    n = 300
    frame = pd.DataFrame({
        "AAA": geometric_series(n, 0.001).values,
        "BBB": geometric_series(n, 0.0005).values,
    }, index=business_days(n))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))

    out = market.get_prices(["aaa", "bbb"], use_cache=False, cache_dir=tmp_cache)
    assert list(out.columns) == ["AAA", "BBB"]
    assert not isinstance(out.columns, pd.MultiIndex)
    assert out.index.is_monotonic_increasing
    assert out.index.tz is None


def test_cache_round_trip(monkeypatch, tmp_cache):
    n = 200
    frame = pd.DataFrame({"AAA": geometric_series(n, 0.001).values}, index=business_days(n))

    calls = {"count": 0}
    base = _fake_download(frame)

    def counting(tickers, start, end):
        calls["count"] += 1
        return base(tickers, start, end)

    monkeypatch.setattr(market, "_download_raw", counting)

    # First call downloads and writes cache.
    market.get_prices(["AAA"], use_cache=True, cache_dir=tmp_cache)
    assert calls["count"] == 1
    assert os.path.isfile(os.path.join(tmp_cache, "AAA.csv"))

    # Second call (same range) should be served fully from cache.
    market.get_prices(
        ["AAA"], start=frame.index[0], end=frame.index[-1],
        use_cache=True, cache_dir=tmp_cache,
    )
    assert calls["count"] == 1  # no new download


def test_dedup_tickers(monkeypatch, tmp_cache):
    n = 100
    frame = pd.DataFrame({"AAA": geometric_series(n, 0.001).values}, index=business_days(n))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))
    out = market.get_prices(["AAA", "aaa", "AAA"], use_cache=False, cache_dir=tmp_cache)
    assert list(out.columns) == ["AAA"]


def test_partial_missing_ticker(monkeypatch, tmp_cache):
    n = 100
    frame = pd.DataFrame({"AAA": geometric_series(n, 0.001).values}, index=business_days(n))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))
    # BBB has no data; AAA does -> result has only AAA.
    out = market.get_prices(["AAA", "BBB"], use_cache=False, cache_dir=tmp_cache)
    assert list(out.columns) == ["AAA"]


def test_no_data_raises(monkeypatch, tmp_cache):
    frame = pd.DataFrame({"AAA": [1.0, 2.0]}, index=business_days(2))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))
    with pytest.raises(market.MarketDataError):
        market.get_prices(["ZZZ"], use_cache=False, cache_dir=tmp_cache)


def test_empty_tickers_raises():
    with pytest.raises(market.MarketDataError):
        market.get_prices([], use_cache=False)


def test_start_after_end_raises():
    with pytest.raises(market.MarketDataError):
        market.get_prices(["AAA"], start="2020-01-01", end="2019-01-01", use_cache=False)


def test_get_returns(monkeypatch, tmp_cache):
    n = 100
    frame = pd.DataFrame({"AAA": geometric_series(n, 0.001).values}, index=business_days(n))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))
    rets = market.get_returns(["AAA"], use_cache=False, cache_dir=tmp_cache)
    # constant geometric drift -> constant daily return ~0.001
    assert abs(rets["AAA"].iloc[-1] - 0.001) < 1e-9


def test_extract_close_multiindex():
    idx = business_days(3)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
    raw = pd.DataFrame(np.arange(3 * 4).reshape(3, 4), index=idx, columns=cols)
    out = market._extract_close(raw, ["AAA", "BBB"])
    assert list(out.columns) == ["AAA", "BBB"]


def test_clear_cache(monkeypatch, tmp_cache):
    n = 50
    frame = pd.DataFrame({"AAA": geometric_series(n, 0.001).values}, index=business_days(n))
    monkeypatch.setattr(market, "_download_raw", _fake_download(frame))
    market.get_prices(["AAA"], use_cache=True, cache_dir=tmp_cache)
    removed = market.clear_cache(tmp_cache)
    assert removed >= 1
