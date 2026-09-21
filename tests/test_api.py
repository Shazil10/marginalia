import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marginalia.api.app import create_app
from marginalia import llm as llm_mod
from marginalia.engine import backtest as backtest_mod
from tests.conftest import business_days, geometric_series


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def synthetic_prices():
    n = 600
    idx = business_days(n)
    df = pd.DataFrame(index=idx)
    for t, drift in [("AAA", 0.0010), ("BBB", 0.0004), ("BIL", 0.00005), ("SPY", 0.0004)]:
        df[t] = geometric_series(n, drift).values
    df.index.name = "date"
    return df


@pytest.fixture
def patch_prices(monkeypatch, synthetic_prices):
    monkeypatch.setattr(backtest_mod, "get_prices", lambda *a, **k: synthetic_prices)


SECTOR_SPEC = {
    "strategy_name": "Sector Momentum Rotation",
    "template": "sector_rotation",
    "universe": ["AAA", "BBB"],
    "benchmark": "SPY",
    "safe_asset": "BIL",
    "rebalance": "monthly",
    "parameters": {"lookback": 40, "top_n": 1, "trend_sma": 50},
}
ROUTE_OK = {
    "matched_template": "sector_rotation",
    "confidence": 0.85,
    "category": "momentum",
    "needs_new_template": False,
}
PROFILE = {"risk_class": "moderate", "capital": 25000}


def _patch_llm(monkeypatch):
    def fake_escalation(system, user, **kw):
        if "classifier" in system.lower() or "matched_template" in system.lower():
            return json.dumps(ROUTE_OK)
        return json.dumps(SECTOR_SPEC)

    def fake_chat(system, user, **kw):
        return json.dumps(PROFILE)

    monkeypatch.setattr(llm_mod, "chat_with_escalation", fake_escalation)
    monkeypatch.setattr(llm_mod, "chat", fake_chat)


# -- deterministic endpoints (no LLM) ---------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "sector_rotation" in body["runnable_templates"]


def test_templates(client):
    r = client.get("/templates")
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["templates"]]
    assert "sector_rotation" in ids


def test_backtest_spec_ok(client, patch_prices):
    r = client.post("/backtest-spec", json={
        "spec": SECTOR_SPEC, "optimize": False, "cost_bps": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert body["template"] == "sector_rotation"


def test_backtest_spec_invalid(client):
    r = client.post("/backtest-spec", json={
        "spec": {"strategy_name": "x", "template": "not_real", "universe": ["AAA"]},
    })
    assert r.status_code == 422


# -- LLM endpoints (mocked) -------------------------------------------------

def test_intake(client, monkeypatch):
    _patch_llm(monkeypatch)
    r = client.post("/intake", json={"user_input": "moderate risk, 25k"})
    assert r.status_code == 200
    assert r.json()["risk_class"] == "moderate"


def test_route(client, monkeypatch):
    _patch_llm(monkeypatch)
    r = client.post("/route", json={"paper_text": "A" * 200})
    assert r.status_code == 200
    assert r.json()["matched_template"] == "sector_rotation"


def test_analyze_paper(client, monkeypatch, patch_prices):
    _patch_llm(monkeypatch)
    r = client.post("/analyze-paper", json={
        "paper_text": "Paper " * 50,
        "user_input": "moderate risk, 25k",
        "optimize": False,
        "cost_bps": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["spec"]["template"] == "sector_rotation"
    assert body["backtest"] is not None
    assert body["sizing"] is not None
