from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from agent2.agents.harness import run_paper
from agent2.agents.models import AgenticRunFailed, HarnessDecision
from agent2.agents.roles import (
    AnalystAgent,
    CriticAgent,
    ExtractorAgent,
    NarrativeVerifierAgent,
    RepairAgent,
    StrategyCodegenAgent,
    StrategyDebuggerAgent,
    VerifierAgent,
)
from agent2.ingest import load_agent1_json
from agent2.openrouter import OpenRouterClient
from agent2.settings import get_settings


class FakeHarnesser:
    def __init__(self, scripted: list[dict[str, object]]) -> None:
        self._scripted = iter(scripted)

    def decide(self, snapshot):
        item = next(self._scripted)
        raw = dict(item.get("raw", {}))
        raw_decision = item.get("decision")
        decision = HarnessDecision.model_validate(raw_decision) if raw_decision is not None else None
        metadata = type("Meta", (), {"model": "mock-harness", "latency_seconds": 0.01, "usage": {"total_tokens": 1}})()
        error = str(item.get("error", ""))
        return raw, decision, metadata, error


class DummyRole:
    def run(self, **kwargs):
        raise AssertionError("This role should not have been called in this test")


def _mock_transport(responses: list[dict[str, object]]) -> httpx.MockTransport:
    iterator = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = next(iterator)
        return httpx.Response(
            200,
            json={
                "model": "mock-openrouter",
                "usage": {"total_tokens": 10},
                "choices": [{"message": {"content": json.dumps(payload)}}],
            },
        )

    return httpx.MockTransport(handler)


def _build_role_agents(monkeypatch, responses: list[dict[str, object]]):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()
    client = OpenRouterClient(transport=_mock_transport(responses))
    return {
        "extractor": ExtractorAgent(client=client),
        "verifier": VerifierAgent(client=client),
        "critic": CriticAgent(client=client),
        "repair": RepairAgent(client=client),
        "codegen": StrategyCodegenAgent(client=client),
        "debugger": StrategyDebuggerAgent(client=client),
        "analyst": AnalystAgent(client=client),
        "narrative_verifier": NarrativeVerifierAgent(client=client),
    }


def test_harness_happy_path_writes_trace_and_final_artifacts(tmp_path: Path, monkeypatch) -> None:
    fixture = load_agent1_json("fixtures/spy_dual_ma.json")
    roles = _build_role_agents(
        monkeypatch,
        [
            {
                "blueprint": {
                    "paper_id": fixture.paper_id,
                    "title": fixture.title,
                    "universe": {
                        "tickers": ["SPY"],
                        "benchmark": "SPY",
                        "asset_universe_type": "etfs",
                        "description": "S&P 500 ETF",
                        "notes": "Single ETF proxy",
                    },
                    "benchmark": "SPY",
                    "data_requirements": [
                        {"name": "price_history", "series_type": "adjusted_close", "tickers": ["SPY"], "frequency": "1d", "notes": "Daily adjusted close"}
                    ],
                    "signal_definition": "Go long when SPY is above its 3-day moving average and go flat otherwise.",
                    "rebalance_schedule": "daily",
                    "holding_period": "until the signal turns off",
                    "risk_controls": ["long-only"],
                    "leverage_rules": "No leverage",
                    "timing_and_delay_rules": ["Shift the signal by one bar before applying returns."],
                    "corporate_action_assumptions": ["Use adjusted close prices."],
                    "required_parameters": {"lookback": 3},
                    "open_questions": [],
                    "implementation_notes": ["This is a lightweight paper-specific executable approximation."],
                    "citations": [{"quote": "moving average timing rule", "section": "methodology", "rationale": "core signal"}],
                },
                "evidence_quotes": ["moving average timing rule"],
                "reasoning": "Blueprint grounded in methodology text.",
                "open_questions": [],
            },
            {
                "status": "pass",
                "findings": [],
                "missing_information": [],
                "evidence_quotes": ["The moving average timing rule is explicit."],
                "reasoning": "Blueprint is supported.",
            },
            {
                "code": "\n".join(
                    [
                        "import numpy as np",
                        "import pandas as pd",
                        "",
                        "def run_strategy(prices: pd.DataFrame, cfg: dict) -> dict:",
                        "    asset = prices.columns[0]",
                        "    signal = (prices[asset] > prices[asset].rolling(3).mean()).astype(float).shift(1).fillna(0.0)",
                        "    weights = signal.to_frame(name=asset)",
                        "    gross_returns = signal * prices[asset].pct_change().fillna(0.0)",
                        "    turnover = signal.diff().abs().fillna(signal.abs())",
                        "    diagnostics = {'friction_handling': 'No explicit frictions modeled in this fixture.'}",
                        "    return {'weights': weights, 'gross_returns': gross_returns, 'turnover': turnover, 'diagnostics': diagnostics}",
                    ]
                ),
                "summary": "Paper-specific moving-average timing implementation.",
                "assumptions": ["Uses reader-provided SPY ticker."],
                "known_limitations": ["Simplified lookback for test coverage."],
            },
            {
                "summary": "The generated strategy is a simple SPY timing rule aligned with the paper fixture.",
                "methodology": "Daily moving-average timing with a one-bar lag.",
                "performance_summary": "Returns are computed deterministically from executed outputs.",
                "robustness_summary": "Basic split metrics are available.",
                "key_risks": ["Single asset proxy"],
                "limitations": ["Test fixture, not a production strategy"],
            },
            {
                "status": "pass",
                "findings": [],
                "reasoning": "Narrative stays within the supplied metrics.",
            },
        ],
    )
    harnesser = FakeHarnesser(
        [
            {"raw": {"action": "invoke_tool", "tool_name": "extract_blueprint"}, "decision": {"action": "invoke_tool", "tool_name": "extract_blueprint", "tool_args": {}, "stop": False, "notes": "extract blueprint"}},
            {"raw": {"action": "invoke_tool", "tool_name": "verify_blueprint"}, "decision": {"action": "invoke_tool", "tool_name": "verify_blueprint", "tool_args": {}, "stop": False, "notes": "verify blueprint"}},
            {"raw": {"action": "invoke_tool", "tool_name": "draft_strategy_code"}, "decision": {"action": "invoke_tool", "tool_name": "draft_strategy_code", "tool_args": {}, "stop": False, "notes": "draft strategy code"}},
            {"raw": {"action": "invoke_tool", "tool_name": "static_scan_strategy_code"}, "decision": {"action": "invoke_tool", "tool_name": "static_scan_strategy_code", "tool_args": {}, "stop": False, "notes": "scan"}},
            {"raw": {"action": "invoke_tool", "tool_name": "execute_strategy"}, "decision": {"action": "invoke_tool", "tool_name": "execute_strategy", "tool_args": {}, "stop": False, "notes": "execute"}},
            {"raw": {"action": "invoke_tool", "tool_name": "compute_metrics"}, "decision": {"action": "invoke_tool", "tool_name": "compute_metrics", "tool_args": {}, "stop": False, "notes": "metrics"}},
            {"raw": {"action": "invoke_tool", "tool_name": "analyst_turn"}, "decision": {"action": "invoke_tool", "tool_name": "analyst_turn", "tool_args": {}, "stop": False, "notes": "narrative"}},
            {"raw": {"action": "invoke_tool", "tool_name": "narrative_verify"}, "decision": {"action": "invoke_tool", "tool_name": "narrative_verify", "tool_args": {}, "stop": False, "notes": "verify narrative"}},
            {"raw": {"action": "invoke_tool", "tool_name": "publish_final_artifacts"}, "decision": {"action": "invoke_tool", "tool_name": "publish_final_artifacts", "tool_args": {}, "stop": False, "notes": "publish"}},
            {"raw": {"action": "stop", "terminal_status": "success"}, "decision": {"action": "stop", "tool_args": {}, "stop": True, "terminal_status": "success", "notes": "done"}},
        ]
    )

    index = pd.date_range("2020-01-01", periods=80, freq="B")
    prices = pd.DataFrame({"SPY": [100 + i * 0.5 for i in range(len(index))]}, index=index).astype(float)
    monkeypatch.setattr("agent2.pipeline.fetch_adj_close", lambda tickers, start, end: prices[tickers])

    result = run_paper(
        fixture,
        start=date(2020, 1, 1),
        end=date(2020, 4, 30),
        artifacts_dir=tmp_path,
        max_harness_turns=12,
        max_tool_calls=12,
        harnesser=harnesser,
        **roles,
    )

    run_dir = tmp_path / result.run_id
    assert result.terminal_status == "success"
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "tear_sheet.json").exists()
    assert (run_dir / "strategy.py").exists()
    assert (run_dir / "state.json").exists()
    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 10
    tear_sheet = json.loads((run_dir / "tear_sheet.json").read_text(encoding="utf-8"))
    assert "blueprint" in tear_sheet
    assert "strategy_artifact" in tear_sheet
    assert "metrics" in tear_sheet


def test_harness_budgets_stop_the_loop(tmp_path: Path) -> None:
    fixture = load_agent1_json("fixtures/spy_dual_ma.json")
    harnesser = FakeHarnesser(
        [
            {"raw": {"action": "invoke_tool", "tool_name": "list_artifacts"}, "decision": {"action": "invoke_tool", "tool_name": "list_artifacts", "tool_args": {}, "stop": False, "notes": "loop"}},
            {"raw": {"action": "invoke_tool", "tool_name": "list_artifacts"}, "decision": {"action": "invoke_tool", "tool_name": "list_artifacts", "tool_args": {}, "stop": False, "notes": "loop"}},
            {"raw": {"action": "invoke_tool", "tool_name": "list_artifacts"}, "decision": {"action": "invoke_tool", "tool_name": "list_artifacts", "tool_args": {}, "stop": False, "notes": "loop"}},
        ]
    )

    with pytest.raises(AgenticRunFailed) as excinfo:
        run_paper(
            fixture,
            artifacts_dir=tmp_path,
            max_harness_turns=2,
            max_tool_calls=10,
            harnesser=harnesser,
            extractor=DummyRole(),
            verifier=DummyRole(),
            critic=DummyRole(),
            repair=DummyRole(),
            codegen=DummyRole(),
            debugger=DummyRole(),
            analyst=DummyRole(),
            narrative_verifier=DummyRole(),
        )

    assert "max_harness_turns" in excinfo.value.failure.message


def test_harness_rejects_invalid_decision_and_retries_once(tmp_path: Path) -> None:
    fixture = load_agent1_json("fixtures/spy_dual_ma.json")
    harnesser = FakeHarnesser(
        [
            {"raw": {"action": "invoke_tool", "tool_name": "does_not_exist"}, "decision": {"action": "invoke_tool", "tool_name": "does_not_exist", "tool_args": {}, "stop": False, "notes": "bad tool"}},
            {"raw": {"action": "invoke_tool", "tool_name": "list_artifacts"}, "decision": {"action": "invoke_tool", "tool_name": "list_artifacts", "tool_args": {}, "stop": False, "notes": "valid retry"}},
            {"raw": {"action": "stop", "terminal_status": "failed"}, "decision": {"action": "stop", "tool_args": {}, "stop": True, "terminal_status": "failed", "notes": "intentional stop"}},
        ]
    )

    with pytest.raises(AgenticRunFailed):
        run_paper(
            fixture,
            artifacts_dir=tmp_path,
            max_harness_turns=5,
            max_tool_calls=5,
            harnesser=harnesser,
            extractor=DummyRole(),
            verifier=DummyRole(),
            critic=DummyRole(),
            repair=DummyRole(),
            codegen=DummyRole(),
            debugger=DummyRole(),
            analyst=DummyRole(),
            narrative_verifier=DummyRole(),
        )

    trace_lines = list(tmp_path.glob(f"{fixture.paper_id}-*/trace.jsonl"))
    assert trace_lines, "Expected harness trace"
    payloads = [json.loads(line) for line in trace_lines[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payloads[0]["tool_invoked"] is None
    assert payloads[1]["tool_invoked"] == "list_artifacts"


def test_same_snapshot_can_lead_to_different_tools(tmp_path: Path) -> None:
    fixture = load_agent1_json("fixtures/spy_dual_ma.json")

    def _run_variant(name: str, first_tool: str, tool_args: dict[str, object]):
        harnesser = FakeHarnesser(
            [
                {"raw": {"action": "invoke_tool", "tool_name": first_tool}, "decision": {"action": "invoke_tool", "tool_name": first_tool, "tool_args": tool_args, "stop": False, "notes": "variant"}},
                {"raw": {"action": "stop", "terminal_status": "failed"}, "decision": {"action": "stop", "tool_args": {}, "stop": True, "terminal_status": "failed", "notes": "stop"}},
            ]
        )
        with pytest.raises(AgenticRunFailed):
            run_paper(
                fixture,
                artifacts_dir=tmp_path,
                max_harness_turns=4,
                max_tool_calls=4,
                run_id=name,
                harnesser=harnesser,
                extractor=DummyRole(),
                verifier=DummyRole(),
                critic=DummyRole(),
                repair=DummyRole(),
                codegen=DummyRole(),
                debugger=DummyRole(),
                analyst=DummyRole(),
                narrative_verifier=DummyRole(),
            )
        trace_path = tmp_path / name / "trace.jsonl"
        first = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        return first["tool_invoked"]

    one = _run_variant("variant-read", "list_artifacts", {})
    two = _run_variant("variant-note", "write_note", {"filename": "note.txt", "content": "hello"})
    assert one != two


def test_codegen_budget_exhaustion_is_enforced(tmp_path: Path, monkeypatch) -> None:
    fixture = load_agent1_json("fixtures/spy_dual_ma.json")
    roles = _build_role_agents(
        monkeypatch,
        [
            {
                "blueprint": {
                    "paper_id": fixture.paper_id,
                    "title": fixture.title,
                    "universe": {"tickers": ["SPY"], "benchmark": "SPY"},
                    "benchmark": "SPY",
                    "signal_definition": "Simple timing rule",
                    "rebalance_schedule": "daily",
                    "holding_period": "1 day",
                    "risk_controls": [],
                    "leverage_rules": "",
                    "timing_and_delay_rules": ["Shift by one bar."],
                    "corporate_action_assumptions": ["Adjusted close."],
                    "required_parameters": {"lookback": 3},
                    "open_questions": [],
                    "implementation_notes": [],
                    "citations": [{"quote": "timing", "section": "methodology", "rationale": "signal"}],
                },
                "evidence_quotes": ["timing"],
                "reasoning": "ok",
                "open_questions": [],
            },
            {"code": "import pandas as pd\n\ndef run_strategy(prices: pd.DataFrame, cfg: dict) -> dict:\n    return {'weights': prices[['SPY']]*0.0, 'gross_returns': prices['SPY']*0.0}\n", "summary": "first", "assumptions": [], "known_limitations": []},
            {"code": "import pandas as pd\n\ndef run_strategy(prices: pd.DataFrame, cfg: dict) -> dict:\n    return {'weights': prices[['SPY']]*0.0, 'gross_returns': prices['SPY']*0.0}\n", "summary": "second", "assumptions": [], "known_limitations": []},
        ],
    )
    harnesser = FakeHarnesser(
        [
            {"raw": {"action": "invoke_tool", "tool_name": "extract_blueprint"}, "decision": {"action": "invoke_tool", "tool_name": "extract_blueprint", "tool_args": {}, "stop": False, "notes": "extract"}},
            {"raw": {"action": "invoke_tool", "tool_name": "draft_strategy_code"}, "decision": {"action": "invoke_tool", "tool_name": "draft_strategy_code", "tool_args": {}, "stop": False, "notes": "draft one"}},
            {"raw": {"action": "invoke_tool", "tool_name": "revise_strategy_code"}, "decision": {"action": "invoke_tool", "tool_name": "revise_strategy_code", "tool_args": {}, "stop": False, "notes": "draft two"}},
            {"raw": {"action": "invoke_tool", "tool_name": "revise_strategy_code"}, "decision": {"action": "invoke_tool", "tool_name": "revise_strategy_code", "tool_args": {}, "stop": False, "notes": "should fail on budget"}},
        ]
    )

    with pytest.raises(AgenticRunFailed) as excinfo:
        run_paper(
            fixture,
            artifacts_dir=tmp_path,
            max_harness_turns=8,
            max_tool_calls=8,
            max_codegen_attempts=2,
            harnesser=harnesser,
            **roles,
        )

    assert "invalid decisions" in excinfo.value.failure.message or "codegen budget exhausted" in excinfo.value.failure.message
