"""OpenRouter-backed specialist role implementations for the harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent2.agents.base import OpenRouterRoleAgent
from agent2.agents.models import (
    AgentRole,
    BlueprintCritiqueResponse,
    BlueprintExtractionResponse,
    BlueprintRepairResponse,
    BlueprintVerificationResponse,
    NarrativeDraft,
    NarrativeReview,
    StrategyBlueprint,
    StrategyCodeResponse,
    StrategyExecutionResult,
    MetricBundle,
)
from agent2.schemas import Agent1ToAgent2Input


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _paper_payload(agent1: Agent1ToAgent2Input) -> dict[str, Any]:
    return {
        "paper_id": agent1.paper_id,
        "title": agent1.title,
        "source_path": agent1.source_path,
        "source_url": agent1.source_url,
        "abstract": agent1.sections.abstract[:3500],
        "methodology": agent1.sections.methodology[:8000],
        "introduction": agent1.sections.introduction[:3000],
        "conclusion": agent1.sections.conclusion[:3000],
        "notes": agent1.sections.notes[:1500],
        "extracted_tickers": agent1.extracted_tickers,
        "reader_metadata": agent1.raw_metadata,
    }


class ExtractorAgent(OpenRouterRoleAgent):
    role = AgentRole.extractor
    response_model = BlueprintExtractionResponse
    system_prompt = _load_prompt("extractor.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        prior_blueprint: StrategyBlueprint | None = None,
        targeted_questions: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["targeted_questions"] = targeted_questions or []
        payload["prior_blueprint"] = prior_blueprint.model_dump(mode="json") if prior_blueprint else None
        payload["response_schema"] = BlueprintExtractionResponse.model_json_schema()
        return payload


class VerifierAgent(OpenRouterRoleAgent):
    role = AgentRole.verifier
    response_model = BlueprintVerificationResponse
    system_prompt = _load_prompt("verifier.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint | None,
        failure_context: str = "",
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["candidate_blueprint"] = blueprint.model_dump(mode="json") if blueprint else None
        payload["failure_context"] = failure_context
        payload["response_schema"] = BlueprintVerificationResponse.model_json_schema()
        return payload


class CriticAgent(OpenRouterRoleAgent):
    role = AgentRole.critic
    response_model = BlueprintCritiqueResponse
    system_prompt = _load_prompt("critic.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint | None,
        stage: str,
        code: str = "",
        execution: StrategyExecutionResult | None = None,
        metrics: MetricBundle | None = None,
        narrative: NarrativeDraft | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["candidate_blueprint"] = blueprint.model_dump(mode="json") if blueprint else None
        payload["stage"] = stage
        payload["strategy_code"] = code
        payload["execution"] = execution.model_dump(mode="json") if execution else None
        payload["metrics"] = metrics.model_dump(mode="json") if metrics else None
        payload["narrative"] = narrative.model_dump(mode="json") if narrative else None
        payload["response_schema"] = BlueprintCritiqueResponse.model_json_schema()
        return payload


class RepairAgent(OpenRouterRoleAgent):
    role = AgentRole.repair
    response_model = BlueprintRepairResponse
    system_prompt = _load_prompt("repair.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint | None,
        validation_error: str = "",
        verifier: BlueprintVerificationResponse | None = None,
        critique: BlueprintCritiqueResponse | None = None,
        strategy_code: str = "",
        execution: StrategyExecutionResult | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["current_blueprint"] = blueprint.model_dump(mode="json") if blueprint else None
        payload["validation_error"] = validation_error
        payload["verifier"] = verifier.model_dump(mode="json") if verifier else None
        payload["critique"] = critique.model_dump(mode="json") if critique else None
        payload["strategy_code"] = strategy_code
        payload["execution"] = execution.model_dump(mode="json") if execution else None
        payload["response_schema"] = BlueprintRepairResponse.model_json_schema()
        return payload


class StrategyCodegenAgent(OpenRouterRoleAgent):
    role = AgentRole.codegen
    response_model = StrategyCodeResponse
    system_prompt = _load_prompt("codegen.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint,
        critique: BlueprintCritiqueResponse | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["blueprint"] = blueprint.model_dump(mode="json")
        payload["critique"] = critique.model_dump(mode="json") if critique else None
        payload["response_schema"] = StrategyCodeResponse.model_json_schema()
        payload["code_contract"] = {
            "entrypoint": "run_strategy(prices: pd.DataFrame, cfg: dict) -> dict",
            "required_keys": ["weights", "gross_returns"],
            "optional_keys": ["net_returns", "turnover", "diagnostics"],
        }
        return payload


class StrategyDebuggerAgent(OpenRouterRoleAgent):
    role = AgentRole.debugger
    response_model = StrategyCodeResponse
    system_prompt = _load_prompt("code_repair.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint,
        current_code: str,
        validation_error: str = "",
        execution: StrategyExecutionResult | None = None,
        critique: BlueprintCritiqueResponse | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["blueprint"] = blueprint.model_dump(mode="json")
        payload["current_code"] = current_code
        payload["validation_error"] = validation_error
        payload["execution"] = execution.model_dump(mode="json") if execution else None
        payload["critique"] = critique.model_dump(mode="json") if critique else None
        payload["response_schema"] = StrategyCodeResponse.model_json_schema()
        payload["code_contract"] = {
            "entrypoint": "run_strategy(prices: pd.DataFrame, cfg: dict) -> dict",
            "required_keys": ["weights", "gross_returns"],
            "optional_keys": ["net_returns", "turnover", "diagnostics"],
        }
        return payload


class AnalystAgent(OpenRouterRoleAgent):
    role = AgentRole.analyst
    response_model = NarrativeDraft
    system_prompt = _load_prompt("analyst.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint,
        strategy_artifact_summary: dict[str, Any],
        metrics: MetricBundle,
        critique: BlueprintCritiqueResponse | None = None,
        prior_narrative: NarrativeDraft | None = None,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["blueprint"] = blueprint.model_dump(mode="json")
        payload["strategy_artifact_summary"] = strategy_artifact_summary
        payload["metrics"] = metrics.model_dump(mode="json")
        payload["critique"] = critique.model_dump(mode="json") if critique else None
        payload["prior_narrative"] = prior_narrative.model_dump(mode="json") if prior_narrative else None
        payload["response_schema"] = NarrativeDraft.model_json_schema()
        return payload


class NarrativeVerifierAgent(OpenRouterRoleAgent):
    role = AgentRole.narrative_verifier
    response_model = NarrativeReview
    system_prompt = _load_prompt("narrative_verify.md")

    def build_payload(
        self,
        *,
        agent1: Agent1ToAgent2Input,
        blueprint: StrategyBlueprint,
        metrics: MetricBundle,
        narrative: NarrativeDraft,
    ) -> dict[str, Any]:
        payload = _paper_payload(agent1)
        payload["blueprint"] = blueprint.model_dump(mode="json")
        payload["metrics"] = metrics.model_dump(mode="json")
        payload["narrative"] = narrative.model_dump(mode="json")
        payload["response_schema"] = NarrativeReview.model_json_schema()
        return payload
