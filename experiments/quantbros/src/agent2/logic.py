"""Deprecated legacy logic extraction helpers.

Agentic mode is now the default execution path. These helpers remain only for an explicit
legacy flag and must never be used silently.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from pydantic import ValidationError

from agent2.openrouter import OpenRouterClient, parse_json_text
from agent2.schemas import (
    Agent1ToAgent2Input,
    AssetUniverseType,
    DualMAParams,
    LogicSpec,
    MomentumParams,
    OpenRouterCallMetadata,
    PaperCategory,
    RegimeClaim,
    StrategyKind,
    UniverseSpec,
)


DEFAULT_ETF_BASKET = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD"]


def _require_legacy_opt_in(allow_legacy_fallbacks: bool) -> None:
    if not allow_legacy_fallbacks:
        raise RuntimeError(
            "Legacy heuristic/preset logic extraction is disabled. Use agentic orchestration "
            "or pass the deprecated legacy flag explicitly."
        )
    warnings.warn(
        "Using deprecated legacy logic extraction helpers. Agentic mode should be preferred.",
        DeprecationWarning,
        stacklevel=2,
    )


def _default_universe(agent1: Agent1ToAgent2Input) -> list[str]:
    if agent1.extracted_tickers:
        return agent1.extracted_tickers[:6]
    if agent1.asset_universe_type in {AssetUniverseType.etfs, AssetUniverseType.multi_asset}:
        return DEFAULT_ETF_BASKET
    return ["SPY"]


def preset_logic(name: str, *, paper_id: str = "", allow_legacy_fallbacks: bool = False) -> LogicSpec:
    _require_legacy_opt_in(allow_legacy_fallbacks)
    if name == "spy_dual_ma":
        return LogicSpec(
            paper_id=paper_id,
            strategy=StrategyKind.dual_moving_average,
            universe=UniverseSpec(
                tickers=["SPY"],
                benchmark="SPY",
                asset_universe_type=AssetUniverseType.etfs,
            ),
            dual_ma=DualMAParams(fast_window=20, slow_window=50),
            strategy_category_hint=PaperCategory.trend,
            regime_hints=[RegimeClaim.bull, RegimeClaim.low_volatility],
            source_method="preset",
            notes="Deprecated legacy preset.",
        )
    if name == "etf_momentum_rank":
        return LogicSpec(
            paper_id=paper_id,
            strategy=StrategyKind.momentum_rank,
            universe=UniverseSpec(
                tickers=DEFAULT_ETF_BASKET,
                benchmark="SPY",
                asset_universe_type=AssetUniverseType.etfs,
            ),
            momentum=MomentumParams(lookback_days=126, top_n=3),
            position_sizing="equal_weight",
            strategy_category_hint=PaperCategory.momentum,
            regime_hints=[RegimeClaim.bull, RegimeClaim.mixed],
            source_method="preset",
            notes="Deprecated legacy preset.",
        )
    raise KeyError(f"Unknown preset: {name}")


def _logic_from_raw_metadata(
    agent1: Agent1ToAgent2Input,
    *,
    allow_legacy_fallbacks: bool = False,
) -> LogicSpec | None:
    _require_legacy_opt_in(allow_legacy_fallbacks)
    raw_logic = agent1.raw_metadata.get("logic_spec")
    if isinstance(raw_logic, dict):
        logic = LogicSpec.model_validate(raw_logic)
        return logic.model_copy(update={"paper_id": agent1.paper_id, "source_method": "embedded"})
    return None


def heuristic_logic(
    agent1: Agent1ToAgent2Input,
    *,
    allow_legacy_fallbacks: bool = False,
) -> LogicSpec:
    _require_legacy_opt_in(allow_legacy_fallbacks)
    hinted = _logic_from_raw_metadata(agent1, allow_legacy_fallbacks=True)
    if hinted is not None:
        return hinted

    text = f"{agent1.title}\n{agent1.sections.methodology}\n{agent1.sections.abstract}".lower()
    if "moving average" in text or "crosses above" in text:
        tickers = _default_universe(agent1)[:1]
        return LogicSpec(
            paper_id=agent1.paper_id,
            strategy=StrategyKind.dual_moving_average,
            universe=UniverseSpec(
                tickers=tickers,
                benchmark=tickers[0],
                asset_universe_type=agent1.asset_universe_type,
            ),
            dual_ma=DualMAParams(fast_window=20, slow_window=50),
            strategy_category_hint=PaperCategory.trend if agent1.paper_category == PaperCategory.unknown else agent1.paper_category,
            regime_hints=[agent1.regime_claim] if agent1.regime_claim not in {RegimeClaim.unknown, RegimeClaim.not_specified} else [],
            source_method="heuristic",
            notes="Deprecated legacy heuristic logic.",
        )
    if any(token in text for token in ["momentum", "rank", "cross-sectional", "relative strength", "top "]):
        tickers = _default_universe(agent1)
        return LogicSpec(
            paper_id=agent1.paper_id,
            strategy=StrategyKind.momentum_rank,
            universe=UniverseSpec(
                tickers=tickers,
                benchmark="SPY",
                asset_universe_type=agent1.asset_universe_type,
            ),
            momentum=MomentumParams(
                lookback_days=126,
                top_n=min(3, max(1, len(tickers))),
            ),
            position_sizing="equal_weight",
            strategy_category_hint=PaperCategory.momentum if agent1.paper_category == PaperCategory.unknown else agent1.paper_category,
            regime_hints=[agent1.regime_claim] if agent1.regime_claim not in {RegimeClaim.unknown, RegimeClaim.not_specified} else [],
            source_method="heuristic",
            notes="Deprecated legacy heuristic logic.",
        )
    raise ValueError(f"Legacy heuristic extraction could not map paper {agent1.paper_id!r}")


def llm_logic(agent1: Agent1ToAgent2Input) -> tuple[LogicSpec, OpenRouterCallMetadata]:
    client = OpenRouterClient()
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quant strategy parser. Convert methodology text into a LogicSpec-style "
                    "JSON object using only supported strategy families."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "paper_id": agent1.paper_id,
                        "title": agent1.title,
                        "abstract": agent1.sections.abstract[:2500],
                        "methodology": agent1.sections.methodology[:3500],
                        "supported_strategy_families": [
                            StrategyKind.dual_moving_average.value,
                            StrategyKind.momentum_rank.value,
                        ],
                        "logic_schema": LogicSpec.model_json_schema(),
                    }
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    payload = parse_json_text(response.content)
    logic = LogicSpec.model_validate(payload).model_copy(
        update={"paper_id": agent1.paper_id, "source_method": "openrouter"}
    )
    return logic, OpenRouterCallMetadata(
        model=response.model,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
    )


def extract_logic(
    agent1: Agent1ToAgent2Input,
    *,
    preset: str | None = None,
    use_llm: bool = False,
    allow_legacy_fallbacks: bool = False,
) -> tuple[LogicSpec, OpenRouterCallMetadata | None]:
    _require_legacy_opt_in(allow_legacy_fallbacks)
    if use_llm:
        logic, metadata = llm_logic(agent1)
        return logic, metadata
    if preset:
        return preset_logic(preset, paper_id=agent1.paper_id, allow_legacy_fallbacks=True), None
    return heuristic_logic(agent1, allow_legacy_fallbacks=True), None


def embedded_preset_name(
    agent1: Agent1ToAgent2Input,
    *,
    allow_legacy_fallbacks: bool = False,
) -> str | None:
    _require_legacy_opt_in(allow_legacy_fallbacks)
    hints = agent1.raw_metadata.get("logic_hints")
    if isinstance(hints, dict):
        preset = hints.get("preset")
        if isinstance(preset, str):
            return preset
    return None
