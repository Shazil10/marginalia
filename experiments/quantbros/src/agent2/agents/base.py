"""Shared helpers for OpenRouter-backed specialist agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from agent2.openrouter import OpenRouterClient, parse_json_text
from agent2.schemas import OpenRouterCallMetadata

from agent2.agents.models import AgentRole


@dataclass
class RoleInvocationOutcome:
    role: AgentRole
    success: bool
    raw_content: str
    parsed: BaseModel | None
    metadata: OpenRouterCallMetadata | None
    validation_error: str = ""


class OpenRouterRoleAgent:
    role: AgentRole
    response_model: Type[BaseModel]
    system_prompt: str
    schema_repair_attempts: int = 1

    def __init__(self, *, client: OpenRouterClient | None = None) -> None:
        self.client = client or OpenRouterClient()

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def run(self, **kwargs: Any) -> RoleInvocationOutcome:
        payload = self.build_payload(**kwargs)
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
        )
        return self._parse_response(
            raw_content=response.content,
            metadata=OpenRouterCallMetadata(
                model=response.model,
                latency_seconds=response.latency_seconds,
                usage=response.usage,
            ),
            payload=payload,
        )

    def _parse_response(
        self,
        *,
        raw_content: str,
        metadata: OpenRouterCallMetadata,
        payload: dict[str, Any],
    ) -> RoleInvocationOutcome:
        current_content = raw_content
        current_metadata = metadata
        last_error = ""
        for attempt in range(self.schema_repair_attempts + 1):
            try:
                parsed_json = parse_json_text(current_content)
                parsed = self.response_model.model_validate(parsed_json)
                return RoleInvocationOutcome(
                    role=self.role,
                    success=True,
                    raw_content=current_content,
                    parsed=parsed,
                    metadata=current_metadata,
                )
            except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt >= self.schema_repair_attempts:
                    break
                repaired = self.client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are SchemaRepairAgent. Return JSON only. Rewrite the invalid "
                                "response so it matches the target schema exactly. Preserve intent, "
                                "do not add unsupported strategy assumptions, and keep missing values minimal."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "role": self.role.value,
                                    "invalid_response": current_content,
                                    "validation_error": last_error,
                                    "target_schema": self.response_model.model_json_schema(),
                                    "original_request_payload": payload,
                                }
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                current_content = repaired.content
                current_metadata = OpenRouterCallMetadata(
                    model=repaired.model,
                    latency_seconds=repaired.latency_seconds,
                    usage=repaired.usage,
                )
        return RoleInvocationOutcome(
            role=self.role,
            success=False,
            raw_content=current_content,
            parsed=None,
            metadata=current_metadata,
            validation_error=last_error,
        )
