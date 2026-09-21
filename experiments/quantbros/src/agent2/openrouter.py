"""Minimal OpenRouter client for schema-constrained extraction tasks."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from agent2.settings import get_settings


LOGGER = logging.getLogger(__name__)


@dataclass
class OpenRouterResult:
    content: str
    model: str
    latency_seconds: float
    usage: dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        *,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        self._api_key = settings.openrouter_api_key
        self._model = settings.openrouter_model
        self._base_url = settings.openrouter_base_url.rstrip("/")
        self._app_name = settings.openrouter_app_name
        self._site_url = settings.openrouter_site_url
        self._timeout = timeout if timeout is not None else settings.openrouter_timeout_seconds
        self._transport = transport
        self._max_retries = max(0, settings.openrouter_max_retries)
        self._retry_backoff_seconds = max(0.0, settings.openrouter_retry_backoff_seconds)

    @property
    def default_model(self) -> str:
        return self._model

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> OpenRouterResult:
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": self._app_name,
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url

        start = time.perf_counter()
        response: httpx.Response | None = None
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            for attempt in range(self._max_retries + 1):
                try:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    break
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt >= self._max_retries:
                        raise
                    sleep_seconds = self._retry_backoff_seconds * (attempt + 1)
                    LOGGER.warning(
                        "OpenRouter transport error on attempt %s/%s for model=%s: %s; retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        payload["model"],
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                    if not retryable or attempt >= self._max_retries:
                        raise
                    sleep_seconds = self._retry_backoff_seconds * (attempt + 1)
                    LOGGER.warning(
                        "OpenRouter HTTP %s on attempt %s/%s for model=%s; retrying in %.1fs",
                        status_code,
                        attempt + 1,
                        self._max_retries + 1,
                        payload["model"],
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
        if response is None:  # pragma: no cover - defensive guard
            raise RuntimeError("OpenRouter request did not produce a response.")
        latency = time.perf_counter() - start

        data = response.json()
        choice = data["choices"][0]["message"]
        content = choice.get("content", "")
        usage = data.get("usage", {})
        actual_model = data.get("model", payload["model"])
        LOGGER.info(
            "OpenRouter call model=%s latency=%.2fs usage=%s",
            actual_model,
            latency,
            usage,
        )
        return OpenRouterResult(
            content=content,
            model=actual_model,
            latency_seconds=latency,
            usage=usage,
        )


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating fenced code blocks."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)
