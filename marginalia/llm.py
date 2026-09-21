"""Nebius Token Factory LLM client with cost-aware model tiering.

Two tiers:

* ``FAST``  -> cheap reasoning model, used for almost everything (intake, the
  first extraction attempt, plain-English summaries).
* ``POWER`` -> expensive large model, used only as a fallback when the cheap
  model produces output we can't validate, or for hard documents.

The two models live behind different Token Factory base URLs, so each tier has
its own configured client. All values can be overridden via environment
variables, so swapping models/keys never requires code changes.
"""

from __future__ import annotations

import enum
import os
import re
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    pass


class ModelTier(str, enum.Enum):
    FAST = "fast"
    POWER = "power"


# (env_model_var, default_model, env_base_var, default_base_url)
_TIER_DEFAULTS = {
    ModelTier.FAST: (
        "NEBIUS_FAST_MODEL",
        "nvidia/Cosmos3-Super-Reasoner",
        "NEBIUS_FAST_BASE_URL",
        "https://api.tokenfactory.nebius.com/v1/",
    ),
    ModelTier.POWER: (
        "NEBIUS_POWER_MODEL",
        "nvidia/Nemotron-3-Ultra-550b-a55b",
        "NEBIUS_POWER_BASE_URL",
        "https://api.tokenfactory.us-central1.nebius.com/v1/",
    ),
}

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0

# Cosmos3/Nemotron are slow reasoning models (~30-60 tok/s). A long request that
# also emits a <think> block can easily run past a default connect/read timeout,
# which surfaces as an APIConnectionError. Give the HTTP client a generous budget.
_REQUEST_TIMEOUT = float(os.getenv("NEBIUS_TIMEOUT_SECONDS", "180"))

_clients = {}  # tier -> OpenAI client (lazy, cached)


def _resolve(tier: ModelTier):
    model_var, model_def, base_var, base_def = _TIER_DEFAULTS[tier]
    model = os.getenv(model_var, model_def)
    base_url = os.getenv(base_var, base_def)
    return model, base_url


def get_client(tier: ModelTier = ModelTier.FAST):
    """Return (and cache) a configured OpenAI-compatible client for a tier."""
    if tier in _clients:
        return _clients[tier]

    from openai import OpenAI

    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise LLMError("NEBIUS_API_KEY not found in environment / .env")

    _, base_url = _resolve(tier)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_REQUEST_TIMEOUT,
        max_retries=2,
    )
    _clients[tier] = client
    return client


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat(
    system: str,
    user: str,
    *,
    tier: ModelTier = ModelTier.FAST,
    temperature: float = 0.1,
    max_tokens: Optional[int] = 4096,
) -> str:
    """Single-turn chat completion. Returns cleaned text content.

    Retries transient failures with backoff. Raises ``LLMError`` on persistent
    failure so callers can decide whether to escalate to a higher tier.
    """
    client = get_client(tier)
    model, _ = _resolve(tier)

    last_err: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return _strip_reasoning(content)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * attempt)
    raise LLMError(f"{tier.value} model call failed after {_MAX_RETRIES} attempts: {last_err}")


def chat_with_escalation(
    system: str,
    user: str,
    *,
    validate=None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = 4096,
) -> str:
    """Try the FAST tier first; escalate to POWER if the result fails ``validate``.

    ``validate`` is an optional callable ``str -> bool``. If provided and the FAST
    result fails it, the POWER tier is tried. Returns the best available text.
    """
    fast_text = None
    try:
        fast_text = chat(system, user, tier=ModelTier.FAST,
                         temperature=temperature, max_tokens=max_tokens)
        if validate is None or validate(fast_text):
            return fast_text
    except LLMError:
        pass

    # Escalate
    power_text = chat(system, user, tier=ModelTier.POWER,
                     temperature=temperature, max_tokens=max_tokens)
    if validate is None or validate(power_text) or fast_text is None:
        return power_text
    return fast_text


def ping(tier: ModelTier = ModelTier.FAST) -> str:
    """Tiny connectivity check. Returns the model's reply to 'ping'."""
    return chat(
        "You are a health check. Reply with exactly the word: pong",
        "ping",
        tier=tier,
        temperature=0.0,
        max_tokens=16,
    )
