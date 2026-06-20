"""Thin wedge for Anthropic / OpenAI LLM calls used in examples and bench.

The core library (recover.py, core.py, etc.) never imports this module.
It exists so examples and RecoveryBench can share one place that knows
about both SDKs without pulling them into the zero-dep core.
"""
from __future__ import annotations

from typing import Any


def _anthropic_complete(messages: list[dict], model: str, **kwargs: Any) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=kwargs.get("max_tokens", 1024),
        messages=messages,
    )
    return response.content[0].text


def _openai_complete(messages: list[dict], model: str, **kwargs: Any) -> str:
    try:
        import openai
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def complete(
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    provider: str | None = None,
    **kwargs: Any,
) -> str:
    """Call an LLM and return the assistant text.

    Provider is inferred from the model name when not specified:
    - claude-* → Anthropic
    - gpt-* / o1-* / o3-* → OpenAI
    """
    if provider is None:
        provider = "anthropic" if model.startswith("claude") else "openai"

    if provider == "anthropic":
        return _anthropic_complete(messages, model, **kwargs)
    if provider == "openai":
        return _openai_complete(messages, model, **kwargs)

    raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")
