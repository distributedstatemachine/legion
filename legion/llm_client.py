"""Unified OpenAI-compatible chat client (OpenRouter-ready).

All configuration via env:
- VSCP_LLM=1 enables the real path (selection happens in callers).
- VSCP_LLM_URL (default OpenRouter's chat completions endpoint).
- API key: OPENROUTER_API_KEY, then VSCP_LLM_API_KEY, then OPENAI_API_KEY.
- VSCP_LLM_MODEL (default openai/gpt-4o-mini); VSCP_VERIFIER_MODEL overrides
  for the verifier specifically.
- VSCP_LLM_REFERER / VSCP_LLM_TITLE -> optional HTTP-Referer / X-Title headers
  (OpenRouter ranks/labels by these; harmless if absent).

Policy: temperature 0, 30 s timeout, at most ONE retry on transport errors,
never a retry on a parse/contract failure. The API key is never logged or
included in any error message.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
TIMEOUT_SECONDS = 30


def resolve_api_key() -> str | None:
    for var in ("OPENROUTER_API_KEY", "VSCP_LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def resolve_url() -> str:
    return os.environ.get("VSCP_LLM_URL") or DEFAULT_URL


def resolve_model(role: str = "worker") -> str:
    if role == "verifier":
        override = os.environ.get("VSCP_VERIFIER_MODEL")
        if override:
            return override
    return os.environ.get("VSCP_LLM_MODEL") or DEFAULT_MODEL


def real_path_enabled() -> bool:
    return os.environ.get("VSCP_LLM") == "1" and resolve_api_key() is not None


def _request(url: str, payload: bytes, api_key: str) -> str:
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    referer = os.environ.get("VSCP_LLM_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("VSCP_LLM_TITLE")
    if title:
        headers["X-Title"] = title
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def openai_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0,
) -> tuple[str, dict[str, Any] | None]:
    """One chat completion. Returns (content, usage-or-None).

    Transport errors (network, 5xx) are retried exactly once; a response that
    fails the contract (no choices/content) raises ValueError without retry.
    """
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError("no API key: set OPENROUTER_API_KEY (or VSCP_LLM_API_KEY/OPENAI_API_KEY)")
    url = resolve_url()
    body: dict[str, Any] = {
        "model": model or resolve_model(),
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    payload = json.dumps(body).encode("utf-8")

    last_transport_error: Exception | None = None
    for attempt in range(2):  # initial + at most one transport retry
        try:
            raw = _request(url, payload, api_key)
            break
        except (urllib.error.URLError, OSError) as exc:
            last_transport_error = exc
            if attempt == 1:
                # Never include the key (or headers) in the surfaced error.
                raise RuntimeError(f"LLM transport failed after retry: {type(exc).__name__}") from None
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(f"LLM transport failed: {last_transport_error}")

    try:
        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise ValueError("LLM response did not match the chat-completions contract") from None
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
    return content, usage


def complete_text(prompt: str, *, role: str = "worker") -> tuple[str, dict[str, Any] | None]:
    return openai_chat(
        [{"role": "user", "content": prompt}], model=resolve_model(role)
    )
