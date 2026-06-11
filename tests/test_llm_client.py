"""Offline tests for the unified OpenAI-compatible client (no network)."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from legion import llm_client


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _ok_response(content="hello", usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return FakeResponse(json.dumps(body).encode("utf-8"))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-key")
    monkeypatch.delenv("VSCP_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VSCP_LLM_URL", raising=False)
    monkeypatch.delenv("VSCP_LLM_MODEL", raising=False)
    monkeypatch.delenv("VSCP_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("VSCP_LLM_REFERER", raising=False)
    monkeypatch.delenv("VSCP_LLM_TITLE", raising=False)
    return monkeypatch


def test_request_shape_and_usage_parsing(env):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _ok_response("answer text", usage={"prompt_tokens": 11, "completion_tokens": 7})

    env.setattr(urllib.request, "urlopen", fake_urlopen)
    text, usage = llm_client.openai_chat([{"role": "user", "content": "q"}])
    assert text == "answer text"
    assert usage == {"prompt_tokens": 11, "completion_tokens": 7}
    assert captured["url"] == llm_client.DEFAULT_URL
    assert captured["body"]["model"] == llm_client.DEFAULT_MODEL
    assert captured["body"]["temperature"] == 0
    assert captured["timeout"] == llm_client.TIMEOUT_SECONDS
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer sk-test-secret-key"
    assert "http-referer" not in headers and "x-title" not in headers


def test_optional_openrouter_headers_forwarded(env):
    env.setenv("VSCP_LLM_REFERER", "https://example.test")
    env.setenv("VSCP_LLM_TITLE", "legion-eval")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        return _ok_response()

    env.setattr(urllib.request, "urlopen", fake_urlopen)
    llm_client.openai_chat([{"role": "user", "content": "q"}])
    assert captured["headers"]["http-referer"] == "https://example.test"
    assert captured["headers"]["x-title"] == "legion-eval"


def test_verifier_model_override(env):
    env.setenv("VSCP_LLM_MODEL", "openai/gpt-4o-mini")
    env.setenv("VSCP_VERIFIER_MODEL", "anthropic/claude-3-haiku")
    assert llm_client.resolve_model("worker") == "openai/gpt-4o-mini"
    assert llm_client.resolve_model("verifier") == "anthropic/claude-3-haiku"


def test_single_retry_on_transport_error_then_success(env):
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("boom")
        return _ok_response("recovered")

    env.setattr(urllib.request, "urlopen", flaky_urlopen)
    text, _ = llm_client.openai_chat([{"role": "user", "content": "q"}])
    assert text == "recovered"
    assert calls["n"] == 2


def test_transport_failure_after_retry_never_leaks_key(env):
    calls = {"n": 0}

    def dead_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("sk-test-secret-key should not surface")

    env.setattr(urllib.request, "urlopen", dead_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        llm_client.openai_chat([{"role": "user", "content": "q"}])
    assert calls["n"] == 2  # exactly one retry
    assert "sk-test-secret-key" not in str(excinfo.value)


def test_contract_failure_is_not_retried(env):
    calls = {"n": 0}

    def bad_body_urlopen(request, timeout=None):
        calls["n"] += 1
        return FakeResponse(b"not json at all")

    env.setattr(urllib.request, "urlopen", bad_body_urlopen)
    with pytest.raises(ValueError):
        llm_client.openai_chat([{"role": "user", "content": "q"}])
    assert calls["n"] == 1  # never retry a bad output


def test_key_resolution_priority(env):
    env.setenv("OPENAI_API_KEY", "openai-key")
    assert llm_client.resolve_api_key() == "sk-test-secret-key"  # openrouter wins
    env.delenv("OPENROUTER_API_KEY")
    env.setenv("VSCP_LLM_API_KEY", "vscp-key")
    assert llm_client.resolve_api_key() == "vscp-key"
    env.delenv("VSCP_LLM_API_KEY")
    assert llm_client.resolve_api_key() == "openai-key"
