"""One real OpenRouter round-trip. Skipped unless VSCP_LLM=1 and a key is set
(mirrors the existing LLM-smoke-skip convention). Costs real money when run."""
from __future__ import annotations

import os

import pytest

from legion.llm_client import openai_chat, real_path_enabled

pytestmark = pytest.mark.skipif(
    not (os.environ.get("VSCP_LLM") == "1" and real_path_enabled()),
    reason="VSCP_LLM unset or no API key",
)


def test_live_round_trip_smallest_prompt():
    text, usage = openai_chat(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=8,
    )
    assert text.strip()
    assert usage is not None and usage.get("prompt_tokens", 0) > 0
