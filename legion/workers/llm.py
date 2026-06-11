from __future__ import annotations

from legion.admission import LLMVerifier


class LLMSolver:
    """Optional OpenAI-compatible worker hook.

    The PoC keeps the LLM path behind VSCP_LLM and uses the same verifier
    interface as deterministic admission. Actual task orchestration stays in
    the local coordinator and SQLite ledger.
    """

    def __init__(self, verifier: LLMVerifier | None = None) -> None:
        self.verifier = verifier or LLMVerifier()

    def propose_body(self, prompt: str) -> str:
        return self.verifier.complete(prompt).strip()
