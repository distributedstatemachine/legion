from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from legion import claims as claim_helpers
from legion import crypto
from legion.admission_constants import CHALLENGE_WINDOW


class Verifier(Protocol):
    def supports(self, claim: dict[str, Any], spans: list[str]) -> bool:
        ...

    def solves(self, task_spec: dict[str, Any], claim: dict[str, Any], spans: list[str]) -> bool:
        ...


@dataclass
class MockVerifier:
    answer_key: dict[str, Any] | None = None

    def _facts(self) -> set[str]:
        if not self.answer_key:
            return set()
        return set(self.answer_key.get("facts", []))

    def _decoys(self) -> set[str]:
        if not self.answer_key:
            return set()
        decoys: set[str] = set()
        for doc_decoys in self.answer_key.get("decoys", {}).values():
            decoys.update(doc_decoys)
        return decoys

    def supports(self, claim: dict[str, Any], spans: list[str]) -> bool:
        body = claim["body"].strip()
        kind = claim["kind"]
        joined_spans = "\n".join(spans)
        facts = self._facts()
        decoys = self._decoys()
        if kind == "ANSWER":
            # Coverage-based: every key fact must appear in the support spans
            # (cited fact bodies and/or evidence spans). The body itself is a
            # compact summary so answers stay O(1) in size regardless of K.
            return bool(body) and bool(facts) and all(fact in joined_spans for fact in facts)
        if kind == "FACT":
            return body in facts and body in joined_spans and body not in decoys
        if kind == "FAIL":
            # A FAIL is a verified negative result: the rejected candidate must
            # actually exist in the cited spans and be a decoy per the answer key.
            return body in decoys and body in joined_spans
        if kind == "CONSTRAINT":
            return bool(body) and body in joined_spans
        return False

    def solves(self, task_spec: dict[str, Any], claim: dict[str, Any], spans: list[str]) -> bool:
        return claim["kind"] == "ANSWER" and self.supports(claim, spans)


class LLMVerifier:
    def __init__(self, complete=None) -> None:
        self.complete = complete or self._openai_compatible_complete

    def _openai_compatible_complete(self, prompt: str) -> str:
        url = os.environ.get("VSCP_LLM_URL")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VSCP_LLM_API_KEY")
        if not url or not api_key:
            raise RuntimeError("VSCP_LLM_URL and API key are required for the LLM verifier")
        payload = crypto.canonical_bytes(
            {
                "model": os.environ.get("VSCP_LLM_MODEL", "gpt-4.1-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        )
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM verifier request failed: {exc}") from exc
        return data

    def supports(self, claim: dict[str, Any], spans: list[str]) -> bool:
        prompt = (
            "Is every assertion in BODY supported by SPANS? Answer YES or NO only.\n"
            f"BODY:\n{claim['body']}\n\nSPANS:\n" + "\n---\n".join(spans)
        )
        return self.complete(prompt).strip().upper().startswith("YES")

    def solves(self, task_spec: dict[str, Any], claim: dict[str, Any], spans: list[str]) -> bool:
        return claim["kind"] == "ANSWER" and self.supports(claim, spans)


def default_verifier(answer_key: dict[str, Any] | None = None) -> Verifier:
    if os.environ.get("VSCP_LLM") == "1":
        return LLMVerifier()
    return MockVerifier(answer_key)


def _find_all(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos == -1:
            return positions
        positions.append(pos)
        start = pos + 1


def resolve_ref_span(document: str, ref: dict[str, str]) -> str | None:
    """Resolve a head/tail ref to the *minimal* valid span.

    Returning the smallest head..tail window (deterministic tie-break on the
    earliest head) prevents an author from picking a common head and a distant
    tail to capture a wide span containing text they never localized.
    """
    head = ref.get("head", "")
    tail = ref.get("tail", "")
    if not head or not tail:
        return None
    tail_starts = _find_all(document, tail)
    best: tuple[int, int] | None = None  # (span_length, head_start)
    for head_start in _find_all(document, head):
        head_end = head_start + len(head)
        for tail_start in tail_starts:
            if tail_start <= head_end:
                continue
            length = tail_start + len(tail) - head_start
            if length <= 1200 and (best is None or (length, head_start) < best):
                best = (length, head_start)
            break  # later tails only widen the span for this head
    if best is None:
        return None
    length, head_start = best
    return document[head_start : head_start + length]


class AdmissionGate:
    def __init__(self, store, verifier: Verifier, challenge_window: int = CHALLENGE_WINDOW) -> None:
        self.store = store
        self.verifier = verifier
        self.challenge_window = challenge_window

    def process_pending(self) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for claim in self.store.pending_claims():
            ok, reason, spans = self._validate(claim)
            if ok and self.verifier.supports(claim, spans):
                self.store.admit_claim(claim["claim_id"])
                results.append((claim["claim_id"], "ADMITTED"))
                if claim["kind"] == "ANSWER" and self.verifier.solves(
                    self.store.task_spec(claim["task_id"]), claim, spans
                ):
                    self.store.close_task_for_answer(
                        claim["task_id"], claim["claim_id"], self.challenge_window
                    )
            else:
                self.store.reject_claim(claim["claim_id"], reason if not ok else "semantic")
                results.append((claim["claim_id"], "REJECTED"))
        return results

    def _validate(self, claim: dict[str, Any]) -> tuple[bool, str, list[str]]:
        if claim["status"] != "PENDING":
            return False, "not_pending", []
        canonical_id = crypto.sha256_bytes(crypto.canonical_claim_bytes(claim))
        if canonical_id != claim["claim_id"]:
            return False, "bad_claim_id", []
        if not crypto.verify(claim["author"], crypto.canonical_claim_bytes(claim), claim["sig"]):
            return False, "bad_signature", []
        if len(claim["body"]) > 600:
            return False, "body_too_long", []
        if len(claim["evidence"]) > 8:
            return False, "too_many_evidence_refs", []
        if len(claim["cites"]) > 16:
            return False, "too_many_cites", []
        if not claim_helpers.validate_derivations_shape(claim["cites"], claim["derivations"]):
            return False, "bad_derivations", []
        if claim.get("subtask_id") is not None:
            try:
                subtask = self.store.subtask(claim["subtask_id"])
            except KeyError:
                return False, "unknown_subtask", []
            if subtask["lease_holder"] != claim["author"]:
                return False, "missing_lease", []
            if subtask["lease_expiry_epoch"] is not None and subtask["lease_expiry_epoch"] < self.store.epoch():
                return False, "expired_lease", []
        for cited_id in claim["cites"]:
            try:
                cited = self.store.claim(cited_id)
            except KeyError:
                return False, "unknown_cite", []
            if cited["status"] != "ADMITTED":
                return False, "unadmitted_cite", []
            if cited["task_id"] != claim["task_id"]:
                return False, "cross_task_cite", []
            if not self.store.has_fetched(claim["author"], cited_id, claim["epoch_submitted"]):
                return False, "unfetched_cite", []
        spans: list[str] = []
        for evidence_ref in claim["evidence"]:
            doc_hash = evidence_ref.get("doc_hash")
            ref = evidence_ref.get("ref", {})
            if not doc_hash or not self.store.evidence_exists(doc_hash):
                return False, "missing_evidence", []
            document = self.store.read_evidence_unmetered(doc_hash)
            span = resolve_ref_span(document, ref)
            if span is None:
                return False, "bad_ref", []
            spans.append(span)
        # Cited admitted claims are provenance: their bodies count as support
        # spans, so claims (notably ANSWERs) need not re-attach upstream
        # evidence that was already verified at the cited claim's admission.
        for cited_id in claim["cites"]:
            spans.append(self.store.claim(cited_id)["body"])
        return True, "ok", spans
