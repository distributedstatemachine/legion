"""LLM-backed worker for the realdoc task family.

Works against any OpenAI-compatible endpoint via the existing env vars (the
`complete` callable is injected; CI uses a deterministic stub). Sentences are
checked verbatim against the fetched document *before* the admission fee is
paid; non-verbatim model output is retried at most twice and then abandoned.
FAILs here are *verified irrelevance statements*, not worker errors - expected
to be rare, the path exists for completeness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from legion import claims, tasks
from legion.admission import LLMVerifier
from legion.crypto import keypair_from_seed
from legion.workers.base import Worker

MAX_EXTRACTION_ATTEMPTS = 3  # initial attempt + at most two resubmissions
ANSWER_BODY_LIMIT = 600


def default_max_calls() -> int:
    return int(os.environ.get("VSCP_MAX_LLM_CALLS_PER_WORKER", "30"))


@dataclass
class LLMWorker(Worker):
    question: str
    complete: Callable[[str], str]
    max_calls: int = field(default_factory=default_max_calls)
    calls: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    fetched_docs: dict[str, str] = field(default_factory=dict)
    fetched_facts: dict[str, str] = field(default_factory=dict)
    seen_fail_ids: set[str] = field(default_factory=set)
    triaged: bool = False
    answered: bool = False

    @classmethod
    def create(
        cls, name: str, question: str, complete: Callable[[str], str], **kwargs
    ) -> "LLMWorker":
        keypair = keypair_from_seed(f"llm:{name}")
        return cls(name=name, keypair=keypair, role="llm", question=question, complete=complete, **kwargs)

    def _call(self, prompt: str) -> str | None:
        if self.calls >= self.max_calls:
            return None  # budget exhausted: stop cleanly
        self.calls += 1
        try:
            return self.complete(prompt)
        except Exception:
            return None

    def step(self, store, task_id: str) -> None:
        if store.task_row(task_id)["status"] != "OPEN":
            return
        if self.calls >= self.max_calls:
            return
        self._absorb_fails(store, task_id)
        lease = tasks.live_lease(store, task_id, self.pubkey) or tasks.lease_available_subtask(
            store, task_id, self.pubkey
        )
        if lease is None:
            self._triage_idle(store, task_id)
            return
        kind = tasks.subtask_kind(lease["subtask_id"])
        if kind == "fact":
            self._work_fact(store, task_id, lease["subtask_id"])
        elif kind == "answer":
            self._work_answer(store, task_id, lease["subtask_id"])

    def _absorb_fails(self, store, task_id: str) -> None:
        """Read published FAILs through the metered fetch (steering readership)."""
        for claim in store.admitted_claims(task_id):
            if claim["kind"] != "FAIL" or claim["claim_id"] in self.seen_fail_ids:
                continue
            self.seen_fail_ids.add(claim["claim_id"])
            if claim["author"] != self.pubkey:
                store.fetch(self.pubkey, claim["claim_id"])

    def _triage_idle(self, store, task_id: str) -> None:
        """Idle workers triage a document others are mining: publish one
        verified irrelevance FAIL ('looks relevant, is not') so leaseholders
        can skip the trap sentence. One triage per worker per task."""
        if self.triaged or self.answered:
            return
        spec = store.task_spec(task_id)
        doc = spec["docs"][int(self.pubkey[:8], 16) % len(spec["docs"])]
        doc_hash = doc["doc_hash"]
        if doc_hash not in self.fetched_docs:
            self.fetched_docs[doc_hash] = store.fetch(self.pubkey, doc_hash)
        text = self.fetched_docs[doc_hash]
        prompt = (
            "TASK: TRIAGE\n"
            "From the DOCUMENT below, quote one sentence that superficially "
            "appears relevant to the QUESTION but is actually irrelevant. Copy "
            "it verbatim, character for character.\n"
            'Return strict JSON only: {"sentence": "..."}\n'
            f"QUESTION: {self.question}\n"
            f"DOCUMENT:\n{text}"
        )
        raw = self._call(prompt)
        self.triaged = True
        if raw is None:
            return
        parsed = LLMVerifier._parse_first_json(raw)
        sentence = parsed.get("sentence") if isinstance(parsed, dict) else None
        if isinstance(sentence, str):
            self.submit_irrelevance_fail(store, task_id, doc_hash, sentence)

    def _work_fact(self, store, task_id: str, subtask_id: str) -> None:
        if self.attempts.get(subtask_id, 0) >= MAX_EXTRACTION_ATTEMPTS:
            return
        spec = store.task_spec(task_id)
        doc = spec["docs"][tasks.subtask_index(subtask_id)]
        doc_hash = doc["doc_hash"]
        if doc_hash not in self.fetched_docs:
            # Read epoch: fetch the document now, extract next epoch. The gap
            # lets FAILs published by triage land before the FACT is authored,
            # which is what makes their readership count for steering v2.
            self.fetched_docs[doc_hash] = store.fetch(self.pubkey, doc_hash)
            return
        text = self.fetched_docs[doc_hash]
        prompt = (
            "TASK: EXTRACT\n"
            "From the DOCUMENT below, extract the single sentence most relevant "
            "to the QUESTION. Copy it verbatim, character for character.\n"
            'Return strict JSON only: {"sentence": "..."}\n'
            f"QUESTION: {self.question}\n"
            f"DOCUMENT:\n{text}"
        )
        raw = self._call(prompt)
        if raw is None:
            return
        parsed = LLMVerifier._parse_first_json(raw)
        sentence = parsed.get("sentence") if isinstance(parsed, dict) else None
        # Local verbatim check BEFORE paying the admission fee.
        if (
            not isinstance(sentence, str)
            or sentence not in text
            or len(sentence.split()) < 10
        ):
            self.attempts[subtask_id] = self.attempts.get(subtask_id, 0) + 1
            return
        claim = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            subtask_id=subtask_id,
            kind="FACT",
            body=sentence,
            evidence=[claims.evidence_ref(doc_hash, sentence)],
        )
        store.submit_claim(claim)
        self.attempts[subtask_id] = MAX_EXTRACTION_ATTEMPTS

    def submit_irrelevance_fail(self, store, task_id: str, doc_hash: str, sentence: str) -> None:
        """A FAIL in this family is a sentence asserted irrelevant despite
        appearing relevant; it is admitted only if the verifier supports the
        statement against the span. Expected to be rare."""
        if doc_hash not in self.fetched_docs:
            self.fetched_docs[doc_hash] = store.fetch(self.pubkey, doc_hash)
        if sentence not in self.fetched_docs[doc_hash] or len(sentence.split()) < 10:
            return
        claim = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            kind="FAIL",
            body=sentence,
            evidence=[claims.evidence_ref(doc_hash, sentence)],
        )
        store.submit_claim(claim)

    def _work_answer(self, store, task_id: str, subtask_id: str) -> None:
        if self.answered:
            return
        spec = store.task_spec(task_id)
        fact_claims: list[dict[str, Any]] = []
        for index in range(spec["K"]):
            admitted = store.claims_for_subtask(f"{task_id}:fact:{index}")
            if not admitted:
                return
            fact_claims.append(admitted[0])
        bodies = []
        for claim in fact_claims:
            claim_id = claim["claim_id"]
            if claim_id not in self.fetched_facts:
                self.fetched_facts[claim_id] = store.fetch(self.pubkey, claim_id)
            bodies.append(self.fetched_facts[claim_id])
        prompt = (
            "TASK: SYNTHESIZE\n"
            "Compose a one-paragraph answer to the QUESTION strictly from the "
            "FACTS below. Plain text, no preamble.\n"
            f"QUESTION: {self.question}\n"
            "FACTS:\n" + "\n".join(f"- {body}" for body in bodies)
        )
        raw = self._call(prompt)
        if raw is None:
            return
        body = "ANSWER: " + " ".join(raw.split())
        if len(body) > ANSWER_BODY_LIMIT:
            body = body[:ANSWER_BODY_LIMIT]
        answer = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            subtask_id=subtask_id,
            kind="ANSWER",
            body=body,
            cites=[claim["claim_id"] for claim in fact_claims],
        )
        store.submit_claim(answer)
        self.answered = True


def openai_complete(prompt: str) -> str:
    """Real-endpoint completion reusing the verifier's transport."""
    return LLMVerifier()._openai_compatible_complete(prompt)
