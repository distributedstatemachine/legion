from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from legion import claims, tasks
from legion.crypto import keypair_from_seed
from legion.workers.base import Worker


class FactOracle:
    def __init__(self, answer_key: dict[str, Any]) -> None:
        self.facts = set(answer_key["facts"])

    def is_fact(self, candidate: str) -> bool:
        return candidate in self.facts


@dataclass
class ScriptedWorker(Worker):
    oracle: FactOracle
    rng_seed: int
    attempts: dict[str, list[str]] = field(default_factory=dict)
    seen_fail_ids: set[str] = field(default_factory=set)
    known_fails: set[str] = field(default_factory=set)
    fetched_facts: dict[str, str] = field(default_factory=dict)
    answered: bool = False

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)

    @classmethod
    def create(cls, name: str, role: str, oracle: FactOracle, rng_seed: int) -> "ScriptedWorker":
        keypair = keypair_from_seed(f"scripted:{name}:{role}:{rng_seed}")
        return cls(name=name, keypair=keypair, role=role, oracle=oracle, rng_seed=rng_seed)

    def step(self, store, task_id: str) -> None:
        if store.task_row(task_id)["status"] != "OPEN":
            return
        if self.role == "hoarder":
            self._race(store, task_id)
            return
        self._absorb_fails(store, task_id)
        lease = tasks.live_lease(store, task_id, self.pubkey) or tasks.lease_available_subtask(
            store, task_id, self.pubkey
        )
        if lease is None:
            return
        kind = tasks.subtask_kind(lease["subtask_id"])
        if kind == "fact":
            self._work_fact(store, task_id, lease["subtask_id"])
        elif kind == "answer":
            self._work_answer(store, task_id, lease["subtask_id"])

    def _absorb_fails(self, store, task_id: str) -> None:
        """Read published FAILs (through the metered fetch) to rule out decoys."""
        for claim in store.admitted_claims(task_id):
            if claim["kind"] != "FAIL" or claim["claim_id"] in self.seen_fail_ids:
                continue
            self.seen_fail_ids.add(claim["claim_id"])
            if claim["author"] == self.pubkey:
                continue  # the author already knows its own FAIL; avoid self-read steering weight
            self.known_fails.add(store.fetch(self.pubkey, claim["claim_id"]))

    def _race(self, store, task_id: str) -> None:
        """Racing hoarder: free-ride on published facts and race for the answer.

        Publishes nothing along the way; once every fact is admitted, submits an
        un-leased ANSWER citing the (freely fetched) fact claims and pays only
        the admission fee. This is the strategy whose defeat the settlement
        design claims.
        """
        if self.answered:
            return
        spec = store.task_spec(task_id)
        body_lines: list[str] = []
        cite_ids: list[str] = []
        ready = True
        for index in range(spec["K"]):
            admitted = store.claims_for_subtask(f"{task_id}:fact:{index}")
            if not admitted:
                ready = False
                continue
            claim_id = admitted[0]["claim_id"]
            if claim_id not in self.fetched_facts:
                self.fetched_facts[claim_id] = store.fetch(self.pubkey, claim_id)
            body_lines.append(self.fetched_facts[claim_id])
            cite_ids.append(claim_id)
        if not ready:
            return
        answer = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            kind="ANSWER",
            body=f"ANSWER: all {spec['K']} facts established via cited FACT claims",
            cites=cite_ids,
        )
        store.submit_claim(answer)
        self.answered = True

    def _work_fact(self, store, task_id: str, subtask_id: str) -> None:
        spec = store.task_spec(task_id)
        index = tasks.subtask_index(subtask_id)
        doc = spec["docs"][index]
        if not store.has_fetched(self.pubkey, doc["doc_hash"]):
            store.fetch(self.pubkey, doc["doc_hash"])
        remaining = self.attempts.setdefault(subtask_id, list(doc["candidates"]))
        remaining[:] = [c for c in remaining if c not in self.known_fails]
        if not remaining:
            return
        candidate = remaining.pop(self.rng.randrange(len(remaining)))
        kind = "FACT" if self.oracle.is_fact(candidate) else "FAIL"
        claim = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            # FAILs carry no subtask: only the FACT completes the lease.
            subtask_id=subtask_id if kind == "FACT" else None,
            kind=kind,
            body=candidate,
            evidence=[claims.evidence_ref(doc["doc_hash"], candidate)],
        )
        store.submit_claim(claim)
        if kind == "FAIL":
            self.known_fails.add(candidate)

    def _work_answer(self, store, task_id: str, subtask_id: str) -> None:
        if self.answered:
            return  # never resubmit an identical claim (claim_id is a PK)
        spec = store.task_spec(task_id)
        fact_claims: list[dict[str, Any]] = []
        for index in range(spec["K"]):
            admitted = store.claims_for_subtask(f"{task_id}:fact:{index}")
            if not admitted:
                return
            fact_claims.append(admitted[0])
        for claim in fact_claims:
            claim_id = claim["claim_id"]
            if claim_id not in self.fetched_facts:
                self.fetched_facts[claim_id] = store.fetch(self.pubkey, claim_id)
        # Cites are the provenance; the facts' evidence (and verbatim text) was
        # already verified at their own admission, so the answer is O(1) in
        # both body and evidence and works for any K.
        answer = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            subtask_id=subtask_id,
            kind="ANSWER",
            body=f"ANSWER: all {spec['K']} facts established via cited FACT claims",
            cites=[claim["claim_id"] for claim in fact_claims],
        )
        store.submit_claim(answer)
        self.answered = True
