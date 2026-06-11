from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from legion import claims, tasks
from legion.crypto import Keypair, keypair_from_seed
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

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)

    @classmethod
    def create(cls, name: str, role: str, oracle: FactOracle, rng_seed: int) -> "ScriptedWorker":
        keypair = keypair_from_seed(f"scripted:{name}:{role}:{rng_seed}")
        return cls(name=name, keypair=keypair, role=role, oracle=oracle, rng_seed=rng_seed)

    def step(self, store, task_id: str) -> None:
        if self.role == "hoarder":
            self._hoard(store, task_id)
            return
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

    def _hoard(self, store, task_id: str) -> None:
        for claim in store.admitted_claims(task_id):
            if not store.has_fetched(self.pubkey, claim["claim_id"]):
                store.fetch(self.pubkey, claim["claim_id"])
                return

    def _work_fact(self, store, task_id: str, subtask_id: str) -> None:
        spec = store.task_spec(task_id)
        index = tasks.subtask_index(subtask_id)
        doc = spec["docs"][index]
        if not store.has_fetched(self.pubkey, doc["doc_hash"]):
            store.fetch(self.pubkey, doc["doc_hash"])
        remaining = self.attempts.setdefault(subtask_id, list(doc["candidates"]))
        if not remaining:
            return
        candidate = remaining.pop(self.rng.randrange(len(remaining)))
        if not self.oracle.is_fact(candidate):
            return
        claim = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            subtask_id=subtask_id,
            kind="FACT",
            body=candidate,
            evidence=[claims.evidence_ref(doc["doc_hash"], candidate)],
        )
        store.submit_claim(claim)

    def _work_answer(self, store, task_id: str, subtask_id: str) -> None:
        spec = store.task_spec(task_id)
        fact_claims: list[dict[str, Any]] = []
        for index in range(spec["K"]):
            admitted = store.claims_for_subtask(f"{task_id}:fact:{index}")
            if not admitted:
                return
            fact_claims.append(admitted[0])
        for claim in fact_claims:
            if not store.has_fetched(self.pubkey, claim["claim_id"]):
                store.fetch(self.pubkey, claim["claim_id"])
        evidence = []
        body_lines = []
        for claim in fact_claims:
            body_lines.append(store.fetch(self.pubkey, claim["claim_id"]))
            evidence.extend(claim["evidence"])
        answer = claims.build_claim(
            private_key=self.keypair.private_key,
            author=self.pubkey,
            task_id=task_id,
            subtask_id=subtask_id,
            kind="ANSWER",
            body="\n".join(body_lines),
            evidence=evidence,
            cites=[claim["claim_id"] for claim in fact_claims],
        )
        store.submit_claim(answer)
