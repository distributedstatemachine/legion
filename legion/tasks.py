from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from legion import crypto
from legion.admission_constants import LEASE_BOND


BOUNTY = 1_000_000
WORD_POOL = (
    "amber basin cedar delta ember fennel glacier harbor iris juniper kelp linen "
    "meadow nickel orchid pebble quartz river saffron timber umber violet willow xenon "
    "yucca zephyr atlas briar cobalt dune elm flint grove hazel ivory jasper kiln lagoon"
).split()


@dataclass(frozen=True)
class SyntheticDocument:
    name: str
    text: str
    fact: str
    decoys: list[str]
    candidates: list[str]


@dataclass(frozen=True)
class SyntheticTask:
    K: int
    D: int
    seed: int
    documents: list[SyntheticDocument]

    @property
    def facts(self) -> list[str]:
        return [doc.fact for doc in self.documents]


def _sentence(rng: random.Random, tag: str, words: int = 12) -> str:
    digits = "".join(ch for ch in tag if ch.isdigit()) or "0"
    prefix = ("d" if tag.startswith("decoy") else "f") + digits
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    tokens = [f"{prefix}w{i}{rng.choice(alphabet)}" for i in range(words)]
    return " ".join(tokens) + "."


def _filler_sentence(rng: random.Random, doc_index: int, sent_index: int) -> str:
    tokens = [
        f"l{doc_index}_{sent_index}_{i}_{rng.choice(WORD_POOL)}" for i in range(12)
    ]
    return " ".join(tokens) + "."


def make_task(K: int, D: int, seed: int) -> SyntheticTask:
    rng = random.Random(seed)
    documents: list[SyntheticDocument] = []
    for i in range(K):
        fact = _sentence(rng, f"fact{i}")
        decoys = [_sentence(rng, f"decoy{i}_{j}") for j in range(D)]
        sentences = [_filler_sentence(rng, i, j) for j in range(50)]
        inserts = [fact, *decoys]
        for sentence in inserts:
            sentences.insert(rng.randrange(0, len(sentences) + 1), sentence)
        candidates = inserts[:]
        rng.shuffle(candidates)
        documents.append(
            SyntheticDocument(
                name=f"doc_{i}.txt",
                text="\n".join(sentences),
                fact=fact,
                decoys=decoys,
                candidates=candidates,
            )
        )
    return SyntheticTask(K=K, D=D, seed=seed, documents=documents)


def create_fact_chain_task(store, K: int, D: int, seed: int, sponsor_pubkey: str) -> str:
    synthetic = make_task(K, D, seed)
    doc_records: list[dict[str, Any]] = []
    decoys: dict[str, list[str]] = {}
    for index, doc in enumerate(synthetic.documents):
        doc_hash = store.put_evidence(doc.text, name=doc.name)
        doc_records.append(
            {
                "index": index,
                "name": doc.name,
                "doc_hash": doc_hash,
                "fact": doc.fact,
                "candidates": doc.candidates,
            }
        )
        decoys[str(index)] = doc.decoys
    spec = {
        "family": "fact-chain",
        "K": K,
        "D": D,
        "seed": seed,
        "docs": doc_records,
    }
    spec_hash = crypto.sha256_bytes(crypto.canonical_bytes(spec))
    task_id = f"task-{spec_hash[:16]}"
    answer_key = {
        "facts": synthetic.facts,
        "decoys": decoys,
        "doc_hash_by_fact": {
            doc.fact: doc_records[index]["doc_hash"]
            for index, doc in enumerate(synthetic.documents)
        },
    }
    store.create_task(
        task_id=task_id,
        spec_hash=spec_hash,
        bounty=BOUNTY,
        spec=spec,
        answer_key=answer_key,
        sponsor_pubkey=sponsor_pubkey,
    )
    fact_subtasks = []
    for index in range(K):
        subtask_id = f"{task_id}:fact:{index}"
        fact_subtasks.append(subtask_id)
        store.create_subtask(subtask_id, task_id, [])
    store.create_subtask(f"{task_id}:answer", task_id, fact_subtasks)
    for record in doc_records:
        store.conn.execute(
            "UPDATE evidence_docs SET task_id = ? WHERE doc_hash = ?",
            (task_id, record["doc_hash"]),
        )
    store.conn.commit()
    return task_id


def _deps_done(store, deps_json: str) -> bool:
    import json

    deps = json.loads(deps_json)
    for dep in deps:
        if store.subtask(dep)["status"] != "DONE":
            return False
    return True


def expire_leases(store) -> None:
    rows = store.conn.execute(
        "SELECT * FROM subtasks WHERE status = 'LEASED' AND lease_expiry_epoch <= ?",
        (store.epoch(),),
    ).fetchall()
    for row in rows:
        with store.conn:
            store.conn.execute(
                "UPDATE subtasks SET status = 'PENDING', lease_holder = NULL, lease_expiry_epoch = NULL "
                "WHERE subtask_id = ?",
                (row["subtask_id"],),
            )
            store._apply_transfer(
                store.epoch(), f"BOND:{row['subtask_id']}", None, LEASE_BOND, "BURN"
            )


def live_lease(store, task_id: str, worker_pubkey: str):
    row = store.conn.execute(
        "SELECT * FROM subtasks WHERE task_id = ? AND lease_holder = ? AND status = 'LEASED' "
        "ORDER BY subtask_id LIMIT 1",
        (task_id, worker_pubkey),
    ).fetchone()
    return row


def lease_available_subtask(store, task_id: str, worker_pubkey: str):
    expire_leases(store)
    existing = live_lease(store, task_id, worker_pubkey)
    if existing is not None:
        return existing
    rows = store.conn.execute(
        "SELECT * FROM subtasks WHERE task_id = ? AND status = 'PENDING' ORDER BY subtask_id",
        (task_id,),
    ).fetchall()
    for row in rows:
        if not _deps_done(store, row["deps_json"]):
            continue
        with store.conn:
            store._apply_transfer(
                store.epoch(), worker_pubkey, f"BOND:{row['subtask_id']}", LEASE_BOND, "BOND"
            )
            store.conn.execute(
                "UPDATE subtasks SET status = 'LEASED', lease_holder = ?, lease_expiry_epoch = ? "
                "WHERE subtask_id = ?",
                (worker_pubkey, store.epoch() + 10, row["subtask_id"]),
            )
        return store.subtask(row["subtask_id"])
    return None


def subtask_kind(subtask_id: str) -> str:
    if subtask_id.endswith(":answer"):
        return "answer"
    if ":fact:" in subtask_id:
        return "fact"
    return "unknown"


def subtask_index(subtask_id: str) -> int:
    return int(subtask_id.rsplit(":", 1)[1])
