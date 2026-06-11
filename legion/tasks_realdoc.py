"""Real multi-document QA task family.

Documents are bundled plain-text files under `corpus/` (no network at task
creation time). `gold_facts` (verbatim sentences, one per document) are stored
as the answer key and used **only** by evaluation - they are never passed to
the LLM verifier or to workers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legion import crypto

BOUNTY = 1_000_000


def load_fixture(path: str | Path) -> dict[str, Any]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(fixture["documents"]) != len(fixture["gold_facts"]):
        raise ValueError(f"fixture {fixture.get('name')}: one gold fact per document required")
    return fixture


def make_realdoc_task(
    store,
    corpus_dir: str | Path,
    question: str,
    gold_facts: list[str],
    documents: list[str],
    sponsor_pubkey: str,
    K: int | None = None,
    bounty: int = BOUNTY,
) -> str:
    corpus_dir = Path(corpus_dir)
    if K is None:
        K = len(documents)
    if K != len(documents) or K != len(gold_facts):
        raise ValueError("K must equal the number of documents and gold facts")
    doc_records: list[dict[str, Any]] = []
    for index, name in enumerate(documents):
        text = (corpus_dir / name).read_text(encoding="utf-8")
        if gold_facts[index] not in text:
            raise ValueError(f"gold fact for {name} is not verbatim in the document")
        doc_hash = store.put_evidence(text, name=name)
        doc_records.append({"index": index, "name": name, "doc_hash": doc_hash})
    spec = {
        "family": "realdoc",
        "question": question,
        "K": K,
        "docs": doc_records,
    }
    spec_hash = crypto.sha256_bytes(crypto.canonical_bytes(spec))
    task_id = f"task-{spec_hash[:16]}"
    answer_key = {
        "facts": gold_facts,
        "doc_hash_by_fact": {
            gold_facts[index]: doc_records[index]["doc_hash"] for index in range(K)
        },
    }
    store.create_task(
        task_id=task_id,
        spec_hash=spec_hash,
        bounty=bounty,
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
