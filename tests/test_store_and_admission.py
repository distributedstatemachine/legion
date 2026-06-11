from __future__ import annotations

import sqlite3

import pytest

from legion import claims, crypto, tasks
from legion.admission import AdmissionGate, MockVerifier
from legion.store import Store


def _identity(store: Store, seed: str, balance: int = 500_000) -> crypto.Keypair:
    keypair = crypto.keypair_from_seed(seed)
    store.create_identity(keypair.pubkey, balance)
    return keypair


def _task(store: Store, tmp_seed: int = 7, K: int = 2) -> str:
    sponsor = _identity(store, f"sponsor-{tmp_seed}", 3_000_000)
    return tasks.create_fact_chain_task(store, K=K, D=3, seed=tmp_seed, sponsor_pubkey=sponsor.pubkey)


def _fact_claim(store: Store, task_id: str, keypair: crypto.Keypair, index: int = 0, cites=None):
    spec = store.task_spec(task_id)
    fact = spec["docs"][index]["fact"]
    doc_hash = spec["docs"][index]["doc_hash"]
    return claims.build_claim(
        private_key=keypair.private_key,
        author=keypair.pubkey,
        task_id=task_id,
        kind="FACT",
        body=fact,
        evidence=[claims.evidence_ref(doc_hash, fact)],
        cites=cites or [],
    )


def test_append_only_triggers_and_balance_guard(tmp_path):
    store = Store(tmp_path)
    task_id = _task(store)
    worker = _identity(store, "worker", 100_000)
    claim = _fact_claim(store, task_id, worker)
    store.submit_claim(claim)
    doc_hash = store.task_spec(task_id)["docs"][0]["doc_hash"]
    store.fetch(worker.pubkey, doc_hash)

    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE claims SET body = 'mutated'")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM claims")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute('UPDATE transfers SET "amount_µ" = 1')
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM transfers")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE fetches SET object_id = 'x'")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM fetches")

    with pytest.raises(ValueError):
        store.add_transfer(
            from_pubkey=worker.pubkey,
            to_pubkey=None,
            amount=store.balance(worker.pubkey) + 1,
            reason="BURN",
        )
    assert all(balance >= 0 for balance in store.balances().values())


def test_ref_tag_soundness_rejects_200_mutations(tmp_path):
    store = Store(tmp_path)
    task_id = _task(store, tmp_seed=11, K=2)
    author = _identity(store, "ref-fuzzer", 3_000_000)
    answer_key = store.answer_key(task_id)
    gate = AdmissionGate(store, MockVerifier(answer_key))
    spec = store.task_spec(task_id)
    fact = spec["docs"][0]["fact"]
    doc0 = spec["docs"][0]["doc_hash"]
    doc1 = spec["docs"][1]["doc_hash"]
    good = claims.ref_for_sentence(fact)["ref"]

    rejected = 0
    for i in range(200):
        if i % 4 == 0:
            ref = {"head": good["tail"], "tail": good["head"]}
            doc_hash = doc0
        elif i % 4 == 1:
            ref = {"head": f"{good['head']} missing{i}", "tail": good["tail"]}
            doc_hash = doc0
        elif i % 4 == 2:
            ref = {"head": good["head"], "tail": f"{good['tail']} missing{i}"}
            doc_hash = doc0
        else:
            ref = good
            doc_hash = doc1
        claim = claims.build_claim(
            private_key=author.private_key,
            author=author.pubkey,
            task_id=task_id,
            kind="FACT",
            body=f"{fact}\nmutation {i}",
            evidence=[{"doc_hash": doc_hash, "ref": ref}],
        )
        store.submit_claim(claim)
        gate.process_pending()
        stored = store.claim(claim["claim_id"])
        assert stored["status"] == "REJECTED"
        assert stored["reject_reason"] == "bad_ref"
        rejected += 1

    assert rejected == 200


def test_fetch_gating_for_citations(tmp_path):
    store = Store(tmp_path)
    task_id = _task(store, tmp_seed=13, K=2)
    answer_key = store.answer_key(task_id)
    gate = AdmissionGate(store, MockVerifier(answer_key))
    author_a = _identity(store, "author-a", 200_000)
    author_b = _identity(store, "author-b", 200_000)
    author_c = _identity(store, "author-c", 200_000)

    first = _fact_claim(store, task_id, author_a, 0)
    store.submit_claim(first)
    gate.process_pending()
    assert store.claim(first["claim_id"])["status"] == "ADMITTED"

    unfetched = _fact_claim(store, task_id, author_b, 1, cites=[first["claim_id"]])
    store.submit_claim(unfetched)
    gate.process_pending()
    assert store.claim(unfetched["claim_id"])["reject_reason"] == "unfetched_cite"

    store.fetch(author_c.pubkey, first["claim_id"])
    fetched = _fact_claim(store, task_id, author_c, 1, cites=[first["claim_id"]])
    store.submit_claim(fetched)
    gate.process_pending()
    assert store.claim(fetched["claim_id"])["status"] == "ADMITTED"
