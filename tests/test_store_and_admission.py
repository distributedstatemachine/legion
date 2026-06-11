from __future__ import annotations

import sqlite3

import pytest

from legion import claims, crypto, tasks
from legion.admission import AdmissionGate, MockVerifier, resolve_ref_span
from legion.cli import run_demo
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


def test_resolve_ref_span_returns_minimal_span():
    # An author picking a repeated head and a distant tail must not capture a
    # wide span containing sentences they never localized.
    document = (
        "alpha beta gamma early words here and lots of intervening filler text. "
        "unrelated sentence the author never localized sits in the middle here. "
        "alpha beta gamma the actual short span tail end"
    )
    ref = {"head": "alpha beta gamma", "tail": "tail end"}
    span = resolve_ref_span(document, ref)
    assert span == "alpha beta gamma the actual short span tail end"
    assert "never localized" not in span


def test_fetch_serves_admitted_claims_only(tmp_path):
    store = Store(tmp_path)
    task_id = _task(store, tmp_seed=17, K=2)
    gate = AdmissionGate(store, MockVerifier(store.answer_key(task_id)))
    author = _identity(store, "fetch-author", 200_000)
    reader = _identity(store, "fetch-reader", 0)

    pending = _fact_claim(store, task_id, author, 0)
    store.submit_claim(pending)
    with pytest.raises(KeyError):
        store.fetch(reader.pubkey, pending["claim_id"])

    gate.process_pending()
    assert store.fetch(reader.pubkey, pending["claim_id"]) == pending["body"]

    spec = store.task_spec(task_id)
    rejected = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FACT",
        body="secret pending body that must not leak before admission ever",
        evidence=[{"doc_hash": spec["docs"][0]["doc_hash"], "ref": {"head": "x", "tail": "y"}}],
    )
    store.submit_claim(rejected)
    gate.process_pending()
    assert store.claim(rejected["claim_id"])["status"] == "REJECTED"
    with pytest.raises(KeyError):
        store.fetch(reader.pubkey, rejected["claim_id"])


def test_lease_bond_refunded_when_task_closes_under_holder(tmp_path):
    # An honest worker holding the answer lease when a racing (unleased) answer
    # closes the task must get its bond back in the same transaction.
    store = Store(tmp_path)
    task_id = _task(store, tmp_seed=23, K=1)
    answer_key = store.answer_key(task_id)
    gate = AdmissionGate(store, MockVerifier(answer_key))
    honest = _identity(store, "lease-honest", 200_000)
    hoarder = _identity(store, "lease-hoarder", 200_000)

    fact = _fact_claim(store, task_id, honest, 0)
    fact["subtask_id"] = None  # plain FACT; the subtask DAG is exercised below
    fact = claims.build_claim(
        private_key=honest.private_key,
        author=honest.pubkey,
        task_id=task_id,
        kind="FACT",
        body=fact["body"],
        evidence=fact["evidence"],
    )
    store.submit_claim(fact)
    gate.process_pending()
    assert store.claim(fact["claim_id"])["status"] == "ADMITTED"

    # Mark the fact subtask DONE so the answer subtask becomes leasable.
    store.conn.execute(
        "UPDATE subtasks SET status = 'DONE' WHERE subtask_id = ?", (f"{task_id}:fact:0",)
    )
    store.conn.commit()
    lease = tasks.lease_available_subtask(store, task_id, honest.pubkey)
    assert lease["subtask_id"] == f"{task_id}:answer"
    balance_after_lease = store.balance(honest.pubkey)

    store.fetch(hoarder.pubkey, fact["claim_id"])
    racing = claims.build_claim(
        private_key=hoarder.private_key,
        author=hoarder.pubkey,
        task_id=task_id,
        kind="ANSWER",
        body="ANSWER: all 1 facts established via cited FACT claims",
        cites=[fact["claim_id"]],
    )
    store.submit_claim(racing)
    gate.process_pending()
    assert store.task_row(task_id)["status"] == "CLOSED"

    # Bond came back and the lease was cleared.
    assert store.balance(honest.pubkey) == balance_after_lease + 5_000
    sub = store.subtask(f"{task_id}:answer")
    assert sub["status"] == "PENDING" and sub["lease_holder"] is None
    # Total BOND inflow == outflow for the run; no bond stranded.
    bonds = store.pseudo_balances()
    assert all(balance == 0 for account, balance in bonds.items() if account.startswith("BOND:"))


def test_pseudo_account_ledger_and_global_conservation(tmp_path):
    run_demo(workers=6, honest=4, hoarders=2, K=6, D=3, seed=42, root=tmp_path)
    store = Store(tmp_path)
    minted = 2_000_000 + 6 * 200_000  # sponsor + worker grants in run_demo
    burned = sum(
        row["amount_µ"] for row in store.transfer_rows() if row["to_pubkey"] is None
    )
    identity_total = sum(store.balances().values())
    pseudo = store.pseudo_balances()
    assert all(balance >= 0 for balance in pseudo.values())
    # Money is only created by the demo's identity grants; everything else is
    # conserved across identities, pseudo accounts (ESCROW/FEE_POOL/BOND), and burns.
    assert identity_total + sum(pseudo.values()) + burned == minted
    # The task escrow is exactly drained by settlement.
    escrow = {k: v for k, v in pseudo.items() if k.startswith("ESCROW:")}
    assert all(balance == 0 for balance in escrow.values())
    store.close()
