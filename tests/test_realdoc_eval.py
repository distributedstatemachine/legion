"""Realdoc task family + LLM workers + eval, on the fake-LLM CI path."""
from __future__ import annotations

import json
from pathlib import Path

from legion import crypto
from legion.admission import LLMVerifier
from legion.coordinator import Coordinator
from legion.evaluate import CountingComplete, make_fake_complete, run_eval
from legion.store import Store
from legion.tasks_realdoc import load_fixture, make_realdoc_task
from legion.workers.llm import LLMWorker

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASKS = CORPUS / "tasks"


def test_fixtures_are_well_formed():
    fixtures = sorted(TASKS.glob("*.json"))
    assert len(fixtures) >= 3
    for path in fixtures:
        fixture = load_fixture(path)
        for name, fact in zip(fixture["documents"], fixture["gold_facts"]):
            text = (CORPUS / name).read_text(encoding="utf-8")
            assert fact in text, f"{path.name}: gold fact not verbatim in {name}"
            assert len(fact.split()) >= 10


def test_eval_fake_llm_end_to_end(tmp_path):
    out = tmp_path / "report.json"
    report = run_eval(TASKS, corpus_dir=CORPUS, n_workers=4, baseline=True, out_path=out)
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["llm_backend"] == "fake"
    assert len(report["tasks"]) == 5  # 3 short + long_deep_archive + xl_town_archive
    assert report["token_ratio"] is not None and report["token_ratio"] > 0
    assert report["token_source"] == "estimated"
    assert report["budget_capped"] is False
    for entry in report["tasks"]:
        protocol = entry["protocol"]
        assert protocol["settled"] is True
        assert protocol["solved"] is True
        assert protocol["llm_calls"] <= 4 * 30  # within the per-worker budget
        assert "distinct_eligible_steering_readers" in protocol
        assert entry["baseline_iterative"]["solved"] is True
        assert entry["baseline_iterative"]["llm_calls"] > 1
        if entry["baseline_onecall"]["feasible"]:
            assert entry["baseline_onecall"]["llm_calls"] == 1
        else:
            assert entry["name"].startswith("xl_")
        # Somebody earned protocol payouts.
        assert any(delta > 0 for delta in protocol["payoffs"].values())


def test_long_document_fixture_is_genuinely_long():
    fixture = load_fixture(TASKS / "long_deep_archive.json")
    assert 4 <= len(fixture["documents"]) <= 8
    for name in fixture["documents"]:
        size = (CORPUS / name).stat().st_size
        assert 20_000 <= size <= 50_000  # the long-document regime per spec
    # The question genuinely needs evidence from >= 3 documents.
    assert len(fixture["gold_facts"]) >= 3


def test_worker_budget_stops_cleanly(tmp_path):
    fixture = load_fixture(sorted(TASKS.glob("*.json"))[0])
    store = Store(tmp_path)
    sponsor = crypto.keypair_from_seed("budget-sponsor")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = make_realdoc_task(
        store,
        CORPUS,
        question=fixture["question"],
        gold_facts=fixture["gold_facts"],
        documents=fixture["documents"],
        sponsor_pubkey=sponsor.pubkey,
    )
    stub = make_fake_complete(fixture, CORPUS)
    worker = LLMWorker.create("budget-worker", fixture["question"], stub, max_calls=0)
    store.create_identity(worker.pubkey, 200_000)
    for _ in range(3):
        worker.step(store, task_id)
    assert worker.calls == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0


def test_worker_rejects_non_verbatim_sentences_before_paying_fee(tmp_path):
    fixture = load_fixture(sorted(TASKS.glob("*.json"))[0])
    store = Store(tmp_path)
    sponsor = crypto.keypair_from_seed("verbatim-sponsor")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = make_realdoc_task(
        store,
        CORPUS,
        question=fixture["question"],
        gold_facts=fixture["gold_facts"],
        documents=fixture["documents"],
        sponsor_pubkey=sponsor.pubkey,
    )
    hallucinating = CountingComplete(
        lambda _prompt: json.dumps(
            {"sentence": "this sentence appears in no document at all and is invented"}
        )
    )
    worker = LLMWorker.create("hallucinator", fixture["question"], hallucinating)
    store.create_identity(worker.pubkey, 200_000)
    for _ in range(6):
        worker.step(store, task_id)
    # Three extraction attempts on the leased subtask, then it gives up; the
    # admission fee was never paid because the local verbatim check failed.
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    fees = store.conn.execute(
        "SELECT COUNT(*) AS n FROM transfers WHERE reason = 'FEE'"
    ).fetchone()["n"]
    assert fees == 0
    assert hallucinating.calls == 3


def test_realdoc_protocol_run_settles_and_conserves(tmp_path):
    fixture = load_fixture(sorted(TASKS.glob("*.json"))[0])
    store = Store(tmp_path)
    sponsor = crypto.keypair_from_seed("realdoc-sponsor")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = make_realdoc_task(
        store,
        CORPUS,
        question=fixture["question"],
        gold_facts=fixture["gold_facts"],
        documents=fixture["documents"],
        sponsor_pubkey=sponsor.pubkey,
    )
    stub = make_fake_complete(fixture, CORPUS)
    workers = [LLMWorker.create(f"rd{i}", fixture["question"], stub) for i in range(3)]
    for worker in workers:
        store.create_identity(worker.pubkey, 200_000)
    coordinator = Coordinator(store, LLMVerifier(complete=stub))
    for _ in range(60):
        for worker in workers:
            worker.step(store, task_id)
        coordinator.tick()
        if store.task_row(task_id)["settlement_applied"]:
            break
    task = store.task_row(task_id)
    assert task["settlement_applied"] == 1
    bounty_out = sum(
        row["amount_µ"]
        for row in store.transfer_rows()
        if row["reason"] in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
        and (row["from_pubkey"] or "").startswith("ESCROW:")
    )
    assert bounty_out == 1_000_000
    # The answer cited the admitted FACTs; the gold facts were extracted verbatim.
    admitted = {c["body"] for c in store.admitted_claims(task_id) if c["kind"] == "FACT"}
    assert set(fixture["gold_facts"]) <= admitted
