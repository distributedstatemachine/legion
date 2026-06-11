from __future__ import annotations

import os
from pathlib import Path

import pytest

from legion import claims, crypto, settlement, tasks
from legion.admission import AdmissionGate, LLMVerifier, MockVerifier
from legion.challenges import ChallengeEngine
from legion.cli import run_demo
from legion.store import Store


def _identity(store: Store, seed: str, balance: int = 1_000_000) -> crypto.Keypair:
    keypair = crypto.keypair_from_seed(seed)
    store.create_identity(keypair.pubkey, balance)
    return keypair


def _create_task(store: Store) -> str:
    sponsor = _identity(store, "challenge-sponsor", 3_000_000)
    return tasks.create_fact_chain_task(store, K=2, D=2, seed=29, sponsor_pubkey=sponsor.pubkey)


def _submit_fact(store: Store, gate: AdmissionGate, task_id: str, author, index: int, cites=None):
    spec = store.task_spec(task_id)
    fact = spec["docs"][index]["fact"]
    doc_hash = spec["docs"][index]["doc_hash"]
    claim = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FACT",
        body=fact,
        evidence=[claims.evidence_ref(doc_hash, fact)],
        cites=cites or [],
    )
    store.submit_claim(claim)
    gate.process_pending()
    assert store.claim(claim["claim_id"])["status"] == "ADMITTED"
    return claim


def test_under_citation_challenge_inserts_edge_and_increases_omitted_payout(tmp_path):
    store = Store(tmp_path)
    task_id = _create_task(store)
    answer_key = store.answer_key(task_id)
    verifier = MockVerifier(answer_key)
    gate = AdmissionGate(store, verifier)
    omitted_author = _identity(store, "omitted-author")
    cited_author = _identity(store, "cited-author")
    finisher = _identity(store, "finisher")
    challenger = _identity(store, "challenger")

    omitted = _submit_fact(store, gate, task_id, omitted_author, 0)
    cited = _submit_fact(store, gate, task_id, cited_author, 1)
    store.fetch(finisher.pubkey, omitted["claim_id"])
    store.fetch(finisher.pubkey, cited["claim_id"])

    spec = store.task_spec(task_id)
    body = "ANSWER: " + "\n".join([spec["docs"][0]["fact"], spec["docs"][1]["fact"]])
    answer = claims.build_claim(
        private_key=finisher.private_key,
        author=finisher.pubkey,
        task_id=task_id,
        kind="ANSWER",
        body=body,
        evidence=[
            claims.evidence_ref(spec["docs"][0]["doc_hash"], spec["docs"][0]["fact"]),
            claims.evidence_ref(spec["docs"][1]["doc_hash"], spec["docs"][1]["fact"]),
        ],
        cites=[cited["claim_id"]],
    )
    store.submit_claim(answer)
    gate.process_pending()
    assert store.claim(answer["claim_id"])["status"] == "ADMITTED"

    unchallenged = settlement.settle(store.snapshot(task_id))
    before = sum(
        transfer.amount_mu
        for transfer in unchallenged
        if transfer.to_pubkey == omitted_author.pubkey
        and transfer.reason in {"PAYOUT_DERIVATION", "PAYOUT_STEERING"}
    )
    assert before == 0

    engine = ChallengeEngine(store, verifier)
    assert engine.file_under_citation(challenger.pubkey, answer["claim_id"], omitted["claim_id"])
    challenged = settlement.settle(store.snapshot(task_id))
    after = sum(
        transfer.amount_mu
        for transfer in challenged
        if transfer.to_pubkey == omitted_author.pubkey
        and transfer.reason in {"PAYOUT_DERIVATION", "PAYOUT_STEERING"}
    )
    assert after > before


def test_materiality_challenge_removes_useless_citation(tmp_path):
    store = Store(tmp_path)
    task_id = _create_task(store)
    answer_key = store.answer_key(task_id)
    verifier = MockVerifier(answer_key)
    gate = AdmissionGate(store, verifier)
    irrelevant_author = _identity(store, "irrelevant-author")
    target_author = _identity(store, "target-author")
    finisher = _identity(store, "materiality-finisher")
    challenger = _identity(store, "materiality-challenger")

    irrelevant = _submit_fact(store, gate, task_id, irrelevant_author, 0)
    store.fetch(target_author.pubkey, irrelevant["claim_id"])
    target = _submit_fact(
        store, gate, task_id, target_author, 1, cites=[irrelevant["claim_id"]]
    )

    spec = store.task_spec(task_id)
    store.fetch(finisher.pubkey, target["claim_id"])
    answer = claims.build_claim(
        private_key=finisher.private_key,
        author=finisher.pubkey,
        task_id=task_id,
        kind="ANSWER",
        body="ANSWER: " + "\n".join([spec["docs"][0]["fact"], spec["docs"][1]["fact"]]),
        evidence=[
            claims.evidence_ref(spec["docs"][0]["doc_hash"], spec["docs"][0]["fact"]),
            claims.evidence_ref(spec["docs"][1]["doc_hash"], spec["docs"][1]["fact"]),
        ],
        cites=[target["claim_id"]],
    )
    store.submit_claim(answer)
    gate.process_pending()
    assert store.task_row(task_id)["status"] == "CLOSED"

    engine = ChallengeEngine(store, verifier)
    assert engine.file_materiality(challenger.pubkey, target["claim_id"], irrelevant["claim_id"])
    refreshed = store.claim(target["claim_id"])
    assert store.latest_cites(refreshed) == []

def test_challenge_resolves_atomically_when_author_cannot_cover_slash(tmp_path):
    # Regression: the cite override used to be committed before the SLASH
    # transfer, so an insolvent author wedged the challenge mid-flight with the
    # bond stranded. Resolution is now one transaction with the slash capped at
    # the author's balance.
    store = Store(tmp_path)
    task_id = _create_task(store)
    answer_key = store.answer_key(task_id)
    verifier = MockVerifier(answer_key)
    gate = AdmissionGate(store, verifier)
    omitted_author = _identity(store, "wedge-omitted")
    cited_author = _identity(store, "wedge-cited")
    finisher = _identity(store, "wedge-finisher", 15_000)  # 5_000 left after the fee
    challenger = _identity(store, "wedge-challenger", 100_000)

    omitted = _submit_fact(store, gate, task_id, omitted_author, 0)
    cited = _submit_fact(store, gate, task_id, cited_author, 1)
    store.fetch(finisher.pubkey, omitted["claim_id"])
    store.fetch(finisher.pubkey, cited["claim_id"])

    spec = store.task_spec(task_id)
    answer = claims.build_claim(
        private_key=finisher.private_key,
        author=finisher.pubkey,
        task_id=task_id,
        kind="ANSWER",
        body="ANSWER: " + "\n".join([spec["docs"][0]["fact"], spec["docs"][1]["fact"]]),
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], spec["docs"][0]["fact"])],
        cites=[cited["claim_id"]],
    )
    store.submit_claim(answer)
    gate.process_pending()
    assert store.claim(answer["claim_id"])["status"] == "ADMITTED"
    assert store.balance(finisher.pubkey) == 5_000

    engine = ChallengeEngine(store, verifier)
    assert engine.file_under_citation(challenger.pubkey, answer["claim_id"], omitted["claim_id"])

    row = store.conn.execute("SELECT status, upheld FROM challenges").fetchone()
    assert row["status"] == "RESOLVED" and row["upheld"] == 1
    assert omitted["claim_id"] in store.latest_cites(store.claim(answer["claim_id"]))
    assert store.balance(finisher.pubkey) == 0  # slash capped at remaining balance
    assert store.balance(challenger.pubkey) == 105_000  # bond back + 5_000 slash
    assert store.pseudo_balances().get(f"BOND:challenge:{answer['claim_id']}", 0) == 0


def test_demo_matches_golden_file(tmp_path):
    output = run_demo(workers=6, honest=4, hoarders=2, K=6, D=3, seed=42, root=tmp_path)
    golden = Path(__file__).with_name("golden_demo_seed42.txt").read_text(encoding="utf-8")
    assert output == golden
    steering_paid = int(
        next(line for line in output.splitlines() if line.startswith("steering_paid ")).split()[1]
    )
    assert steering_paid > 0  # gamma must flow under settlement v2 as well
    assert "lease_bonds_burned 0" in output


def _quote_from_prompt(prompt: str) -> str:
    # Pull the first SPAN block's text back out of the structured prompt, the
    # way an honest model would quote it.
    marker = "SPAN 0\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n</data>", start)
    return prompt[start:end][:80]


@pytest.mark.skipif(not os.environ.get("VSCP_LLM"), reason="VSCP_LLM unset")
def test_llm_verifier_smoke_path(tmp_path):
    import json as json_module

    yes = LLMVerifier(
        complete=lambda prompt: json_module.dumps(
            {"supported": True, "quote": _quote_from_prompt(prompt)}
        )
    )
    no = LLMVerifier(complete=lambda _prompt: json_module.dumps({"supported": False, "quote": ""}))
    span = "a sufficiently long span of genuine evidence text for the smoke test"
    assert yes.supports({"body": "supported claim", "kind": "FACT"}, [span])
    assert not no.supports({"body": "unsupported", "kind": "FACT"}, ["different long span text here"])

    store = Store(tmp_path)
    sponsor = _identity(store, "llm-sponsor", 2_000_000)
    task_id = tasks.create_fact_chain_task(store, K=1, D=1, seed=3, sponsor_pubkey=sponsor.pubkey)
    author = _identity(store, "llm-author")
    spec = store.task_spec(task_id)
    fact = spec["docs"][0]["fact"]
    claim = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FACT",
        body=fact,
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], fact)],
    )
    store.submit_claim(claim)
    AdmissionGate(store, yes).process_pending()
    assert store.claim(claim["claim_id"])["status"] == "ADMITTED"
