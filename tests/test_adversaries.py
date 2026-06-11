"""Adversarial strategies from the simulation zoo, run against the real engine.

Qualitative results these tests pin down:
- honest > racing hoarder (the hoarder free-rides, races the answer, and loses)
- a poisoner's decoy FACT is rejected at admission and its fee is never refunded
- a citation ring cannot increase its combined derivation take (flat-flow
  invariance) and a materiality challenge makes the ring strictly worse off
- duplicate claims earn nothing under flat backward flow
"""
from __future__ import annotations

from legion import claims, crypto, settlement, tasks
from legion.admission import AdmissionGate, MockVerifier
from legion.admission_constants import ADMISSION_FEE, CHALLENGE_BOND
from legion.challenges import ChallengeEngine
from legion.cli import run_demo
from legion.store import Store


def _identity(store: Store, seed: str, balance: int = 1_000_000) -> crypto.Keypair:
    keypair = crypto.keypair_from_seed(seed)
    store.create_identity(keypair.pubkey, balance)
    return keypair


def _setup(store: Store, *, seed: int = 29, K: int = 2):
    sponsor = _identity(store, f"adversary-sponsor-{seed}", 3_000_000)
    task_id = tasks.create_fact_chain_task(store, K=K, D=2, seed=seed, sponsor_pubkey=sponsor.pubkey)
    answer_key = store.answer_key(task_id)
    verifier = MockVerifier(answer_key)
    gate = AdmissionGate(store, verifier)
    return task_id, answer_key, verifier, gate


def _submit_fact(store, gate, task_id, author, index, cites=None):
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


def _submit_answer(store, gate, task_id, author, cited_claims, evidence=None):
    spec = store.task_spec(task_id)
    for cited in cited_claims:
        store.fetch(author.pubkey, cited["claim_id"])
    answer = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="ANSWER",
        body="\n".join(doc["fact"] for doc in spec["docs"]),
        evidence=evidence or [],
        cites=[cited["claim_id"] for cited in cited_claims],
    )
    store.submit_claim(answer)
    gate.process_pending()
    assert store.claim(answer["claim_id"])["status"] == "ADMITTED"
    return answer


def _derivation_paid(transfers, claim_id):
    return sum(
        transfer.amount_mu
        for transfer in transfers
        if transfer.reason == "PAYOUT_DERIVATION" and transfer.claim_id == claim_id
    )


def test_racing_hoarder_free_rides_races_and_loses(tmp_path):
    output = run_demo(workers=6, honest=4, hoarders=2, K=6, D=3, seed=42, root=tmp_path)
    assert "mean_hoarder_delta -10000" in output  # exactly one lost admission fee
    store = Store(tmp_path)
    answers = store.conn.execute(
        "SELECT subtask_id, status FROM claims WHERE kind = 'ANSWER' ORDER BY rowid"
    ).fetchall()
    # The winning honest answer plus one racing answer per hoarder.
    assert len(answers) == 3
    assert answers[0]["subtask_id"] is not None  # honest, leased
    assert all(row["subtask_id"] is None for row in answers[1:])  # hoarders raced unleased
    # FAIL claims were published, admitted, and the steering pool flowed.
    fails = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE kind = 'FAIL' AND status = 'ADMITTED'"
    ).fetchone()
    assert fails["n"] > 0
    steering = sum(
        row["amount_µ"]
        for row in store.transfer_rows()
        if row["reason"] == "PAYOUT_STEERING"
    )
    assert steering == settlement.GAMMA
    store.close()


def test_fail_admission_gate_is_real(tmp_path):
    store = Store(tmp_path)
    task_id, answer_key, _, gate = _setup(store)
    spec = store.task_spec(task_id)
    author = _identity(store, "fail-author")

    # A body that merely claims failure is rejected: no span, not a decoy.
    fake = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FAIL",
        body="FAIL: trust me, this candidate was wrong",
    )
    store.submit_claim(fake)
    gate.process_pending()
    assert store.claim(fake["claim_id"])["status"] == "REJECTED"

    # A fact sentence dressed up as a FAIL is rejected: it is not a decoy.
    not_a_decoy = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FAIL",
        body=spec["docs"][0]["fact"],
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], spec["docs"][0]["fact"])],
    )
    store.submit_claim(not_a_decoy)
    gate.process_pending()
    assert store.claim(not_a_decoy["claim_id"])["status"] == "REJECTED"

    # A genuine verified negative result is admitted.
    decoy = answer_key["decoys"]["0"][0]
    real = claims.build_claim(
        private_key=author.private_key,
        author=author.pubkey,
        task_id=task_id,
        kind="FAIL",
        body=decoy,
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], decoy)],
    )
    store.submit_claim(real)
    gate.process_pending()
    assert store.claim(real["claim_id"])["status"] == "ADMITTED"


def test_poisoner_decoy_fact_rejected_and_fee_lost(tmp_path):
    store = Store(tmp_path)
    task_id, answer_key, _, gate = _setup(store)
    spec = store.task_spec(task_id)
    poisoner = _identity(store, "poisoner", 100_000)
    decoy = answer_key["decoys"]["0"][0]
    claim = claims.build_claim(
        private_key=poisoner.private_key,
        author=poisoner.pubkey,
        task_id=task_id,
        kind="FACT",
        body=decoy,
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], decoy)],
    )
    store.submit_claim(claim)
    gate.process_pending()
    stored = store.claim(claim["claim_id"])
    assert stored["status"] == "REJECTED"
    assert stored["reject_reason"] == "semantic"
    assert store.balance(poisoner.pubkey) == 100_000 - ADMISSION_FEE


def test_citation_ring_gains_nothing_and_loses_the_slash(tmp_path):
    # Ring = {Y, Z}. Z publishes a (real) FAIL; Y pads its fact claim with a
    # citation of Z to funnel derivation flow into the ring.
    store = Store(tmp_path)
    task_id, answer_key, verifier, gate = _setup(store)
    spec = store.task_spec(task_id)
    x = _identity(store, "honest-x")
    y = _identity(store, "ring-y")
    z = _identity(store, "ring-z")
    finisher = _identity(store, "ring-finisher")
    challenger = _identity(store, "ring-challenger")

    decoy = answer_key["decoys"]["0"][0]
    z_fail = claims.build_claim(
        private_key=z.private_key,
        author=z.pubkey,
        task_id=task_id,
        kind="FAIL",
        body=decoy,
        evidence=[claims.evidence_ref(spec["docs"][0]["doc_hash"], decoy)],
    )
    store.submit_claim(z_fail)
    gate.process_pending()
    assert store.claim(z_fail["claim_id"])["status"] == "ADMITTED"

    x_fact = _submit_fact(store, gate, task_id, x, 0)
    store.fetch(y.pubkey, z_fail["claim_id"])
    y_fact = _submit_fact(store, gate, task_id, y, 1, cites=[z_fail["claim_id"]])
    _submit_answer(store, gate, task_id, finisher, [x_fact, y_fact])

    pre = settlement.settle(store.snapshot(task_id))
    # Flat-flow invariance: the ring's combined derivation take equals what Y
    # alone would have kept without the padding cite (112_500 µ); padding only
    # moves money inside the ring, away from no one outside it except Y itself.
    y_pre = _derivation_paid(pre, y_fact["claim_id"])
    z_pre = _derivation_paid(pre, z_fail["claim_id"])
    x_pre = _derivation_paid(pre, x_fact["claim_id"])
    assert z_pre > 0
    assert y_pre + z_pre == 112_500
    assert x_pre == 112_500

    # The materiality challenge strips the immaterial cite and slashes Y.
    engine = ChallengeEngine(store, verifier)
    y_balance_before = store.balance(y.pubkey)
    assert engine.file_materiality(challenger.pubkey, y_fact["claim_id"], z_fail["claim_id"])
    assert store.balance(y.pubkey) == y_balance_before - CHALLENGE_BOND

    post = settlement.settle(store.snapshot(task_id))
    assert _derivation_paid(post, z_fail["claim_id"]) == 0
    assert _derivation_paid(post, y_fact["claim_id"]) == 112_500
    assert _derivation_paid(post, x_fact["claim_id"]) == 112_500
    # TODO: the steering pool is still gameable by a ring (Y fetching Z's FAIL
    # makes Z a paid steerer once Y is productive). Needs a usefulness rule
    # stronger than raw readership before this PoC graduates.


def test_duplicate_claim_earns_nothing_under_flat_flow(tmp_path):
    store = Store(tmp_path)
    task_id, _, _, gate = _setup(store)
    original_author = _identity(store, "dup-original")
    duplicate_author = _identity(store, "dup-copycat")
    other_author = _identity(store, "dup-other")
    finisher = _identity(store, "dup-finisher")

    original = _submit_fact(store, gate, task_id, original_author, 0)
    duplicate = _submit_fact(store, gate, task_id, duplicate_author, 0)
    assert duplicate["claim_id"] != original["claim_id"]
    other = _submit_fact(store, gate, task_id, other_author, 1)
    _submit_answer(store, gate, task_id, finisher, [original, other])

    transfers = settlement.settle(store.snapshot(task_id))
    assert not [t for t in transfers if t.claim_id == duplicate["claim_id"]]
    assert _derivation_paid(transfers, original["claim_id"]) > 0


def test_demo_with_K9_admits_answer(tmp_path):
    # Regression: the answer used to re-attach one evidence ref per fact, so
    # K=9 exceeded the 8-ref admission cap and no answer could ever be admitted.
    output = run_demo(workers=6, honest=4, hoarders=2, K=9, D=3, seed=7, root=tmp_path)
    assert "bounty_paid_or_burned 1000000" in output
