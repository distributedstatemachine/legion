from __future__ import annotations

import json
import subprocess
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from legion import settlement


def _snapshot(claims, answer_id="answer", fetches=None, version=None):
    snapshot = {
        "task_id": "task",
        "bounty_µ": 1_000_000,
        "answer_claim_id": answer_id,
        "claims": claims,
        "fetches": fetches or [],
    }
    if version is not None:
        snapshot["settlement_version"] = version
    return snapshot


def _claim(claim_id, author, kind="FACT", cites=(), docs=(), epoch=0):
    return {
        "claim_id": claim_id,
        "task_id": "task",
        "author": author,
        "kind": kind,
        "body": f"body {claim_id}",
        "evidence": [{"doc_hash": doc, "ref": {}} for doc in docs],
        "cites": list(cites),
        "epoch_submitted": epoch,
        "status": "ADMITTED",
    }


def _steering_to(transfers, author):
    return sum(
        t.amount_mu for t in transfers if t.reason == "PAYOUT_STEERING" and t.to_pubkey == author
    )


@st.composite
def snapshot_strategy(draw):
    n = draw(st.integers(min_value=2, max_value=8))
    claims = {}
    for i in range(n):
        claim_id = f"c{i}"
        if i == n - 1:
            kind = "ANSWER"
        else:
            kind = draw(st.sampled_from(["FACT", "FACT", "FAIL", "CONSTRAINT"]))
        cite_indices = draw(
            st.lists(
                st.integers(min_value=0, max_value=max(i - 1, 0)),
                unique=True,
                max_size=min(3, i),
            )
        )
        if i == 0:
            cite_indices = []
        claims[claim_id] = {
            "claim_id": claim_id,
            "task_id": "task",
            "author": f"author{draw(st.integers(min_value=0, max_value=3))}",
            "kind": kind,
            "body": f"body {claim_id}",
            "evidence": [],
            "cites": [f"c{j}" for j in sorted(cite_indices)],
            "epoch_submitted": i,
            "status": "ADMITTED",
        }
    fetches = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "reader": st.sampled_from([f"author{i}" for i in range(4)]),
                    "object_id": st.sampled_from([f"c{i}" for i in range(n)]),
                    "epoch": st.integers(min_value=0, max_value=20),
                }
            ),
            max_size=20,
        )
    )
    return _snapshot(claims, answer_id=f"c{n - 1}", fetches=fetches)


@given(snapshot_strategy())
@settings(max_examples=50, deadline=None)
def test_conservation_property(snapshot):
    for version in (1, 2):
        transfers = settlement.settle(snapshot, version=version)
        bounty_total = sum(
            transfer.amount_mu
            for transfer in transfers
            if transfer.reason in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
        )
        assert bounty_total == 1_000_000
        assert all(transfer.amount_mu >= 0 for transfer in transfers)


@given(snapshot_strategy(), st.sampled_from([1, 2]))
@settings(max_examples=12, deadline=None)
def test_determinism_property_across_runs_and_processes(snapshot, version):
    snapshot = dict(snapshot, settlement_version=version)
    local = [
        json.dumps([transfer.to_dict() for transfer in settlement.settle(snapshot)], sort_keys=True)
        for _ in range(3)
    ]
    assert local[0] == local[1] == local[2]
    code = (
        "import json,sys; "
        "from legion.settlement import settle; "
        "snap=json.loads(sys.stdin.read()); "
        "print(json.dumps([t.to_dict() for t in settle(snap)], sort_keys=True, separators=(',', ':')))"
    )
    payload = json.dumps(snapshot, sort_keys=True).encode()
    external_a = subprocess.check_output([sys.executable, "-c", code], input=payload)
    external_b = subprocess.check_output([sys.executable, "-c", code], input=payload)
    assert external_a == external_b
    assert external_a.decode().strip() == json.dumps(
        [transfer.to_dict() for transfer in settlement.settle(snapshot)],
        sort_keys=True,
        separators=(",", ":"),
    )


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=40, deadline=None)
def test_keep_fraction_invariance_property(_):
    base_claims = {
        "p0": {
            "claim_id": "p0",
            "task_id": "task",
            "author": "root0",
            "kind": "FACT",
            "body": "root",
            "evidence": [],
            "cites": [],
            "epoch_submitted": 0,
            "status": "ADMITTED",
        },
        "p1": {
            "claim_id": "p1",
            "task_id": "task",
            "author": "root1",
            "kind": "FACT",
            "body": "extra",
            "evidence": [],
            "cites": [],
            "epoch_submitted": 0,
            "status": "ADMITTED",
        },
        "n": {
            "claim_id": "n",
            "task_id": "task",
            "author": "node",
            "kind": "FACT",
            "body": "node",
            "evidence": [],
            "cites": ["p0"],
            "epoch_submitted": 1,
            "status": "ADMITTED",
        },
        "answer": {
            "claim_id": "answer",
            "task_id": "task",
            "author": "finisher",
            "kind": "ANSWER",
            "body": "answer",
            "evidence": [],
            "cites": ["n"],
            "epoch_submitted": 2,
            "status": "ADMITTED",
        },
    }
    variant_claims = json.loads(json.dumps(base_claims))
    variant_claims["n"]["cites"] = ["p0", "p1"]
    _, base_kept = settlement.derivation_flows(_snapshot(base_claims))
    _, variant_kept = settlement.derivation_flows(_snapshot(variant_claims))
    assert base_kept["n"] == variant_kept["n"]


def test_uncited_claim_earns_nothing_and_no_fee_refund():
    claims = {
        "c0": {
            "claim_id": "c0",
            "task_id": "task",
            "author": "useful",
            "kind": "FACT",
            "body": "useful",
            "evidence": [],
            "cites": [],
            "epoch_submitted": 0,
            "status": "ADMITTED",
        },
        "unused": {
            "claim_id": "unused",
            "task_id": "task",
            "author": "unused-author",
            "kind": "FACT",
            "body": "unused",
            "evidence": [],
            "cites": [],
            "epoch_submitted": 0,
            "status": "ADMITTED",
        },
        "answer": {
            "claim_id": "answer",
            "task_id": "task",
            "author": "finisher",
            "kind": "ANSWER",
            "body": "answer",
            "evidence": [],
            "cites": ["c0"],
            "epoch_submitted": 1,
            "status": "ADMITTED",
        },
    }
    transfers = settlement.settle(_snapshot(claims))
    assert not [transfer for transfer in transfers if transfer.claim_id == "unused"]


def test_golden_seven_claim_derivation_vector():
    # BETA starts at answer=450000. answer keeps 225000 and passes 225000
    # equally to c1/c2/c3: 75000 each. c1 keeps 37500 and passes 18750
    # to c4 and c5. c2 keeps 37500 and passes 18750 to c5 and c6. c3 has
    # no parents and keeps 75000. Leaves: c4=18750, c5=37500, c6=18750.
    claims = {
        claim_id: {
            "claim_id": claim_id,
            "task_id": "task",
            "author": claim_id,
            "kind": "FACT",
            "body": claim_id,
            "evidence": [],
            "cites": cites,
            "epoch_submitted": index,
            "status": "ADMITTED",
        }
        for index, (claim_id, cites) in enumerate(
            [
                ("c4", []),
                ("c5", []),
                ("c6", []),
                ("c1", ["c4", "c5"]),
                ("c2", ["c5", "c6"]),
                ("c3", []),
                ("answer", ["c1", "c2", "c3"]),
            ]
        )
    }
    claims["answer"]["kind"] = "ANSWER"
    _, kept = settlement.derivation_flows(_snapshot(claims))
    assert kept == {
        "answer": 225_000,
        "c1": 37_500,
        "c2": 37_500,
        "c3": 75_000,
        "c4": 18_750,
        "c5": 37_500,
        "c6": 18_750,
    }
    # This golden vector pins v1 semantics; the derivation pool is identical in
    # v2, so the v2 default must byte-match it as well.
    transfers_v1 = settlement.settle(_snapshot(claims), version=1)
    transfers_v2 = settlement.settle(_snapshot(claims))
    assert [t.to_dict() for t in transfers_v1] == [t.to_dict() for t in transfers_v2]
    refunds = [transfer for transfer in transfers_v1 if transfer.reason == "FEE_REFUND"]
    assert len(refunds) == 7


def _ring_capture_snapshot():
    """Four honest productive readers + one ring reader that fetched 3 ring FAILs."""
    claims = {}
    for i in range(1, 5):  # facts f1..f4 by R1..R4, doc d1..d4, epoch 1
        claims[f"f{i}"] = _claim(f"f{i}", f"R{i}", docs=[f"d{i}"], epoch=1)
    claims["hf"] = _claim("hf", "H", kind="FAIL", docs=["d2", "d3", "d4"], epoch=0)
    for j in range(1, 4):  # ring FAILs by Z, all on d1 (overlapping R1's work)
        claims[f"zf{j}"] = _claim(f"zf{j}", "Z", kind="FAIL", docs=["d1"], epoch=0)
    claims["answer"] = _claim(
        "answer", "F", kind="ANSWER", cites=["f1", "f2", "f3", "f4"], epoch=2
    )
    fetches = [
        {"reader": "R2", "object_id": "hf", "epoch": 0},
        {"reader": "R3", "object_id": "hf", "epoch": 0},
        {"reader": "R4", "object_id": "hf", "epoch": 0},
        {"reader": "F", "object_id": "hf", "epoch": 1},
        # R1 is the ring's productive member: it fetches every ring FAIL.
        {"reader": "R1", "object_id": "zf1", "epoch": 0},
        {"reader": "R1", "object_id": "zf2", "epoch": 0},
        {"reader": "R1", "object_id": "zf3", "epoch": 0},
    ]
    return _snapshot(claims, fetches=fetches)


def test_steering_v2_ring_capture_bound():
    snapshot = _ring_capture_snapshot()
    n_claims = len(snapshot["claims"])
    n_productive_readers = 5  # R1..R4 + finisher F

    ring_v1 = _steering_to(settlement.settle(snapshot, version=1), "Z")
    ring_v2 = _steering_to(settlement.settle(snapshot, version=2), "Z")

    # v1: each fetched ring FAIL adds a full weight unit -> 3/(3+4) of GAMMA.
    assert ring_v1 == settlement.GAMMA * 3 // 7
    # v2: the ring reader's endowment is normalized; capture is bounded by
    # GAMMA / (#productive readers) regardless of how many ring FAILs exist.
    assert ring_v2 <= settlement.GAMMA // n_productive_readers + n_claims
    assert ring_v2 < ring_v1
    # The honest FAIL takes the rest.
    honest_v2 = _steering_to(settlement.settle(snapshot, version=2), "H")
    assert honest_v2 + ring_v2 == settlement.GAMMA


def test_steering_v2_relevance_scoping():
    claims = {
        "f1": _claim("f1", "R", docs=["d1"], epoch=1),
        # Disjoint docs: R fetched it, but it informed nothing R authored.
        "fail_disjoint": _claim("fail_disjoint", "H", kind="FAIL", docs=["dX"], epoch=0),
        # Right doc, but fetched after R's last productive claim.
        "fail_late": _claim("fail_late", "H", kind="FAIL", docs=["d1"], epoch=4),
        "answer": _claim("answer", "F", kind="ANSWER", cites=["f1"], epoch=2),
    }
    fetches = [
        {"reader": "R", "object_id": "fail_disjoint", "epoch": 0},
        {"reader": "R", "object_id": "fail_late", "epoch": 5},
    ]
    snapshot = _snapshot(claims, fetches=fetches)

    transfers_v2 = settlement.settle(snapshot, version=2)
    assert _steering_to(transfers_v2, "H") == 0
    assert any(t.reason == "BURN" and t.amount_mu == settlement.GAMMA for t in transfers_v2)
    # v1 on the same snapshot pays the readership rubber stamp - that is the bug.
    assert _steering_to(settlement.settle(snapshot, version=1), "H") == settlement.GAMMA


def test_steering_v1_raw_count_backward_compat():
    claims = {
        "f1": _claim("f1", "A", docs=["d1"], epoch=0),
        "fail": _claim("fail", "H", kind="FAIL", docs=["d1"], epoch=0),
        "answer": _claim("answer", "F", kind="ANSWER", cites=["f1"], epoch=1),
    }
    fetches = [{"reader": "A", "object_id": "fail", "epoch": 0}]
    transfers = settlement.settle(_snapshot(claims, fetches=fetches), version=1)
    assert _steering_to(transfers, "H") == settlement.GAMMA
    # The snapshot's own settlement_version key is honored when no explicit arg.
    pinned = _snapshot(claims, fetches=fetches, version=1)
    assert [t.to_dict() for t in settlement.settle(pinned)] == [
        t.to_dict() for t in transfers
    ]
