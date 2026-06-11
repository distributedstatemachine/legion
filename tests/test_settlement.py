from __future__ import annotations

import json
import subprocess
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from legion import settlement


def _snapshot(claims, answer_id="answer", fetches=None):
    return {
        "task_id": "task",
        "bounty_µ": 1_000_000,
        "answer_claim_id": answer_id,
        "claims": claims,
        "fetches": fetches or [],
    }


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
    transfers = settlement.settle(snapshot)
    bounty_total = sum(
        transfer.amount_mu
        for transfer in transfers
        if transfer.reason in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
    )
    assert bounty_total == 1_000_000
    assert all(transfer.amount_mu >= 0 for transfer in transfers)


@given(snapshot_strategy())
@settings(max_examples=12, deadline=None)
def test_determinism_property_across_runs_and_processes(snapshot):
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
    transfers = settlement.settle(_snapshot(claims))
    refunds = [transfer for transfer in transfers if transfer.reason == "FEE_REFUND"]
    assert len(refunds) == 7
