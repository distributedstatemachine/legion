"""Settlement-equivalence adapters: one scenario format, two engines.

`settle_legion` builds a legion snapshot from a scenario and runs the integer
engine; `settle_sim` drives the vendored research sim on an isomorphic Episode
(claims/fetches injected directly, bypassing its discovery loop) and converts
its float payout fractions to µ with round(x * 1_000_000).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legion import settlement
from tools.sim.sim import Episode, SimClaim, settle_episode

PAYOUT_REASONS = {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING"}


def load_scenario(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _claim_docs(scenario: dict[str, Any], claim: dict[str, Any]) -> list[str]:
    # Docs may live on the claim itself or in the top-level "docs" map.
    return list(claim.get("docs") or scenario.get("docs", {}).get(claim["id"], []))


def _claim_epoch(claim: dict[str, Any], index: int) -> int:
    # Submission order doubles as the epoch when the scenario omits one.
    return int(claim.get("epoch", index))


def settle_legion(scenario: dict[str, Any], version: int = 2) -> dict[str, int]:
    claims: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(scenario["claims"]):
        claims[claim["id"]] = {
            "claim_id": claim["id"],
            "task_id": scenario["name"],
            "author": claim["author"],
            "kind": claim.get("kind", "FACT"),
            "body": f"body {claim['id']}",
            "evidence": [
                {"doc_hash": doc, "ref": {}} for doc in _claim_docs(scenario, claim)
            ],
            "cites": list(claim.get("cites", [])),
            "epoch_submitted": _claim_epoch(claim, index),
            "status": "ADMITTED",
        }
    snapshot = {
        "task_id": scenario["name"],
        "bounty_µ": scenario["bounty_µ"],
        "answer_claim_id": scenario["answer"],
        "claims": claims,
        "fetches": [
            {"reader": f["reader"], "object_id": f["object"], "epoch": f["epoch"]}
            for f in scenario.get("fetches", [])
        ],
    }
    totals: dict[str, int] = {}
    for transfer in settlement.settle(snapshot, version=version):
        if transfer.reason in PAYOUT_REASONS:
            totals[transfer.to_pubkey] = totals.get(transfer.to_pubkey, 0) + transfer.amount_mu
    return totals


def settle_sim(scenario: dict[str, Any], version: int = 1) -> dict[str, int]:
    episode = Episode(answer_id=scenario["answer"])
    for index, claim in enumerate(scenario["claims"]):
        episode.add_claim(
            SimClaim(
                claim_id=claim["id"],
                author=claim["author"],
                kind=claim.get("kind", "FACT"),
                cites=tuple(claim.get("cites", [])),
                docs=frozenset(_claim_docs(scenario, claim)),
                epoch=_claim_epoch(claim, index),
            )
        )
    episode.fetches = [
        (f["reader"], f["object"], f["epoch"]) for f in scenario.get("fetches", [])
    ]
    fractions = settle_episode(episode, steering_version=version)
    bounty = scenario["bounty_µ"]
    return {author: round(frac * bounty) for author, frac in fractions.items()}
