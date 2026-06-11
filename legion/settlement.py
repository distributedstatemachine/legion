from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


ALPHA = 350_000
BETA = 450_000
GAMMA = 200_000
DELTA_NUM = 1
DELTA_DEN = 2
ADMISSION_FEE = 10_000
SETTLEMENT_VERSION = 2
READER_ENDOWMENT = 1_000_000  # endorsement micro-units per productive reader (v2)


@dataclass(frozen=True)
class Transfer:
    reason: str
    from_pubkey: str | None
    to_pubkey: str | None
    amount_mu: int
    claim_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["amount_µ"] = data.pop("amount_mu")
        return data


def canonical_snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _ancestor_order(answer_id: str, claims: dict[str, dict[str, Any]]) -> list[str]:
    ancestors: set[str] = set()

    def collect(node: str) -> None:
        if node in ancestors:
            return
        ancestors.add(node)
        for parent in claims[node].get("cites", []):
            if parent in claims:
                collect(parent)

    collect(answer_id)
    indegree = {claim_id: 0 for claim_id in ancestors}
    parents_by_child: dict[str, list[str]] = {}
    for child in ancestors:
        parents = [p for p in claims[child].get("cites", []) if p in ancestors]
        parents_by_child[child] = sorted(parents)
        for parent in parents:
            indegree[parent] += 1
    ready = sorted([node for node, degree in indegree.items() if degree == 0])
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for parent in parents_by_child[node]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                ready.append(parent)
                ready.sort()
    if len(order) != len(ancestors):
        raise ValueError("claim citations must form a DAG")
    return order


def _split_weighted(total: int, weights: dict[str, int]) -> dict[str, int]:
    positive = {key: weight for key, weight in weights.items() if weight > 0}
    if not positive:
        return {}
    denominator = sum(positive.values())
    amounts = {key: total * weight // denominator for key, weight in positive.items()}
    used = sum(amounts.values())
    remainders = {
        key: (total * weight) % denominator for key, weight in positive.items()
    }
    for key, _ in sorted(remainders.items(), key=lambda item: (-item[1], item[0]))[
        : total - used
    ]:
        amounts[key] += 1
    return amounts


def derivation_flows(snapshot: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    claims = snapshot["claims"]
    answer_id = snapshot["answer_claim_id"]
    if answer_id not in claims:
        raise ValueError("snapshot has no admitted answer")
    order = _ancestor_order(answer_id, claims)
    inflow = {answer_id: BETA}
    kept: dict[str, int] = {}
    for node in order:
        amount = inflow.get(node, 0)
        parents = [parent for parent in sorted(claims[node].get("cites", [])) if parent in claims]
        if not parents:
            kept[node] = kept.get(node, 0) + amount
            continue
        passed = amount * DELTA_NUM // DELTA_DEN
        kept[node] = kept.get(node, 0) + amount - passed
        share = passed // len(parents)
        remainder = passed % len(parents)
        for index, parent in enumerate(parents):
            inflow[parent] = inflow.get(parent, 0) + share + (1 if index < remainder else 0)
    return inflow, kept


def _claim_docs(claim: dict[str, Any], claims: dict[str, dict[str, Any]]) -> set[str]:
    """docs(c): evidence doc hashes; for evidence-less claims (notably ANSWERs)
    the union of the cited parents' own evidence docs (one level, no recursion)."""
    docs = {ref["doc_hash"] for ref in claim.get("evidence", []) if ref.get("doc_hash")}
    if docs:
        return docs
    for parent in claim.get("cites", []):
        if parent in claims:
            docs |= {
                ref["doc_hash"]
                for ref in claims[parent].get("evidence", [])
                if ref.get("doc_hash")
            }
    return docs


def _steering_weights_v1(
    snapshot: dict[str, Any], productive_authors: set[str]
) -> dict[str, int]:
    claims = snapshot["claims"]
    readers_by_object: dict[str, set[str]] = {}
    for fetch in snapshot.get("fetches", []):
        readers_by_object.setdefault(fetch["object_id"], set()).add(fetch["reader"])
    return {
        claim_id: len(readers_by_object.get(claim_id, set()) & productive_authors)
        for claim_id, claim in claims.items()
        if claim["kind"] in {"FAIL", "CONSTRAINT"}
    }


def _steering_weights_v2(
    snapshot: dict[str, Any], derivation_paid: dict[str, int]
) -> dict[str, int]:
    """Reader-normalized, relevance-scoped steering weights (settlement v2).

    Each productive reader holds exactly READER_ENDOWMENT endorsement units,
    split equally over the FAIL/CONSTRAINT claims *eligible* for that reader:
    fetched by the reader (not its own), at an epoch no later than one of the
    reader's productive claims whose docs overlap the steered claim's docs.
    A colluding reader therefore caps the ring's capture at one endowment no
    matter how many ring claims it fetches.
    """
    claims = snapshot["claims"]
    answer_id = snapshot["answer_claim_id"]
    docs = {claim_id: _claim_docs(claim, claims) for claim_id, claim in claims.items()}

    productive_claims_by_author: dict[str, set[str]] = {}
    for claim_id, amount in derivation_paid.items():
        if amount > 0:
            productive_claims_by_author.setdefault(claims[claim_id]["author"], set()).add(claim_id)
    productive_claims_by_author.setdefault(claims[answer_id]["author"], set()).add(answer_id)

    first_fetch: dict[tuple[str, str], int] = {}
    for fetch in snapshot.get("fetches", []):
        key = (fetch["reader"], fetch["object_id"])
        if key not in first_fetch or fetch["epoch"] < first_fetch[key]:
            first_fetch[key] = fetch["epoch"]

    steerable = sorted(
        claim_id
        for claim_id, claim in claims.items()
        if claim["kind"] in {"FAIL", "CONSTRAINT"}
    )
    weights = {claim_id: 0 for claim_id in steerable}
    for reader in sorted(productive_claims_by_author):
        eligible: list[str] = []
        for claim_id in steerable:
            if claims[claim_id]["author"] == reader:
                continue
            epoch = first_fetch.get((reader, claim_id))
            if epoch is None:
                continue
            fail_docs = docs[claim_id]
            if not fail_docs:
                continue
            if any(
                epoch <= claims[productive_id]["epoch_submitted"]
                and fail_docs & docs[productive_id]
                for productive_id in productive_claims_by_author[reader]
            ):
                eligible.append(claim_id)
        if not eligible:
            continue
        share = READER_ENDOWMENT // len(eligible)
        remainder = READER_ENDOWMENT % len(eligible)
        for index, claim_id in enumerate(eligible):  # already ascending claim_id
            weights[claim_id] += share + (1 if index < remainder else 0)
    return weights


def settle(snapshot: dict[str, Any], version: int | None = None) -> list[Transfer]:
    if snapshot.get("bounty_µ") != ALPHA + BETA + GAMMA:
        raise ValueError("this PoC settlement expects the fixed 1_000_000 µ bounty")
    resolved_version = version or snapshot.get("settlement_version") or SETTLEMENT_VERSION
    if resolved_version not in (1, 2):
        raise ValueError(f"unknown settlement version: {resolved_version}")
    claims = snapshot["claims"]
    answer_id = snapshot["answer_claim_id"]
    answer = claims[answer_id]
    transfers: list[Transfer] = [
        Transfer("PAYOUT_FINISHER", None, answer["author"], ALPHA, answer_id)
    ]

    _, kept = derivation_flows(snapshot)
    derivation_paid: dict[str, int] = {}
    for claim_id in sorted(kept):
        amount = kept[claim_id]
        if amount <= 0:
            continue
        derivation_paid[claim_id] = amount
        transfers.append(
            Transfer("PAYOUT_DERIVATION", None, claims[claim_id]["author"], amount, claim_id)
        )

    if resolved_version >= 2:
        steering_weights = _steering_weights_v2(snapshot, derivation_paid)
    else:
        productive_authors = {
            claims[claim_id]["author"]
            for claim_id, amount in derivation_paid.items()
            if amount > 0
        }
        productive_authors.add(answer["author"])
        steering_weights = _steering_weights_v1(snapshot, productive_authors)
    steering_paid = _split_weighted(GAMMA, steering_weights)
    if steering_paid:
        for claim_id in sorted(steering_paid):
            transfers.append(
                Transfer(
                    "PAYOUT_STEERING",
                    None,
                    claims[claim_id]["author"],
                    steering_paid[claim_id],
                    claim_id,
                )
            )
    else:
        transfers.append(Transfer("BURN", None, None, GAMMA, None))

    earned = {claim_id: derivation_paid.get(claim_id, 0) for claim_id in claims}
    for claim_id, amount in steering_paid.items():
        earned[claim_id] = earned.get(claim_id, 0) + amount
    for claim_id in sorted(claims):
        if earned.get(claim_id, 0) >= ADMISSION_FEE:
            transfers.append(
                Transfer("FEE_REFUND", "FEE_POOL", claims[claim_id]["author"], ADMISSION_FEE, claim_id)
            )

    bounty_total = sum(
        transfer.amount_mu
        for transfer in transfers
        if transfer.reason in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
    )
    assert bounty_total == ALPHA + BETA + GAMMA
    return transfers
