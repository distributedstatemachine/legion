"""Research-sim settlement reference (stand-in vendor of `delm_market_sim`).

NOTE: the Phase 2 spec asks to vendor `delm_market_sim/sim.py`; that repo is
not available in this environment, so this module is an *independently written*
reference implementation of the same mechanism (recorded in docs/DECISIONS.md).
It deliberately differs from `legion.settlement` in every implementation
choice that does not change the math, so agreement between the two is a real
cross-check rather than a copy-paste tautology:

- float arithmetic throughout (the engine is integer-only),
- recursive depth-first flow propagation (the engine uses an explicit
  reverse-topological pass with integer largest-remainder splits),
- per-author accumulation (the engine emits per-claim transfers).

Payouts are normalized fractions of a bounty of 1.0; the equivalence adapter
converts to µ with round(x * 1_000_000).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

ALPHA_FRAC = 0.35
BETA_FRAC = 0.45
GAMMA_FRAC = 0.20
KEEP_FRACTION = 0.5
EPSILON = 1e-12


@dataclass(frozen=True)
class SimClaim:
    claim_id: str
    author: str
    kind: str = "FACT"
    cites: tuple[str, ...] = ()
    docs: frozenset[str] = frozenset()
    epoch: int = 0


@dataclass
class Episode:
    """An isomorphic episode whose claims/fetches are injected directly,
    bypassing the sim's discovery loop."""

    answer_id: str
    claims: dict[str, SimClaim] = field(default_factory=dict)
    fetches: list[tuple[str, str, int]] = field(default_factory=list)  # (reader, object, epoch)

    def add_claim(self, claim: SimClaim) -> None:
        self.claims[claim.claim_id] = claim


def _effective_docs(episode: Episode, claim: SimClaim) -> frozenset[str]:
    if claim.docs:
        return claim.docs
    docs: set[str] = set()
    for parent_id in claim.cites:
        parent = episode.claims.get(parent_id)
        if parent is not None:
            docs |= parent.docs  # one level only, no recursion
    return frozenset(docs)


def _derivation_flow(episode: Episode) -> dict[str, float]:
    """Recursive flow: each push keeps KEEP_FRACTION and forwards the rest
    equally to in-graph parents. Linear, so equivalent to accumulate-then-split."""
    flow: dict[str, float] = defaultdict(float)

    def push(node_id: str, amount: float) -> None:
        claim = episode.claims[node_id]
        parents = [p for p in claim.cites if p in episode.claims]
        if not parents:
            flow[node_id] += amount
            return
        forwarded = amount * (1.0 - KEEP_FRACTION)
        flow[node_id] += amount - forwarded
        share = forwarded / len(parents)
        for parent in parents:
            push(parent, share)

    push(episode.answer_id, BETA_FRAC)
    return dict(flow)


def _steering_weights(episode: Episode, flow: dict[str, float], version: int) -> dict[str, float]:
    answer = episode.claims[episode.answer_id]
    productive_authors = {
        episode.claims[node].author for node, amount in flow.items() if amount > EPSILON
    }
    productive_authors.add(answer.author)
    steerable = [c for c in episode.claims.values() if c.kind in {"FAIL", "CONSTRAINT"}]

    if version == 1:
        readers: dict[str, set[str]] = defaultdict(set)
        for reader, object_id, _epoch in episode.fetches:
            readers[object_id].add(reader)
        return {c.claim_id: float(len(readers[c.claim_id] & productive_authors)) for c in steerable}

    # v2: reader-normalized, relevance-scoped (mirrors the Phase 2 spec §2.3).
    productive_claims: dict[str, list[SimClaim]] = defaultdict(list)
    for node, amount in flow.items():
        if amount > EPSILON:
            productive_claims[episode.claims[node].author].append(episode.claims[node])
    if answer not in productive_claims[answer.author]:
        productive_claims[answer.author].append(answer)

    first_fetch: dict[tuple[str, str], int] = {}
    for reader, object_id, epoch in episode.fetches:
        key = (reader, object_id)
        if key not in first_fetch or epoch < first_fetch[key]:
            first_fetch[key] = epoch

    weights: dict[str, float] = {c.claim_id: 0.0 for c in steerable}
    for reader, authored in productive_claims.items():
        eligible: list[str] = []
        for claim in steerable:
            if claim.author == reader:
                continue
            epoch = first_fetch.get((reader, claim.claim_id))
            if epoch is None:
                continue
            claim_docs = _effective_docs(episode, claim)
            if not claim_docs:
                continue
            if any(
                epoch <= produced.epoch and claim_docs & _effective_docs(episode, produced)
                for produced in authored
            ):
                eligible.append(claim.claim_id)
        for claim_id in eligible:
            weights[claim_id] += 1.0 / len(eligible)
    return weights


def settle_episode(episode: Episode, steering_version: int = 1) -> dict[str, float]:
    """Per-author payout fractions of a 1.0 bounty. Unallocated GAMMA burns."""
    payouts: dict[str, float] = defaultdict(float)
    answer = episode.claims[episode.answer_id]
    payouts[answer.author] += ALPHA_FRAC

    flow = _derivation_flow(episode)
    for node, amount in flow.items():
        if amount > EPSILON:
            payouts[episode.claims[node].author] += amount

    weights = _steering_weights(episode, flow, steering_version)
    total_weight = sum(weights.values())
    if total_weight > EPSILON:
        for claim_id, weight in weights.items():
            if weight > EPSILON:
                payouts[episode.claims[claim_id].author] += GAMMA_FRAC * weight / total_weight
    return dict(payouts)
