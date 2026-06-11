"""Agent-based research simulation of the knowledge market (§4A).

The v2.1 spec text for §4A was not available; this implements the Phase 3
restatement (world model, seven strategies, flat + Shapley settlement,
`run_many`) with design gaps filled by the simplest choices consistent with
the invariants, recorded in docs/DECISIONS.md. Floats throughout - this is the
research model, not the engine - and it must not import `legion` (enforced by
test) so it remains an independent cross-check.

World: K documents, each hiding 1 fact among D decoys. One oracle probe per
agent per epoch, uniform over candidates the agent has not ruled out. The
episode closes when an agent submits an ANSWER citing all K facts; settlement
is either the flat backward flow of `tools/sim/sim.py` (steering v1 or v2) or
a permutation-sampled Shapley split of the derivation pool over *apparent*
coverage - the naive attribution the flat design exists to replace, kept here
so its exploitability is measurable.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tools.sim.sim import (
    ALPHA_FRAC,
    BETA_FRAC,
    Episode,
    SimClaim,
    settle_episode,
)

HONEST = "HONEST"
HOARDER = "HOARDER"
SPAMMER = "SPAMMER"
RING_SYBIL = "RING_SYBIL"
RING_BENEF = "RING_BENEF"
POISONER = "POISONER"
HYBRID = "HYBRID"
STRATEGIES = (HONEST, HOARDER, SPAMMER, RING_SYBIL, RING_BENEF, POISONER, HYBRID)

ADMISSION_FEE = 0.005
PROBE_COST = 0.01
VERIFIER_LEAK = 0.4  # probability a plausible decoy-as-FACT slips past the semantic check
MAX_EPOCHS = 400
SHAPLEY_PERMUTATIONS = 120


@dataclass
class Agent:
    agent_id: str
    index: int
    strategy: str
    ruled_out: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    fees: float = 0.0
    probes: int = 0
    seen_fails: set[str] = field(default_factory=set)

    @property
    def publishes_fails(self) -> bool:
        return self.strategy in {HONEST, RING_BENEF}

    @property
    def races_answer(self) -> bool:
        return self.strategy in {HONEST, HOARDER, RING_BENEF, HYBRID}

    @property
    def probes_docs(self) -> bool:
        return self.strategy in {HONEST, RING_BENEF, HYBRID}

    @property
    def reads_fails(self) -> bool:
        return self.strategy in {HONEST, RING_BENEF}


@dataclass
class EpisodeResult:
    payouts: dict[str, float]  # agent_id -> gross settlement income
    fees: dict[str, float]
    probe_costs: dict[str, float]
    epochs: int
    claim_payouts: dict[str, float]  # claim_id -> derivation+steering income
    claims: dict[str, SimClaim]
    answer_id: str | None

    def net(self, agent_id: str) -> float:
        return (
            self.payouts.get(agent_id, 0.0)
            - self.fees.get(agent_id, 0.0)
            - self.probe_costs.get(agent_id, 0.0)
        )


def _doc_candidates(K: int, D: int) -> dict[int, list[str]]:
    return {
        doc: [f"d{doc}:fact"] + [f"d{doc}:decoy{j}" for j in range(D)] for doc in range(K)
    }


def run_episode(
    population: list[str],
    K: int = 4,
    D: int = 3,
    seed: int = 0,
    settlement: str = "flat",
    steering_version: int = 2,
    ring_pad_count: int = 1,
) -> EpisodeResult:
    rng = random.Random(seed)
    candidates = _doc_candidates(K, D)
    agents = [
        Agent(agent_id=f"a{i}:{strategy}", index=i, strategy=strategy)
        for i, strategy in enumerate(population)
    ]

    episode = Episode(answer_id="")
    fetches: list[tuple[str, str, int]] = []
    fact_claim_by_doc: dict[int, str] = {}
    published_bodies: dict[str, str] = {}  # claim_id -> candidate body
    fails_by_doc: dict[int, set[str]] = defaultdict(set)  # candidates ruled out publicly
    fail_claims_by_doc: dict[int, list[str]] = defaultdict(list)
    pending: list[tuple[Agent, str, int | None, str, tuple[str, ...], int]] = []
    claim_seq = 0
    answer_id: str | None = None
    epochs_used = MAX_EPOCHS
    ring_sybil_claims: list[str] = []

    def submit(agent: Agent, kind: str, doc: int | None, body: str, cites: tuple[str, ...] = (), epoch: int = 0) -> None:
        """Queue a submission; it becomes visible at epoch end (the sim's
        admission tick), so same-epoch duplicate publications are possible -
        exactly as in the engine."""
        agent.fees += ADMISSION_FEE
        pending.append((agent, kind, doc, body, cites, epoch))

    def admit_pending() -> None:
        nonlocal claim_seq, answer_id
        for agent, kind, doc, body, cites, epoch in pending:
            # Deterministic + semantic admission, collapsed to the answer key.
            if kind == "FACT":
                if not body.endswith(":fact") and rng.random() >= VERIFIER_LEAK:
                    continue  # caught by the verifier
            elif kind == "FAIL":
                if "decoy" not in body:
                    continue
            elif kind == "SPAM":
                continue  # malformed refs always fail the deterministic check
            claim_seq += 1
            claim_id = f"c{claim_seq:04d}"
            docs = frozenset({f"doc{doc}"}) if doc is not None else frozenset()
            episode.add_claim(
                SimClaim(
                    claim_id=claim_id,
                    author=agent.agent_id,
                    kind="FACT" if kind == "FACT" else ("FAIL" if kind == "FAIL" else "ANSWER"),
                    cites=cites,
                    docs=docs,
                    epoch=epoch,
                )
            )
            published_bodies[claim_id] = body
            if kind == "FACT" and body.endswith(":fact") and doc is not None and doc not in fact_claim_by_doc:
                fact_claim_by_doc[doc] = claim_id  # first publication is canonical
            if kind == "FAIL" and doc is not None:
                fails_by_doc[doc].add(body)
                fail_claims_by_doc[doc].append(claim_id)
                if agent.strategy == RING_SYBIL:
                    ring_sybil_claims.append(claim_id)
            if kind == "ANSWER" and answer_id is None:
                answer_id = claim_id  # first racer in queue order wins
        pending.clear()

    for epoch in range(MAX_EPOCHS):
        # Phase A: every agent that can see a complete fact set races the
        # answer this epoch; all racers pay the fee, the first one wins.
        if len(fact_claim_by_doc) == K:
            cite_ids = tuple(fact_claim_by_doc[doc] for doc in sorted(fact_claim_by_doc))
            for agent in agents:
                if agent.races_answer:
                    for claim_id in cite_ids:
                        fetches.append((agent.agent_id, claim_id, epoch))
                    submit(agent, "ANSWER", None, "answer", cites=cite_ids, epoch=epoch)
            admit_pending()
            epochs_used = epoch + 1
            break

        # Phase B: production.
        for agent in agents:
            if agent.strategy == SPAMMER:
                submit(agent, "SPAM", None, "junk", epoch=epoch)
            elif agent.strategy == POISONER:
                doc = rng.randrange(K)
                decoy = rng.choice([c for c in candidates[doc] if "decoy" in c])
                submit(agent, "FACT", doc, decoy, epoch=epoch)
            elif agent.strategy == RING_SYBIL:
                # Post-hoc trivial negative results: verified-true FAILs on
                # already-solved docs that steer nobody. Cheap to mint; only
                # the ring partner ever reads them.
                solved = sorted(fact_claim_by_doc)
                if solved:
                    doc = solved[rng.randrange(len(solved))]
                    decoys = [c for c in candidates[doc] if "decoy" in c]
                    submit(agent, "FAIL", doc, decoys[rng.randrange(len(decoys))], epoch=epoch)
            elif agent.probes_docs:
                open_docs = [doc for doc in range(K) if doc not in fact_claim_by_doc]
                if not open_docs:
                    continue
                doc = open_docs[agent.index % len(open_docs)]
                if agent.reads_fails:
                    # Metered fetch of FAILs relevant to the doc being worked.
                    for claim_id in fail_claims_by_doc[doc]:
                        claim = episode.claims[claim_id]
                        if claim_id not in agent.seen_fails and claim.author != agent.agent_id:
                            agent.seen_fails.add(claim_id)
                            fetches.append((agent.agent_id, claim_id, epoch))
                            agent.ruled_out[doc].add(published_bodies[claim_id])
                if agent.strategy == RING_BENEF:
                    # The collusion: fetch every sybil claim, relevant or not.
                    for claim_id in ring_sybil_claims:
                        if claim_id not in agent.seen_fails:
                            agent.seen_fails.add(claim_id)
                            fetches.append((agent.agent_id, claim_id, epoch))
                remaining = [c for c in candidates[doc] if c not in agent.ruled_out[doc]]
                if not remaining:
                    continue
                agent.probes += 1
                probe = remaining[rng.randrange(len(remaining))]
                agent.ruled_out[doc].add(probe)
                if probe.endswith(":fact"):
                    cites: tuple[str, ...] = ()
                    if agent.strategy == RING_BENEF and ring_sybil_claims:
                        cites = tuple(ring_sybil_claims[:ring_pad_count])
                    submit(agent, "FACT", doc, probe, cites=cites, epoch=epoch)
                elif agent.publishes_fails and probe not in fails_by_doc[doc]:
                    submit(agent, "FAIL", doc, probe, epoch=epoch)
        admit_pending()

    payouts: dict[str, float] = defaultdict(float)
    claim_payouts: dict[str, float] = {}
    if answer_id is not None:
        episode.answer_id = answer_id
        episode.fetches = fetches
        if settlement == "flat":
            for author, amount in settle_episode(episode, steering_version=steering_version).items():
                payouts[author] += amount
            claim_payouts = _flat_claim_payouts(episode, steering_version)
        elif settlement == "shapley":
            payouts_map, claim_payouts = _shapley_settlement(episode, rng)
            for author, amount in payouts_map.items():
                payouts[author] += amount
        else:
            raise ValueError(f"unknown settlement: {settlement}")

    return EpisodeResult(
        payouts=dict(payouts),
        fees={agent.agent_id: agent.fees for agent in agents},
        probe_costs={agent.agent_id: agent.probes * PROBE_COST for agent in agents},
        epochs=epochs_used,
        claim_payouts=claim_payouts,
        claims=dict(episode.claims),
        answer_id=answer_id,
    )


def _flat_claim_payouts(episode: Episode, steering_version: int) -> dict[str, float]:
    from tools.sim.sim import _derivation_flow, _steering_weights, GAMMA_FRAC, EPSILON

    flow = _derivation_flow(episode)
    weights = _steering_weights(episode, flow, steering_version)
    total_weight = sum(weights.values())
    out = {claim_id: amount for claim_id, amount in flow.items() if amount > EPSILON}
    if total_weight > EPSILON:
        for claim_id, weight in weights.items():
            if weight > EPSILON:
                out[claim_id] = out.get(claim_id, 0.0) + GAMMA_FRAC * weight / total_weight
    return out


def _shapley_settlement(
    episode: Episode, rng: random.Random
) -> tuple[dict[str, float], dict[str, float]]:
    """Naive Shapley over *apparent* document coverage by admitted FACT claims.

    v(S) = (#distinct docs apparently covered by S) / K. This is deliberately
    the exploitable attribution: an admitted poison FACT 'covers' its doc just
    as a genuine one does, so it earns a share - the property the flat design
    is measured against."""
    answer = episode.claims[episode.answer_id]
    fact_claims = [c for c in episode.claims.values() if c.kind == "FACT" and c.docs]
    all_docs = {doc for claim in fact_claims for doc in claim.docs}
    if not fact_claims or not all_docs:
        return {answer.author: ALPHA_FRAC + BETA_FRAC}, {}
    shapley: dict[str, float] = defaultdict(float)
    ids = [claim.claim_id for claim in fact_claims]
    for _ in range(SHAPLEY_PERMUTATIONS):
        rng.shuffle(ids)
        covered: set[str] = set()
        for claim_id in ids:
            docs = episode.claims[claim_id].docs
            gain = len(docs - covered) / len(all_docs)
            shapley[claim_id] += gain
            covered |= docs
    claim_payouts = {
        claim_id: BETA_FRAC * value / SHAPLEY_PERMUTATIONS for claim_id, value in shapley.items()
    }
    payouts: dict[str, float] = defaultdict(float)
    payouts[answer.author] += ALPHA_FRAC
    for claim_id, amount in claim_payouts.items():
        payouts[episode.claims[claim_id].author] += amount
    return dict(payouts), claim_payouts


def run_many(
    population: list[str],
    episodes: int = 30,
    K: int = 4,
    D: int = 3,
    seed: int = 0,
    settlement: str = "flat",
    steering_version: int = 2,
    ring_pad_count: int = 1,
) -> dict[str, Any]:
    """Mean net payoff per strategy, mean epochs, and total welfare."""
    by_strategy: dict[str, list[float]] = defaultdict(list)
    epochs: list[int] = []
    welfare = 0.0
    for episode_index in range(episodes):
        result = run_episode(
            population,
            K=K,
            D=D,
            seed=seed * 100_003 + episode_index,
            settlement=settlement,
            steering_version=steering_version,
            ring_pad_count=ring_pad_count,
        )
        epochs.append(result.epochs)
        for index, strategy in enumerate(population):
            agent_id = f"a{index}:{strategy}"
            net = result.net(agent_id)
            by_strategy[strategy].append(net)
            welfare += net
    return {
        "mean_net": {strategy: sum(v) / len(v) for strategy, v in by_strategy.items()},
        "mean_epochs": sum(epochs) / len(epochs),
        "welfare": welfare,
        "episodes": episodes,
    }
