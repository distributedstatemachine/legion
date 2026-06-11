"""Named experiments backing the eight statistical goldens (§4A.6).

Each function runs a seeded population and returns the raw measured numbers;
`tests/test_sim_goldens.py` pins the ordering assertions (never weakened) and
the numeric bands (measured values recorded in docs/DECISIONS.md)."""
from __future__ import annotations

from typing import Any

from tools.sim.model import (
    HOARDER,
    HONEST,
    HYBRID,
    POISONER,
    RING_BENEF,
    RING_SYBIL,
    SPAMMER,
    run_episode,
    run_many,
)

EPISODES = 30


def exp_honest_vs_hoarder(seed: int = 1) -> dict[str, Any]:
    stats = run_many([HONEST] * 4 + [HOARDER] * 2, episodes=EPISODES, seed=seed)
    return {"honest": stats["mean_net"][HONEST], "hoarder": stats["mean_net"][HOARDER]}


def exp_lone_poisoner(seed: int = 2) -> dict[str, Any]:
    population = [HONEST] * 4 + [POISONER]
    flat = run_many(population, episodes=EPISODES, seed=seed, settlement="flat")
    shapley = run_many(population, episodes=EPISODES, seed=seed, settlement="shapley")
    return {
        "poisoner_flat": flat["mean_net"][POISONER],
        "poisoner_shapley": shapley["mean_net"][POISONER],
        "honest_flat": flat["mean_net"][HONEST],
    }


def exp_ring_steering(seed: int = 3) -> dict[str, Any]:
    """Steering captured by the sybil's post-hoc FAILs (the parasite income),
    isolated from the benefactor's honest-side earnings: ring_pad_count=0 so
    no derivation flows into sybil claims and their payouts are pure gamma."""
    population = [HONEST] * 3 + [RING_SYBIL, RING_BENEF]
    sybil_id = f"a3:{RING_SYBIL}"
    captures = {1: 0.0, 2: 0.0}
    for version in (1, 2):
        for episode_index in range(EPISODES):
            result = run_episode(
                population,
                seed=seed * 100_003 + episode_index,
                steering_version=version,
                ring_pad_count=0,
            )
            captures[version] += sum(
                amount
                for claim_id, amount in result.claim_payouts.items()
                if result.claims[claim_id].author == sybil_id
            )
    return {"sybil_steering_v1": captures[1], "sybil_steering_v2": captures[2]}


def exp_spammer(seed: int = 4) -> dict[str, Any]:
    population = [HONEST] * 4 + [SPAMMER]
    flat = run_many(population, episodes=EPISODES, seed=seed, settlement="flat")
    shapley = run_many(population, episodes=EPISODES, seed=seed, settlement="shapley")
    return {
        "spammer_flat": flat["mean_net"][SPAMMER],
        "spammer_shapley": shapley["mean_net"][SPAMMER],
    }


def exp_all_hybrid_welfare(seed: int = 5) -> dict[str, Any]:
    # D=8: a deep decoy space makes unshared negative results expensive.
    honest = run_many([HONEST] * 6, episodes=EPISODES, seed=seed, K=4, D=8)
    hybrid = run_many([HYBRID] * 6, episodes=EPISODES, seed=seed, K=4, D=8)
    return {
        "welfare_honest": honest["welfare"],
        "welfare_hybrid": hybrid["welfare"],
        "epochs_honest": honest["mean_epochs"],
        "epochs_hybrid": hybrid["mean_epochs"],
    }


def exp_duplicate_facts(seed: int = 6) -> dict[str, Any]:
    """First vs duplicate publication of the same fact under flat flow.

    With more honest agents than documents, two agents probe the same doc; the
    first admitted FACT is cited by the answer, the duplicate is not."""
    original_total = 0.0
    duplicate_total = 0.0
    episodes = 0
    for episode_index in range(EPISODES):
        result = run_episode([HONEST] * 6, K=2, D=3, seed=seed * 100_003 + episode_index)
        if result.answer_id is None:
            continue
        answer_cites = set(result.claims[result.answer_id].cites)
        fact_claims = [c for c in result.claims.values() if c.kind == "FACT"]
        by_doc: dict[frozenset, list] = {}
        for claim in sorted(fact_claims, key=lambda c: (c.epoch, c.claim_id)):
            by_doc.setdefault(claim.docs, []).append(claim)
        found_pair = False
        for claims_for_doc in by_doc.values():
            if len(claims_for_doc) < 2:
                continue
            cited = [c for c in claims_for_doc if c.claim_id in answer_cites]
            uncited = [c for c in claims_for_doc if c.claim_id not in answer_cites]
            if cited and uncited:
                original_total += result.claim_payouts.get(cited[0].claim_id, 0.0)
                duplicate_total += result.claim_payouts.get(uncited[0].claim_id, 0.0)
                found_pair = True
        if found_pair:
            episodes += 1
    return {
        "original_total": original_total,
        "duplicate_total": duplicate_total,
        "episodes_with_duplicates": episodes,
    }


def exp_keep_fraction_padding(seed: int = 7) -> dict[str, Any]:
    """Adding an *extra* padded citation (1 -> 2 sybil cites) must not change
    the citer's own derivation income: the statistical form of keep-fraction
    invariance. (0 -> 1 parents legitimately halves the kept amount, so the
    invariant is stated over already-citing claims, as in the engine test.)"""
    population = [HONEST] * 3 + [RING_SYBIL, RING_BENEF]
    one_pad_total = 0.0
    two_pad_total = 0.0
    benef_id = f"a4:{RING_BENEF}"
    for episode_index in range(EPISODES):
        episode_seed = seed * 100_003 + episode_index
        for pad_count in (1, 2):
            result = run_episode(population, seed=episode_seed, ring_pad_count=pad_count)
            own = sum(
                amount
                for claim_id, amount in result.claim_payouts.items()
                if result.claims[claim_id].author == benef_id
                and result.claims[claim_id].kind == "FACT"
                and result.claims[claim_id].cites  # only already-citing claims
            )
            if pad_count == 1:
                one_pad_total += own
            else:
                two_pad_total += own
    return {"one_pad": one_pad_total, "two_pad": two_pad_total}


def exp_seed_determinism(seed: int = 8) -> dict[str, Any]:
    population = [HONEST] * 3 + [HOARDER, POISONER]
    first = run_many(population, episodes=10, seed=seed)
    second = run_many(population, episodes=10, seed=seed)
    return {"first": first, "second": second}
