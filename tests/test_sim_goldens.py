"""The eight statistical goldens for the §4A research simulation.

Conformance rule: the ordering assertions (>/< clauses) may never be weakened.
Numeric bands may be widened only with the measured value recorded in
docs/DECISIONS.md. Measured values at the pinned seeds are noted inline and in
DECISIONS.md. `tools/sim/` must not import `legion` (guarded below)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sim import experiments  # noqa: E402


def test_golden_1_honest_beats_racing_hoarder():
    stats = experiments.exp_honest_vs_hoarder()
    # measured: honest 0.2005, hoarder -0.005 (hoarder races and forfeits fees)
    assert stats["honest"] > stats["hoarder"]
    assert 0.10 < stats["honest"] < 0.35
    assert -0.05 < stats["hoarder"] < 0.0


def test_golden_2_lone_poisoner_profitable_under_shapley_not_flat():
    stats = experiments.exp_lone_poisoner()
    # measured: flat -0.017, shapley +0.0389 - naive coverage attribution pays
    # admitted poison; flat backward flow from a verified answer does not.
    assert stats["poisoner_flat"] < 0.0
    assert stats["poisoner_shapley"] > 0.0
    assert stats["honest_flat"] > stats["poisoner_flat"]
    assert -0.10 < stats["poisoner_flat"] < 0.0
    assert 0.0 < stats["poisoner_shapley"] < 0.15


def test_golden_3_ring_steering_capture_collapses_under_v2():
    stats = experiments.exp_ring_steering()
    # measured: v1 0.2067, v2 0.0 - the partner's wholesale fetching mints
    # weight under raw readership and nothing under relevance scoping.
    assert stats["sybil_steering_v2"] < stats["sybil_steering_v1"]
    assert stats["sybil_steering_v1"] > 0.05
    assert stats["sybil_steering_v2"] <= 0.01


def test_golden_4_spammer_always_loses():
    stats = experiments.exp_spammer()
    # measured: -0.0158 under both settlements (deterministic checks catch spam)
    assert stats["spammer_flat"] < 0.0
    assert stats["spammer_shapley"] < 0.0
    assert -0.06 < stats["spammer_flat"]
    assert -0.06 < stats["spammer_shapley"]


def test_golden_5_all_hybrid_welfare_at_most_three_quarters_of_all_honest():
    stats = experiments.exp_all_hybrid_welfare()
    # measured: hybrid 12.19 vs honest 16.86 (ratio 0.723) - withholding FAILs
    # burns gamma and forces every agent to re-pay the decoy search.
    assert stats["welfare_honest"] > 0.0
    assert stats["welfare_hybrid"] <= 0.75 * stats["welfare_honest"]
    assert stats["epochs_hybrid"] > stats["epochs_honest"]


def test_golden_6_duplicate_facts_earn_under_ten_percent():
    stats = experiments.exp_duplicate_facts()
    # measured: originals 2.925, duplicates 0.0 across 22 duplicate episodes.
    assert stats["episodes_with_duplicates"] >= 5
    assert stats["original_total"] > 0.0
    assert stats["duplicate_total"] < 0.10 * stats["original_total"]


def test_golden_7_keep_fraction_invariance_statistical():
    stats = experiments.exp_keep_fraction_padding()
    # measured: 0.1125 == 0.1125 - an extra padded cite never changes the
    # citer's own kept derivation income.
    assert abs(stats["one_pad"] - stats["two_pad"]) <= 1e-9
    assert stats["one_pad"] > 0.0


def test_golden_8_seed_determinism():
    stats = experiments.exp_seed_determinism()
    assert stats["first"] == stats["second"]


def test_sim_package_does_not_import_legion():
    for path in (ROOT / "tools" / "sim").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import legion" not in source and "from legion" not in source, path.name
