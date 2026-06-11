"""Settlement-equivalence harness: the integer engine and the float research
sim implement the same mechanism independently; they must agree to the µ
(within integer-rounding slack) on every shared scenario."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legion.cli import run_demo  # noqa: E402
from tools.equivalence import load_scenario, settle_legion, settle_sim  # noqa: E402

SCENARIO_DIR = Path(__file__).with_name("scenarios")
SCENARIOS = sorted(SCENARIO_DIR.glob("*.json"))


def _assert_totals_agree(scenario, version):
    legion_totals = settle_legion(scenario, version=version)
    sim_totals = settle_sim(scenario, version=version)
    slack = len(scenario["claims"])  # integer-rounding slack only
    authors = set(legion_totals) | set(sim_totals)
    for author in sorted(authors):
        a = legion_totals.get(author, 0)
        b = sim_totals.get(author, 0)
        assert abs(a - b) <= slack, (
            f"{scenario['name']} v{version} author {author}: legion={a} sim={b}"
        )


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
@pytest.mark.parametrize("version", [1, 2])
def test_engines_agree_on_scenario(path, version):
    assert len(SCENARIOS) >= 6
    _assert_totals_agree(load_scenario(path), version)


def test_ring_padding_reproduces_keep_fraction_numbers():
    # The keep-fraction invariance numbers from the settlement test, via the harness.
    scenario = load_scenario(SCENARIO_DIR / "ring_padding.json")
    totals = settle_legion(scenario, version=1)
    # answer keeps 225_000 (+ ALPHA); n keeps 112_500; p0/p1 get 56_250 each.
    assert totals["finisher"] == 350_000 + 225_000
    assert totals["node"] == 112_500
    assert totals["root0"] == 56_250
    assert totals["root1"] == 56_250


def test_ring_capture_v2_strictly_below_v1_via_harness():
    scenario = load_scenario(SCENARIO_DIR / "ring_steering_capture.json")
    ring_v1 = settle_legion(scenario, version=1).get("Z", 0)
    ring_v2 = settle_legion(scenario, version=2).get("Z", 0)
    assert 0 < ring_v2 < ring_v1
    sim_ring_v2 = settle_sim(scenario, version=2).get("Z", 0)
    assert abs(ring_v2 - sim_ring_v2) <= len(scenario["claims"])


@pytest.mark.parametrize("seed", [42, 7, 11])
def test_ordering_golden_honest_beats_racing_hoarder(tmp_path, seed):
    # run_demo itself raises unless mean(honest) > mean(racing hoarder); we
    # additionally pin that the hoarders' mean is non-positive at every seed.
    output = run_demo(
        workers=6, honest=4, hoarders=2, K=6, D=3, seed=seed, root=tmp_path / str(seed)
    )
    hoarder_line = next(
        line for line in output.splitlines() if line.startswith("mean_hoarder_delta ")
    )
    honest_line = next(
        line for line in output.splitlines() if line.startswith("mean_honest_delta ")
    )
    assert int(honest_line.split()[1]) > int(hoarder_line.split()[1])
    assert int(hoarder_line.split()[1]) <= 0
