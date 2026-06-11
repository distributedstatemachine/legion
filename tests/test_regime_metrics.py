"""Regime-study metrics on the long fixture, fake-LLM path (§3.4).

No assertion on crossover direction - that's a finding, not an invariant."""
from __future__ import annotations

import json
from pathlib import Path

from legion.evaluate import run_eval, run_sweep

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASKS = CORPUS / "tasks"
LONG_FIXTURE = TASKS / "long_deep_archive.json"


def test_long_fixture_metrics_fan_out(tmp_path):
    report = run_eval(
        TASKS,
        corpus_dir=CORPUS,
        n_workers=8,
        baseline=True,
        out_path=tmp_path / "report.json",
        fixture_paths=[LONG_FIXTURE],
    )
    (entry,) = report["tasks"]
    protocol = entry["protocol"]
    assert protocol["solved"] is True
    # The multi-doc fan-out the Phase 2 review predicted: steering eligibility
    # is no longer the finisher alone.
    assert protocol["distinct_eligible_steering_readers"] >= 2
    assert protocol["redundant_work_avoided"] >= 0
    assert protocol["peak_parallel_workers"] >= 2
    assert protocol["verifier_calls"] > 0
    assert report["cost_ratio"] is not None and report["cost_ratio"] > 0
    assert report["char_ratio"] is not None and report["char_ratio"] > 0


def test_sweep_writes_regime_grid(tmp_path):
    out = tmp_path / "regime.json"
    sweep = run_sweep(TASKS, corpus_dir=CORPUS, out_path=out, workers_grid=(1, 4))
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == sweep
    # Two document classes x two worker counts.
    assert len(sweep["cells"]) == 4
    for cell in sweep["cells"]:
        assert cell["solved"] is True
        assert cell["cost_ratio"] > 0
        assert cell["redundant_work_avoided"] >= 0
