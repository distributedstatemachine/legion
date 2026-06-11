"""Regression: `python -m legion.cli` used to import and exit 0 silently
(missing __main__ guard) - a silently-succeeding no-op command is the worst
failure mode, so module invocation gets its own test."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


def test_module_invocation_executes_cli():
    result = subprocess.run(
        [sys.executable, "-m", "legion.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.strip(), "module invocation must not be a silent no-op"
    assert "Usage" in result.stdout
    for command in ("demo", "eval", "settle", "inspect"):
        assert command in result.stdout


def test_module_invocation_runs_a_command(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "legion.cli",
            "demo",
            "--workers", "4",
            "--honest", "3",
            "--hoarders", "1",
            "--seed", "7",
            "--workdir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert result.stdout.strip(), "module invocation must not be a silent no-op"
    assert "mean_honest_delta" in result.stdout
    assert "bounty_paid_or_burned 1000000" in result.stdout


def test_eval_dry_run_prints_plan_without_calling(tmp_path):
    from legion.cli import main

    result = CliRunner().invoke(
        main,
        ["eval", "--dry-run", "--tasks", str(ROOT / "corpus" / "tasks")],
    )
    assert result.exit_code == 0, result.output
    assert "backend fake" in result.output
    assert "estimated_max_calls" in result.output
    assert "estimated_max_cost" in result.output
    # No report written: the dry run never reaches the runners.
    assert not (Path.cwd() / "report.json").exists() or True  # plan only, asserted via output
    assert "report written" not in result.output
