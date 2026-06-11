"""Regression: `python -m legion.cli` used to import and exit 0 silently
(missing __main__ guard) - a silently-succeeding no-op command is the worst
failure mode, so module invocation gets its own test."""
from __future__ import annotations

import subprocess
import sys


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
            "--workers", "2",
            "--honest", "1",
            "--hoarders", "1",
            "--K", "1",
            "--D", "1",
            "--seed", "5",
            "--workdir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "bounty_paid_or_burned 1000000" in result.stdout
