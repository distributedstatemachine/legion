"""Light-client re-derivation: a separate process replays the ledger and must
agree to the µ; a tampered ledger must fail with a clear diff. The
cross-process requirement is the point - re-derivation cannot depend on any
in-memory state of the writer."""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from legion.cli import run_demo

ROOT = Path(__file__).resolve().parents[1]


def _run_audit(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "legion.cli", "audit", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )


def test_audit_passes_on_fresh_ledger_in_separate_process(tmp_path):
    workdir = tmp_path / "demo"
    workdir.mkdir()
    run_demo(workers=6, honest=4, hoarders=2, K=6, D=3, seed=42, root=workdir)

    result = _run_audit(workdir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "identities" in result.stdout and "re-settled" in result.stdout
    # The trust boundary is stated, not hidden.
    assert "verifier" in result.stdout


def test_audit_fails_on_tampered_payout(tmp_path):
    workdir = tmp_path / "demo"
    workdir.mkdir()
    run_demo(workers=6, honest=4, hoarders=2, K=6, D=3, seed=42, root=workdir)

    tampered = tmp_path / "tampered"
    shutil.copytree(workdir, tampered)
    conn = sqlite3.connect(tampered / "ledger.db")
    # An attacker with file access can drop the append-only trigger; the audit
    # is what catches the rewrite afterwards.
    conn.execute("DROP TRIGGER transfers_no_update")
    conn.execute(
        'UPDATE transfers SET "amount_µ" = "amount_µ" + 1000 WHERE id = '
        "(SELECT id FROM transfers WHERE reason = 'PAYOUT_FINISHER' LIMIT 1)"
    )
    conn.commit()
    conn.close()

    result = _run_audit(tampered)
    assert result.returncode == 1
    assert "FAIL" in result.stdout
    # The first divergence names what diverged.
    assert "balance" in result.stdout or "settlement mismatch" in result.stdout
