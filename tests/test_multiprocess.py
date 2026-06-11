"""Multi-process operation: N worker processes + 1 coordinator on one WAL
ledger. Validates the Phase 1 claim that only lease acquisition needs
serialization while claim submissions are commutative."""
from __future__ import annotations

import multiprocessing
import time

import pytest

from legion import crypto, tasks
from legion.cluster import run_cluster
from legion.store import Store


@pytest.mark.slow
def test_cluster_settles_with_four_worker_processes(tmp_path):
    summary = run_cluster(tmp_path, n_workers=4, K=3, D=2, seed=11, timeout_s=90.0)
    assert summary["settled"] is True
    # Conservation to the µ: the escrow was exactly drained.
    assert summary["bounty_paid_or_burned"] == 1_000_000
    assert all(balance >= 0 for balance in summary["balances"].values())
    # Total BOND in == out: no bond stranded in any pseudo account.
    bonds = {
        account: balance
        for account, balance in summary["pseudo_balances"].items()
        if account.startswith("BOND:")
    }
    assert all(balance == 0 for balance in bonds.values())

    # No subtask was completed by two authors (no double-lease).
    store = Store(tmp_path)
    rows = store.conn.execute(
        "SELECT subtask_id, COUNT(DISTINCT author) AS n FROM claims "
        "WHERE subtask_id IS NOT NULL AND status = 'ADMITTED' GROUP BY subtask_id"
    ).fetchall()
    assert rows, "expected admitted subtask claims"
    assert all(row["n"] == 1 for row in rows)
    subtasks = store.conn.execute("SELECT status FROM subtasks").fetchall()
    assert all(row["status"] in {"DONE", "PENDING"} for row in subtasks)
    store.close()


def _lease_contender(root: str, task_id: str, worker_seed: str, barrier, queue) -> None:
    store = Store(root)
    keypair = crypto.keypair_from_seed(worker_seed)
    store.create_identity(keypair.pubkey, 100_000)
    barrier.wait(timeout=30)
    lease = tasks.lease_available_subtask(store, task_id, keypair.pubkey)
    queue.put((worker_seed, None if lease is None else lease["subtask_id"]))
    store.close()


def test_lease_contention_exactly_one_winner(tmp_path):
    store = Store(tmp_path)
    sponsor = crypto.keypair_from_seed("contention-sponsor")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = tasks.create_fact_chain_task(store, K=1, D=1, seed=3, sponsor_pubkey=sponsor.pubkey)
    store.close()

    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    contenders = [
        ctx.Process(
            target=_lease_contender,
            args=(str(tmp_path), task_id, f"contender-{i}", barrier, queue),
        )
        for i in range(2)
    ]
    for process in contenders:
        process.start()
    results = [queue.get(timeout=60) for _ in range(2)]
    for process in contenders:
        process.join(timeout=30)

    # Only one fact subtask was available: exactly one contender acquired it,
    # the other backed off cleanly with None.
    won = [r for r in results if r[1] is not None]
    lost = [r for r in results if r[1] is None]
    assert len(won) == 1 and len(lost) == 1

    store = Store(tmp_path)
    row = store.conn.execute(
        "SELECT lease_holder FROM subtasks WHERE subtask_id = ?", (f"{task_id}:fact:0",)
    ).fetchone()
    assert row["lease_holder"] is not None
    # Exactly one lease bond was taken.
    bonds = store.conn.execute(
        "SELECT COUNT(*) AS n FROM transfers WHERE reason = 'BOND'"
    ).fetchone()
    assert bonds["n"] == 1
    store.close()
