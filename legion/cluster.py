"""Multi-process operation: N worker processes + 1 coordinator, one WAL ledger.

Concurrency discipline (see docs/DECISIONS.md): the coordinator is the sole
writer for admission, epoch advance, close, and settlement; workers only ever
lease/fetch/submit. Lease acquisition is an atomic BEGIN IMMEDIATE +
conditional UPDATE (legion.tasks._try_acquire_lease) so exactly one worker
wins a contended subtask; claim submissions are independent inserts; all
connections set busy_timeout=5000ms and retry bounded on SQLITE_BUSY.
"""
from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from legion import crypto, tasks
from legion.admission import MockVerifier
from legion.coordinator import Coordinator
from legion.store import Store
from legion.workers.scripted import FactOracle, ScriptedWorker

TICK_SECONDS = 0.03
WORKER_POLL_SECONDS = 0.01
WORKER_GRANT = 200_000


def create_cluster_task(root: str | Path, K: int, D: int, seed: int) -> str:
    store = Store(root)
    sponsor = crypto.keypair_from_seed(f"cluster-sponsor:{seed}")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = tasks.create_fact_chain_task(store, K, D, seed, sponsor.pubkey)
    store.close()
    return task_id


def worker_main(root: str, task_id: str, name: str, seed: int, timeout_s: float) -> None:
    store = Store(root)
    answer_key = store.answer_key(task_id)
    worker = ScriptedWorker.create(name, "honest", FactOracle(answer_key), seed)
    store.create_identity(worker.pubkey, WORKER_GRANT)
    deadline = time.monotonic() + timeout_s
    last_epoch = -1
    try:
        while time.monotonic() < deadline:
            task = store.task_row(task_id)
            if task["settlement_applied"]:
                break
            epoch = store.epoch()
            if epoch == last_epoch:
                time.sleep(WORKER_POLL_SECONDS)
                continue
            last_epoch = epoch
            try:
                worker.step(store, task_id)
            except ValueError:
                # e.g. "task ledger is frozen": the task closed between the
                # status check and the submit - benign race, stand down.
                time.sleep(WORKER_POLL_SECONDS)
    finally:
        store.close()


def coordinator_main(root: str, task_id: str, timeout_s: float) -> None:
    store = Store(root)
    coordinator = Coordinator(store, MockVerifier(store.answer_key(task_id)))
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            coordinator.tick()
            if store.task_row(task_id)["settlement_applied"]:
                break
            time.sleep(TICK_SECONDS)
    finally:
        store.close()


def run_cluster(
    root: str | Path,
    n_workers: int = 4,
    K: int = 3,
    D: int = 2,
    seed: int = 11,
    timeout_s: float = 60.0,
) -> dict:
    """Spawn the coordinator and N worker subprocesses; block until the task
    settles (or timeout); return a summary for printing."""
    root = str(root)
    task_id = create_cluster_task(root, K=K, D=D, seed=seed)
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(
            target=worker_main,
            args=(root, task_id, f"clusterworker{i}", seed + i + 1, timeout_s),
        )
        for i in range(n_workers)
    ]
    coordinator = ctx.Process(target=coordinator_main, args=(root, task_id, timeout_s))
    for process in workers:
        process.start()
    coordinator.start()
    coordinator.join(timeout=timeout_s + 10)
    for process in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
    if coordinator.is_alive():
        coordinator.terminate()

    store = Store(root)
    task = store.task_row(task_id)
    transfers = store.transfer_rows()
    bounty_paid_or_burned = sum(
        row["amount_µ"]
        for row in transfers
        if row["reason"] in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
        and (row["from_pubkey"] or "").startswith("ESCROW:")
    )
    summary = {
        "task_id": task_id,
        "settled": bool(task["settlement_applied"]),
        "closed_epoch": task["closed_epoch"],
        "epoch": store.epoch(),
        "bounty_paid_or_burned": bounty_paid_or_burned,
        "balances": store.balances(),
        "pseudo_balances": store.pseudo_balances(),
    }
    store.close()
    return summary


def format_cluster_summary(summary: dict) -> str:
    lines = [
        f"task {summary['task_id']}",
        f"settled {summary['settled']}",
        f"closed_epoch {summary['closed_epoch']}",
        f"final_epoch {summary['epoch']}",
        f"bounty_paid_or_burned {summary['bounty_paid_or_burned']}",
    ]
    for pubkey, balance in sorted(summary["balances"].items()):
        lines.append(f"identity {pubkey[:10]} balance={balance}")
    return "\n".join(lines)
