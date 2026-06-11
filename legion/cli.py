from __future__ import annotations

import json
import tempfile
from pathlib import Path

import click

from legion import crypto, tasks
from legion.admission import MockVerifier
from legion.coordinator import Coordinator
from legion.store import Store
from legion.workers.scripted import FactOracle, ScriptedWorker


def _short(pubkey: str) -> str:
    return pubkey[:10]


def run_demo(
    *,
    workers: int,
    honest: int,
    hoarders: int,
    K: int,
    D: int,
    seed: int,
    root: str | Path,
) -> str:
    if honest + hoarders != workers:
        raise ValueError("honest + hoarders must equal workers")
    store = Store(root)
    sponsor = crypto.keypair_from_seed(f"sponsor:{seed}:{K}:{D}")
    store.create_identity(sponsor.pubkey, 2_000_000)
    task_id = tasks.create_fact_chain_task(store, K, D, seed, sponsor.pubkey)
    answer_key = store.answer_key(task_id)
    assert answer_key is not None
    oracle = FactOracle(answer_key)
    worker_objs: list[ScriptedWorker] = []
    initial: dict[str, int] = {}
    for index in range(workers):
        role = "honest" if index < honest else "hoarder"
        worker = ScriptedWorker.create(f"worker{index}", role, oracle, seed + index + 1)
        store.create_identity(worker.pubkey, 200_000)
        initial[worker.pubkey] = store.balance(worker.pubkey)
        worker_objs.append(worker)
    coordinator = Coordinator(store, MockVerifier(answer_key))
    closed_epoch = None
    for _ in range(200):
        for worker in worker_objs:
            worker.step(store, task_id)
        coordinator.tick()
        task = store.task_row(task_id)
        if task["closed_epoch"] is not None and closed_epoch is None:
            closed_epoch = int(task["closed_epoch"])
        if task["settlement_applied"]:
            break
    else:
        raise RuntimeError("demo did not settle within 200 epochs")
    task = store.task_row(task_id)
    if task["settlement_applied"] != 1 or closed_epoch is None or closed_epoch >= 200:
        raise RuntimeError("demo failed to close and settle")
    final = store.balances()
    honest_deltas = [
        final[worker.pubkey] - initial[worker.pubkey]
        for worker in worker_objs
        if worker.role == "honest"
    ]
    hoarder_deltas = [
        final[worker.pubkey] - initial[worker.pubkey]
        for worker in worker_objs
        if worker.role == "hoarder"
    ]
    mean_honest = sum(honest_deltas) // len(honest_deltas)
    mean_hoarder = sum(hoarder_deltas) // len(hoarder_deltas)
    if mean_honest <= mean_hoarder:
        raise RuntimeError("honest workers did not outperform hoarders")
    transfers = store.transfer_rows()
    bounty_paid_or_burned = sum(
        row["amount_µ"]
        for row in transfers
        if row["reason"] in {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}
    )
    lines = [
        f"task {task_id}",
        f"closed_epoch {closed_epoch}",
        f"settled_epoch {store.epoch()}",
        f"bounty_paid_or_burned {bounty_paid_or_burned}",
        f"mean_honest_delta {mean_honest}",
        f"mean_hoarder_delta {mean_hoarder}",
    ]
    for worker in worker_objs:
        delta = final[worker.pubkey] - initial[worker.pubkey]
        lines.append(
            f"identity {_short(worker.pubkey)} role={worker.role} initial={initial[worker.pubkey]} "
            f"final={final[worker.pubkey]} delta={delta}"
        )
    store.close()
    return "\n".join(lines) + "\n"


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--workers", type=int, default=6)
@click.option("--honest", type=int, default=4)
@click.option("--hoarders", type=int, default=2)
@click.option("--K", "--k", "K", type=int, default=6)
@click.option("--D", "--d", "D", type=int, default=3)
@click.option("--seed", type=int, default=42)
@click.option("--workdir", type=click.Path(path_type=Path), default=None)
def demo(workers: int, honest: int, hoarders: int, K: int, D: int, seed: int, workdir: Path | None) -> None:
    if workdir is None:
        with tempfile.TemporaryDirectory() as tmp:
            click.echo(run_demo(workers=workers, honest=honest, hoarders=hoarders, K=K, D=D, seed=seed, root=tmp), nl=False)
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        click.echo(run_demo(workers=workers, honest=honest, hoarders=hoarders, K=K, D=D, seed=seed, root=workdir), nl=False)


@main.command()
@click.argument("db_root", type=click.Path(path_type=Path))
@click.argument("task_id")
def settle(db_root: Path, task_id: str) -> None:
    from legion import settlement

    store = Store(db_root)
    transfers = settlement.settle(store.snapshot(task_id))
    click.echo(json.dumps([transfer.to_dict() for transfer in transfers], indent=2, sort_keys=True))
    store.close()


@main.command()
@click.argument("db_root", type=click.Path(path_type=Path))
def inspect(db_root: Path) -> None:
    store = Store(db_root)
    tasks_rows = [
        dict(row)
        for row in store.conn.execute(
            "SELECT task_id, status, closed_epoch, settlement_epoch, settlement_applied FROM tasks ORDER BY task_id"
        ).fetchall()
    ]
    click.echo(
        json.dumps(
            {"epoch": store.epoch(), "balances": store.balances(), "tasks": tasks_rows},
            indent=2,
            sort_keys=True,
        )
    )
    store.close()
