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
    steering_paid = sum(
        row["amount_µ"] for row in transfers if row["reason"] == "PAYOUT_STEERING"
    )
    steering_burned = sum(
        row["amount_µ"]
        for row in transfers
        if row["reason"] == "BURN" and row["from_pubkey"] == f"ESCROW:{task_id}"
    )
    steering_by_author: dict[str, int] = {}
    for row in transfers:
        if row["reason"] == "PAYOUT_STEERING":
            steering_by_author[row["to_pubkey"]] = (
                steering_by_author.get(row["to_pubkey"], 0) + row["amount_µ"]
            )
    lease_bonds_burned = sum(
        row["amount_µ"]
        for row in transfers
        if row["reason"] == "BURN"
        and row["from_pubkey"] is not None
        and row["from_pubkey"].startswith("BOND:")
    )
    pseudo = store.pseudo_balances()
    if any(balance < 0 for balance in pseudo.values()):
        raise RuntimeError("pseudo-account solvency violated")
    lines = [
        f"task {task_id}",
        f"closed_epoch {closed_epoch}",
        f"settled_epoch {store.epoch()}",
        f"bounty_paid_or_burned {bounty_paid_or_burned}",
        f"steering_paid {steering_paid}",
        f"steering_burned {steering_burned}",
        f"lease_bonds_burned {lease_bonds_burned}",
        f"mean_honest_delta {mean_honest}",
        f"mean_hoarder_delta {mean_hoarder}",
    ]
    for author in sorted(steering_by_author):
        lines.append(f"steering_paid_by_author {_short(author)} {steering_by_author[author]}")
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


@main.command(name="eval")
@click.option("--tasks", "tasks_dir", type=click.Path(path_type=Path), default=Path("corpus/tasks"))
@click.option("--corpus", "corpus_dir", type=click.Path(path_type=Path), default=None)
@click.option("--workers", "n_workers", type=int, default=4)
@click.option("--baseline/--no-baseline", default=True)
@click.option("--sweep", is_flag=True, default=False, help="Run the regime-study grid.")
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None)
def eval_command(
    tasks_dir: Path,
    corpus_dir: Path | None,
    n_workers: int,
    baseline: bool,
    sweep: bool,
    out_path: Path | None,
) -> None:
    """Run the protocol with LLM workers vs a single-agent baseline.

    Uses the real endpoint when VSCP_LLM=1, otherwise a deterministic fake
    LLM answering from the fixtures' gold facts. With --sweep, runs the
    document-length x worker-count regime grid and writes regime.json."""
    from legion.evaluate import format_report, format_sweep, run_eval, run_sweep

    if sweep:
        out = out_path or Path("regime.json")
        result = run_sweep(tasks_dir, corpus_dir=corpus_dir, out_path=out)
        click.echo(format_sweep(result))
        click.echo(f"regime grid written to {out}")
        return
    out = out_path or Path("report.json")
    report = run_eval(
        tasks_dir,
        corpus_dir=corpus_dir,
        n_workers=n_workers,
        baseline=baseline,
        out_path=out,
    )
    click.echo(format_report(report))
    click.echo(f"report written to {out}")


@main.command(name="run-cluster")
@click.option("--workers", "n_workers", type=int, default=4)
@click.option("--K", "--k", "K", type=int, default=3)
@click.option("--D", "--d", "D", type=int, default=2)
@click.option("--seed", type=int, default=11)
@click.option("--timeout", "timeout_s", type=float, default=60.0)
@click.option("--workdir", type=click.Path(path_type=Path), required=True)
def run_cluster_command(
    n_workers: int, K: int, D: int, seed: int, timeout_s: float, workdir: Path
) -> None:
    """Run N worker processes + 1 coordinator against one WAL ledger."""
    from legion.cluster import format_cluster_summary, run_cluster

    workdir.mkdir(parents=True, exist_ok=True)
    summary = run_cluster(
        workdir, n_workers=n_workers, K=K, D=D, seed=seed, timeout_s=timeout_s
    )
    click.echo(format_cluster_summary(summary))
    if not summary["settled"]:
        raise SystemExit(1)


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
@click.argument("path", type=click.Path(path_type=Path))
def audit(path: Path) -> None:
    """Light-client audit: re-derive every balance, settlement, and
    deterministic admission check from the immutable log."""
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from tools.lightclient import audit as run_audit

    result = run_audit(path)
    click.echo(result.summary())
    if not result.passed:
        for divergence in result.divergences[1:]:
            click.echo(f"  also: {divergence}")
        raise SystemExit(1)


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


if __name__ == "__main__":
    main()
