"""`legion eval`: protocol vs single-agent baseline on realdoc fixtures.

With `VSCP_LLM=1` the run uses the real OpenAI-compatible endpoint; otherwise
a deterministic fake `complete` stub answers from the fixture's gold facts so
the whole path (workers, hardened verifier, settlement, report) runs in CI
with no network. The deliverable is the measurement, not a winner.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from legion import crypto, settlement
from legion.admission import LLMVerifier
from legion.coordinator import Coordinator
from legion.store import Store
from legion.tasks_realdoc import load_fixture, make_realdoc_task
from legion.workers.llm import LLMWorker, openai_complete

MAX_EPOCHS = 100
WORKER_GRANT = 200_000
SPONSOR_GRANT = 2_000_000


class CountingComplete:
    def __init__(self, inner: Callable[[str], str]) -> None:
        self.inner = inner
        self.calls = 0
        self.prompt_chars = 0  # context volume actually shipped to the model

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompt_chars += len(prompt)
        return self.inner(prompt)


def _span0_from_verifier_prompt(prompt: str) -> str:
    marker = "SPAN 0\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n</data>", start)
    return prompt[start:end]


def make_fake_complete(fixture: dict[str, Any], corpus_dir: Path) -> Callable[[str], str]:
    """Deterministic stub answering from the gold facts (CI path, no network)."""
    docs = {
        name: (corpus_dir / name).read_text(encoding="utf-8")
        for name in fixture["documents"]
    }
    gold_by_doc = dict(zip(fixture["documents"], fixture["gold_facts"]))

    def _irrelevant_sentence(text: str, gold: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if len(line.split()) >= 10 and line.endswith(".") and gold not in line:
                return line
        for chunk in text.split(". "):
            sentence = chunk.strip() + "."
            if len(sentence.split()) >= 10 and sentence != gold and sentence in text:
                return sentence
        return ""

    def complete(prompt: str) -> str:
        if '"supported"' in prompt:  # hardened-verifier call
            quote = _span0_from_verifier_prompt(prompt)[:80]
            return json.dumps({"supported": True, "quote": quote})
        if prompt.startswith("TASK: EXTRACT"):
            for name, text in docs.items():
                if text in prompt:
                    return json.dumps({"sentence": gold_by_doc[name]})
            return json.dumps({"sentence": ""})
        if prompt.startswith("TASK: TRIAGE"):
            for name, text in docs.items():
                if text in prompt:
                    return json.dumps({"sentence": _irrelevant_sentence(text, gold_by_doc[name])})
            return json.dumps({"sentence": ""})
        if prompt.startswith("TASK: SYNTHESIZE") or prompt.startswith("TASK: BASELINE"):
            return " ".join(fixture["gold_facts"])
        return ""

    return complete


def _run_protocol(
    fixture: dict[str, Any],
    corpus_dir: Path,
    workdir: Path,
    n_workers: int,
    complete: Callable[[str], str],
) -> dict[str, Any]:
    worker_counting = CountingComplete(complete)
    verifier_counting = CountingComplete(complete)
    store = Store(workdir)
    sponsor = crypto.keypair_from_seed(f"eval-sponsor:{fixture['name']}")
    store.create_identity(sponsor.pubkey, SPONSOR_GRANT)
    task_id = make_realdoc_task(
        store,
        corpus_dir,
        question=fixture["question"],
        gold_facts=fixture["gold_facts"],
        documents=fixture["documents"],
        sponsor_pubkey=sponsor.pubkey,
    )
    workers = []
    initial: dict[str, int] = {}
    for index in range(n_workers):
        worker = LLMWorker.create(
            f"{fixture['name']}:worker{index}", fixture["question"], worker_counting
        )
        store.create_identity(worker.pubkey, WORKER_GRANT)
        initial[worker.pubkey] = WORKER_GRANT
        workers.append(worker)
    coordinator = Coordinator(store, LLMVerifier(complete=verifier_counting))

    settled = False
    peak_parallel_workers = 0
    for _ in range(MAX_EPOCHS):
        for worker in workers:
            worker.step(store, task_id)
        live = store.conn.execute(
            "SELECT COUNT(DISTINCT lease_holder) AS n FROM subtasks WHERE status = 'LEASED'"
        ).fetchone()["n"]
        peak_parallel_workers = max(peak_parallel_workers, int(live))
        coordinator.tick()
        if store.task_row(task_id)["settlement_applied"]:
            settled = True
            break

    admitted_facts = {
        claim["body"]
        for claim in store.admitted_claims(task_id)
        if claim["kind"] == "FACT"
    }
    solved = all(fact in admitted_facts for fact in fixture["gold_facts"])
    final = store.balances()
    payoffs = {
        worker.pubkey[:10]: final[worker.pubkey] - initial[worker.pubkey] for worker in workers
    }
    epochs = store.epoch()
    # Steering-concentration metric: how many distinct readers held v2
    # steering eligibility. On single-doc decompositions this collapses to the
    # finisher; fan-out on multi-document tasks is the Phase 3 signal that
    # search paths crossed.
    distinct_eligible_steering_readers = 0
    redundant = 0
    if store.task_row(task_id)["answer_claim_id"] is not None:
        snapshot = store.snapshot(task_id)
        distinct_eligible_steering_readers = len(
            settlement.eligible_steering_readers(snapshot)
        )
        redundant = settlement.redundant_work_avoided(snapshot)
    store.close()
    return {
        "solved": bool(solved and settled),
        "settled": settled,
        "epochs": epochs,
        "llm_calls": worker_counting.calls + verifier_counting.calls,
        "verifier_calls": verifier_counting.calls,
        "prompt_chars": worker_counting.prompt_chars + verifier_counting.prompt_chars,
        "distinct_eligible_steering_readers": distinct_eligible_steering_readers,
        "redundant_work_avoided": redundant,
        "peak_parallel_workers": peak_parallel_workers,
        "payoffs": payoffs,
    }


def _run_baseline(
    fixture: dict[str, Any], corpus_dir: Path, complete: Callable[[str], str]
) -> dict[str, Any]:
    counting = CountingComplete(complete)
    documents = "\n\n".join(
        (corpus_dir / name).read_text(encoding="utf-8") for name in fixture["documents"]
    )
    prompt = (
        "TASK: BASELINE\n"
        "Answer the QUESTION using only the DOCUMENTS. Include the exact "
        "sentences that support your answer.\n"
        f"QUESTION: {fixture['question']}\n"
        f"DOCUMENTS:\n{documents}"
    )
    try:
        answer = counting(prompt)
    except Exception:
        answer = ""
    solved = all(fact in answer for fact in fixture["gold_facts"])
    return {
        "solved": solved,
        "llm_calls": counting.calls,
        "prompt_chars": counting.prompt_chars,
        "answer_chars": len(answer),
    }


def run_eval(
    tasks_dir: str | Path,
    corpus_dir: str | Path | None = None,
    n_workers: int = 4,
    baseline: bool = True,
    complete: Callable[[str], str] | None = None,
    out_path: str | Path | None = "report.json",
    workdir: str | Path | None = None,
    fixture_paths: list[Path] | None = None,
) -> dict[str, Any]:
    tasks_dir = Path(tasks_dir)
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else tasks_dir.parent
    cost_per_call = float(os.environ.get("VSCP_COST_PER_CALL", "0.002"))
    use_real = complete is None and os.environ.get("VSCP_LLM") == "1"

    report: dict[str, Any] = {
        "cost_per_call": cost_per_call,
        "llm_backend": "real" if use_real else ("injected" if complete else "fake"),
        "n_workers": n_workers,
        "tasks": [],
    }
    for fixture_path in fixture_paths or sorted(tasks_dir.glob("*.json")):
        fixture = load_fixture(fixture_path)
        if complete is not None:
            task_complete = complete
        elif use_real:
            task_complete = openai_complete
        else:
            task_complete = make_fake_complete(fixture, corpus_dir)
        if workdir is None:
            with tempfile.TemporaryDirectory() as tmp:
                protocol = _run_protocol(fixture, corpus_dir, Path(tmp), n_workers, task_complete)
        else:
            task_workdir = Path(workdir) / fixture["name"]
            task_workdir.mkdir(parents=True, exist_ok=True)
            protocol = _run_protocol(fixture, corpus_dir, task_workdir, n_workers, task_complete)
        entry: dict[str, Any] = {
            "name": fixture["name"],
            "protocol": protocol,
            "est_cost_protocol": round(protocol["llm_calls"] * cost_per_call, 6),
        }
        if baseline:
            base = _run_baseline(fixture, corpus_dir, task_complete)
            entry["baseline"] = base
            entry["est_cost_baseline"] = round(base["llm_calls"] * cost_per_call, 6)
        report["tasks"].append(entry)

    report["total_llm_calls_protocol"] = sum(e["protocol"]["llm_calls"] for e in report["tasks"])
    report["total_llm_calls_baseline"] = sum(
        e.get("baseline", {}).get("llm_calls", 0) for e in report["tasks"]
    )
    total_cost_protocol = report["total_llm_calls_protocol"] * cost_per_call
    total_cost_baseline = report["total_llm_calls_baseline"] * cost_per_call
    report["cost_ratio"] = (
        round(total_cost_protocol / total_cost_baseline, 4) if total_cost_baseline else None
    )
    protocol_chars = sum(e["protocol"]["prompt_chars"] for e in report["tasks"])
    baseline_chars = sum(e.get("baseline", {}).get("prompt_chars", 0) for e in report["tasks"])
    report["prompt_chars_protocol"] = protocol_chars
    report["prompt_chars_baseline"] = baseline_chars
    report["char_ratio"] = (
        round(protocol_chars / baseline_chars, 4) if baseline_chars else None
    )

    if out_path is not None:
        Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_sweep(
    tasks_dir: str | Path,
    corpus_dir: str | Path | None = None,
    out_path: str | Path = "regime.json",
    workers_grid: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """The regime study grid: document class x worker count. Deterministic on
    the fake-LLM path; with a real endpoint it produces the headline numbers."""
    tasks_dir = Path(tasks_dir)
    all_fixtures = sorted(tasks_dir.glob("*.json"))
    classes = {
        "short": [p for p in all_fixtures if not p.name.startswith("long_")],
        "long": [p for p in all_fixtures if p.name.startswith("long_")],
    }
    sweep: dict[str, Any] = {"cells": []}
    for doc_class, fixtures in classes.items():
        if not fixtures:
            continue
        for n_workers in workers_grid:
            report = run_eval(
                tasks_dir,
                corpus_dir=corpus_dir,
                n_workers=n_workers,
                baseline=True,
                out_path=None,
                fixture_paths=fixtures,
            )
            sweep["cells"].append(
                {
                    "doc_class": doc_class,
                    "n_workers": n_workers,
                    "solved": all(e["protocol"]["solved"] for e in report["tasks"]),
                    "cost_ratio": report["cost_ratio"],
                    "char_ratio": report["char_ratio"],
                    "distinct_eligible_steering_readers": max(
                        e["protocol"]["distinct_eligible_steering_readers"]
                        for e in report["tasks"]
                    ),
                    "redundant_work_avoided": sum(
                        e["protocol"]["redundant_work_avoided"] for e in report["tasks"]
                    ),
                    "peak_parallel_workers": max(
                        e["protocol"]["peak_parallel_workers"] for e in report["tasks"]
                    ),
                    "total_llm_calls_protocol": report["total_llm_calls_protocol"],
                    "total_llm_calls_baseline": report["total_llm_calls_baseline"],
                }
            )
    Path(out_path).write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    return sweep


def format_sweep(sweep: dict[str, Any]) -> str:
    header = (
        f"{'docs':<7} {'workers':<8} {'solved':<7} {'cost_ratio':<11} "
        f"{'char_ratio':<11} {'steer_rdrs':<11} {'reuse':<6} {'parallel':<8}"
    )
    lines = [header, "-" * len(header)]
    for cell in sweep["cells"]:
        lines.append(
            f"{cell['doc_class']:<7} {cell['n_workers']:<8} {str(cell['solved']):<7} "
            f"{cell['cost_ratio']:<11} {cell['char_ratio']:<11} "
            f"{cell['distinct_eligible_steering_readers']:<11} "
            f"{cell['redundant_work_avoided']:<6} {cell['peak_parallel_workers']:<8}"
        )
    return "\n".join(lines)


def format_report(report: dict[str, Any]) -> str:
    header = (
        f"{'task':<26} {'solved':<7} {'base':<5} {'epochs':<7} "
        f"{'calls':<6} {'steer_rdrs':<10} {'cost':<9} {'base_cost':<9}"
    )
    lines = [header, "-" * len(header)]
    for entry in report["tasks"]:
        protocol = entry["protocol"]
        base = entry.get("baseline", {})
        lines.append(
            f"{entry['name']:<26} {str(protocol['solved']):<7} "
            f"{str(base.get('solved', '-')):<5} {protocol['epochs']:<7} "
            f"{protocol['llm_calls']:<6} "
            f"{protocol.get('distinct_eligible_steering_readers', 0):<10} "
            f"{entry['est_cost_protocol']:<9} "
            f"{entry.get('est_cost_baseline', '-'):<9}"
        )
        for pubkey, delta in sorted(protocol["payoffs"].items()):
            lines.append(f"  payoff {pubkey} {delta:+}")
    return "\n".join(lines)
