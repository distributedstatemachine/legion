"""`legion eval`: protocol vs honest baselines on realdoc fixtures.

Baselines (Phase 3.1 - the Phase 3 single-call comparison was false and is
superseded; see docs/REGIME_FINDINGS.md):

- `baseline_onecall`: the infinite-context oracle reference - every document
  in one prompt. Kept as an optimistic lower bound; recorded as infeasible
  (not silently truncated) when the corpus exceeds VSCP_CONTEXT_WINDOW_TOKENS.
- `baseline_iterative`: the fair comparator - a competent agent loop with no
  privileged access: one extraction call per document, one synthesis call,
  at most one re-query, capped at n_docs + 2 calls.

Gold facts are referenced ONLY by the grader. The fake stub is answer-key-free:
it extracts by deterministic keyword-overlap over the document and the
question, so it can miss (a baseline that cannot fail is not a baseline).
Token costing is primary: tokens are estimated as ceil(chars/4) and replaced
by provider usage numbers when the real endpoint returns them.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from legion import crypto, settlement
from legion.admission import LLMVerifier
from legion.coordinator import Coordinator
from legion.store import Store
from legion.tasks_realdoc import load_fixture, make_realdoc_task
from legion.workers.llm import LLMWorker

MAX_EPOCHS = 100
WORKER_GRANT = 200_000
SPONSOR_GRANT = 2_000_000
CHARS_PER_TOKEN = 4  # documented approximation; superseded by provider usage


def context_window_tokens() -> int:
    return int(os.environ.get("VSCP_CONTEXT_WINDOW_TOKENS", "128000"))


def cost_per_1k_tokens() -> float:
    return float(os.environ.get("VSCP_COST_PER_1K_TOKENS", "0.15"))


def max_total_llm_calls() -> int:
    return int(os.environ.get("VSCP_MAX_TOTAL_LLM_CALLS", "500"))


def estimate_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """Per-run hard cap on total LLM calls (real money rail)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self.capped = False

    def spend(self) -> None:
        if self.used >= self.limit:
            self.capped = True
            raise BudgetExceeded("per-run LLM call budget exceeded")
        self.used += 1


class CountingComplete:
    """Counts calls, chars, and tokens. Accepts inners returning either `str`
    or `(str, usage)`; provider usage (when present) supersedes the estimate."""

    def __init__(self, inner: Callable[[str], Any], budget: CallBudget | None = None) -> None:
        self.inner = inner
        self.budget = budget
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.provider_tokens = 0
        self.usage_seen = False
        self.prompts: list[str] = []
        self.completions: list[str] = []

    @property
    def tokens(self) -> int:
        if self.usage_seen:
            return self.provider_tokens
        return estimate_tokens(self.prompt_chars + self.completion_chars)

    @property
    def token_source(self) -> str:
        return "provider" if self.usage_seen else "estimated"

    def __call__(self, prompt: str) -> str:
        if self.budget is not None:
            self.budget.spend()
        self.calls += 1
        self.prompt_chars += len(prompt)
        self.prompts.append(prompt)
        result = self.inner(prompt)
        if isinstance(result, tuple):
            text, usage = result
            if isinstance(usage, dict):
                self.usage_seen = True
                self.provider_tokens += int(usage.get("prompt_tokens", 0)) + int(
                    usage.get("completion_tokens", 0)
                )
        else:
            text = result
        self.completion_chars += len(text)
        self.completions.append(text)
        return text


# ---------------------------------------------------------------------------
# Answer-key-free fake LLM (the CI path)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are",
    "what", "which", "how", "did", "does", "do", "each", "its", "their", "that",
    "was", "were", "by", "with", "from", "at", "as", "it", "they", "them", "who",
    "where", "when", "why", "this", "these", "those", "not", "but", "after",
    "before", "into", "over", "under", "about", "town", "make", "made", "them",
}


def _question_keywords(question: str) -> list[str]:
    words = re.findall(r"[a-z]+", question.lower())
    return sorted({w for w in words if len(w) >= 4 and w not in _STOPWORDS})


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        for chunk in line.strip().split(". "):
            sentence = chunk.strip()
            if not sentence:
                continue
            if not sentence.endswith("."):
                sentence += "."  # restore the period consumed by the split
            if len(sentence.split()) >= 10:
                out.append(sentence)
    return out


def _score(sentence: str, keywords: list[str]) -> int:
    lowered = sentence.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _ranked_sentences(document: str, question: str) -> list[str]:
    keywords = _question_keywords(question)
    sentences = _sentences(document)
    return sorted(
        sentences, key=lambda s: (-_score(s, keywords), sentences.index(s))
    )


def _between(prompt: str, start_marker: str, end_marker: str | None = None) -> str:
    start = prompt.index(start_marker) + len(start_marker)
    if end_marker is None:
        return prompt[start:]
    end = prompt.index(end_marker, start)
    return prompt[start:end]


def _span0_from_verifier_prompt(prompt: str) -> str:
    return _between(prompt, "SPAN 0\n", "\n</data>")


def heuristic_fake_complete(prompt: str) -> str:
    """Deterministic, answer-key-free stub. Extraction is keyword-overlap
    relevance over the document - it can and does miss when the question does
    not point at the right sentence. Gold facts never appear here."""
    if '"supported"' in prompt:  # hardened-verifier call
        return json.dumps({"supported": True, "quote": _span0_from_verifier_prompt(prompt)[:80]})
    if prompt.startswith("TASK: EXTRACT") or prompt.startswith("TASK: TRIAGE"):
        question = _between(prompt, "QUESTION: ", "\n")
        document = _between(prompt, "DOCUMENT:\n")
        ranked = _ranked_sentences(document, question)
        if not ranked:
            return json.dumps({"sentence": ""})
        index = 0 if prompt.startswith("TASK: EXTRACT") else min(1, len(ranked) - 1)
        return json.dumps({"sentence": ranked[index]})
    if prompt.startswith("TASK: SYNTHESIZE"):
        facts = [
            line[2:].strip()
            for line in prompt.splitlines()
            if line.startswith("- ")
        ]
        return " ".join(facts)
    if prompt.startswith("TASK: BASELINE"):
        question = _between(prompt, "QUESTION: ", "\n")
        corpus = _between(prompt, "DOCUMENTS:\n")
        ranked = _ranked_sentences(corpus, question)
        return " ".join(ranked[:8])
    return ""


def make_fake_complete(fixture: dict[str, Any], corpus_dir: Path) -> Callable[[str], str]:
    """Kept for signature compatibility; the stub is answer-key-free and
    ignores the fixture entirely (gold facts enter only the grader)."""
    del fixture, corpus_dir
    return heuristic_fake_complete


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_protocol(
    fixture: dict[str, Any],
    corpus_dir: Path,
    workdir: Path,
    n_workers: int,
    complete: Callable[[str], Any],
    budget: CallBudget | None = None,
) -> dict[str, Any]:
    worker_counting = CountingComplete(complete, budget)
    verifier_counting = CountingComplete(complete, budget)
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
    distinct_eligible_steering_readers = 0
    redundant = 0
    if store.task_row(task_id)["answer_claim_id"] is not None:
        snapshot = store.snapshot(task_id)
        distinct_eligible_steering_readers = len(
            settlement.eligible_steering_readers(snapshot)
        )
        redundant = settlement.redundant_work_avoided(snapshot)
    store.close()
    tokens = worker_counting.tokens + verifier_counting.tokens
    return {
        "solved": bool(solved and settled),
        "settled": settled,
        "epochs": epochs,
        "llm_calls": worker_counting.calls + verifier_counting.calls,
        "verifier_calls": verifier_counting.calls,
        "tokens": tokens,
        "token_source": (
            "provider"
            if worker_counting.usage_seen or verifier_counting.usage_seen
            else "estimated"
        ),
        "distinct_eligible_steering_readers": distinct_eligible_steering_readers,
        "redundant_work_avoided": redundant,
        "peak_parallel_workers": peak_parallel_workers,
        "payoffs": payoffs,
    }


def _grade(answer: str, gold_facts: list[str]) -> bool:
    # The grader - the ONLY place gold facts are referenced.
    return all(fact in answer for fact in gold_facts)


def _run_baseline_onecall(
    fixture: dict[str, Any],
    corpus_dir: Path,
    complete: Callable[[str], Any],
    budget: CallBudget | None = None,
) -> dict[str, Any]:
    """The infinite-context oracle reference: every document in one prompt.
    Recorded as infeasible (never truncated) when the corpus exceeds the
    context window."""
    counting = CountingComplete(complete, budget)
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
    window = context_window_tokens()
    if estimate_tokens(len(prompt)) > window:
        return {
            "feasible": False,
            "reason": "corpus exceeds context window",
            "context_window_tokens": window,
            "estimated_prompt_tokens": estimate_tokens(len(prompt)),
            "solved": False,
            "llm_calls": 0,
            "tokens": 0,
            "token_source": "estimated",
        }
    try:
        answer = counting(prompt)
    except Exception:
        answer = ""
    return {
        "feasible": True,
        "solved": _grade(answer, fixture["gold_facts"]),
        "llm_calls": counting.calls,
        "tokens": counting.tokens,
        "token_source": counting.token_source,
    }


def _run_baseline_iterative(
    fixture: dict[str, Any],
    corpus_dir: Path,
    complete: Callable[[str], Any],
    budget: CallBudget | None = None,
) -> dict[str, Any]:
    """The fair comparator: a competent agent loop with no privileged access.
    One extraction call per document, one synthesis call, at most one
    re-query; capped at n_docs + 2 calls total."""
    counting = CountingComplete(complete, budget)
    docs = {
        name: (corpus_dir / name).read_text(encoding="utf-8")
        for name in fixture["documents"]
    }
    call_cap = len(docs) + 2

    def extract(name: str, text: str) -> str | None:
        prompt = (
            "TASK: EXTRACT\n"
            "From the DOCUMENT below, extract the single sentence most relevant "
            "to the QUESTION. Copy it verbatim, character for character.\n"
            'Return strict JSON only: {"sentence": "..."}\n'
            f"QUESTION: {fixture['question']}\n"
            f"DOCUMENT:\n{text}"
        )
        try:
            raw = counting(prompt)
        except Exception:
            return None
        parsed = LLMVerifier._parse_first_json(raw)
        sentence = parsed.get("sentence") if isinstance(parsed, dict) else None
        if isinstance(sentence, str) and sentence in text:
            return sentence
        return None

    extracted: dict[str, str] = {}
    missed: list[str] = []
    for name, text in docs.items():
        if counting.calls >= call_cap:
            break
        sentence = extract(name, text)
        if sentence:
            extracted[name] = sentence
        else:
            missed.append(name)
    # One optional re-query for a doc whose extraction failed the verbatim check.
    if missed and counting.calls < call_cap - 1:
        name = missed[0]
        sentence = extract(name, docs[name])
        if sentence:
            extracted[name] = sentence

    answer = ""
    if extracted and counting.calls < call_cap + 1:
        synthesis_prompt = (
            "TASK: SYNTHESIZE\n"
            "Compose a one-paragraph answer to the QUESTION strictly from the "
            "FACTS below. Plain text, no preamble.\n"
            f"QUESTION: {fixture['question']}\n"
            "FACTS:\n" + "\n".join(f"- {sentence}" for sentence in extracted.values())
        )
        try:
            answer = counting(synthesis_prompt)
        except Exception:
            answer = ""
    return {
        "solved": _grade(answer, fixture["gold_facts"]),
        "llm_calls": counting.calls,
        "tokens": counting.tokens,
        "token_source": counting.token_source,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_complete(
    explicit: Callable[[str], Any] | None,
) -> tuple[Callable[[str], Any], str]:
    from legion import llm_client

    if explicit is not None:
        return explicit, "injected"
    if llm_client.real_path_enabled():
        return lambda prompt: llm_client.complete_text(prompt), "real"
    return heuristic_fake_complete, "fake"


def run_eval(
    tasks_dir: str | Path,
    corpus_dir: str | Path | None = None,
    n_workers: int = 4,
    baseline: bool = True,
    complete: Callable[[str], Any] | None = None,
    out_path: str | Path | None = "report.json",
    workdir: str | Path | None = None,
    fixture_paths: list[Path] | None = None,
) -> dict[str, Any]:
    tasks_dir = Path(tasks_dir)
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else tasks_dir.parent
    resolved_complete, backend = _resolve_complete(complete)
    budget = CallBudget(max_total_llm_calls())
    rate = cost_per_1k_tokens()

    from legion import llm_client

    report: dict[str, Any] = {
        "llm_backend": backend,
        "model": llm_client.resolve_model() if backend == "real" else None,
        "cost_per_1k_tokens": rate,
        "cost_per_call": float(os.environ.get("VSCP_COST_PER_CALL", "0.002")),  # legacy
        "context_window_tokens": context_window_tokens(),
        "n_workers": n_workers,
        "tasks": [],
    }
    for fixture_path in fixture_paths or sorted(tasks_dir.glob("*.json")):
        fixture = load_fixture(fixture_path)
        if workdir is None:
            with tempfile.TemporaryDirectory() as tmp:
                protocol = _run_protocol(
                    fixture, corpus_dir, Path(tmp), n_workers, resolved_complete, budget
                )
        else:
            task_workdir = Path(workdir) / fixture["name"]
            task_workdir.mkdir(parents=True, exist_ok=True)
            protocol = _run_protocol(
                fixture, corpus_dir, task_workdir, n_workers, resolved_complete, budget
            )
        entry: dict[str, Any] = {
            "name": fixture["name"],
            "protocol": protocol,
            "est_cost_protocol": round(protocol["tokens"] / 1000 * rate, 6),
        }
        if baseline:
            onecall = _run_baseline_onecall(fixture, corpus_dir, resolved_complete, budget)
            iterative = _run_baseline_iterative(fixture, corpus_dir, resolved_complete, budget)
            entry["baseline_onecall"] = onecall
            entry["baseline_iterative"] = iterative
            entry["est_cost_onecall"] = round(onecall["tokens"] / 1000 * rate, 6)
            entry["est_cost_iterative"] = round(iterative["tokens"] / 1000 * rate, 6)
        report["tasks"].append(entry)

    protocol_tokens = sum(e["protocol"]["tokens"] for e in report["tasks"])
    iterative_tokens = sum(
        e.get("baseline_iterative", {}).get("tokens", 0) for e in report["tasks"]
    )
    feasible_onecall = [
        e["baseline_onecall"]
        for e in report["tasks"]
        if e.get("baseline_onecall", {}).get("feasible")
    ]
    onecall_tokens = sum(b["tokens"] for b in feasible_onecall)
    report["total_tokens_protocol"] = protocol_tokens
    report["total_tokens_iterative"] = iterative_tokens
    report["total_tokens_onecall_feasible"] = onecall_tokens
    # Headline: token_ratio vs the fair (iterative) comparator.
    report["token_ratio"] = (
        round(protocol_tokens / iterative_tokens, 4) if iterative_tokens else None
    )
    report["token_ratio_vs_onecall"] = (
        round(protocol_tokens / onecall_tokens, 4) if onecall_tokens else None
    )
    # Legacy secondary: per-call pricing structurally penalizes any decomposed
    # system; informative only at equal call granularity.
    protocol_calls = sum(e["protocol"]["llm_calls"] for e in report["tasks"])
    iterative_calls = sum(
        e.get("baseline_iterative", {}).get("llm_calls", 0) for e in report["tasks"]
    )
    report["total_llm_calls_protocol"] = protocol_calls
    report["total_llm_calls_iterative"] = iterative_calls
    report["cost_ratio_calls_legacy"] = (
        round(protocol_calls / iterative_calls, 4) if iterative_calls else None
    )
    report["token_source"] = (
        "provider"
        if any(e["protocol"]["token_source"] == "provider" for e in report["tasks"])
        else "estimated"
    )
    report["budget_capped"] = budget.capped
    report["total_llm_calls_all_runners"] = budget.used
    report["est_cost_total"] = round(
        (protocol_tokens + iterative_tokens + onecall_tokens) / 1000 * rate, 6
    )

    if out_path is not None:
        Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _fixture_class(path: Path) -> str:
    if path.name.startswith("xl_"):
        return "xl"
    if path.name.startswith("long_"):
        return "long"
    return "short"


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
    classes: dict[str, list[Path]] = {}
    for path in all_fixtures:
        classes.setdefault(_fixture_class(path), []).append(path)
    sweep: dict[str, Any] = {"cells": []}
    for doc_class in ("short", "long", "xl"):
        fixtures = classes.get(doc_class)
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
            onecall_feasible = all(
                e["baseline_onecall"]["feasible"] for e in report["tasks"]
            )
            sweep["cells"].append(
                {
                    "doc_class": doc_class,
                    "n_workers": n_workers,
                    "solved_protocol": all(e["protocol"]["solved"] for e in report["tasks"]),
                    "solved_iterative": all(
                        e["baseline_iterative"]["solved"] for e in report["tasks"]
                    ),
                    "onecall_feasible": onecall_feasible,
                    "token_ratio": report["token_ratio"],
                    "token_ratio_vs_onecall": report["token_ratio_vs_onecall"],
                    "cost_ratio_calls_legacy": report["cost_ratio_calls_legacy"],
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
                }
            )
    Path(out_path).write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    return sweep


def format_report(report: dict[str, Any]) -> str:
    header = (
        f"{'task':<24} {'proto':<6} {'iter':<5} {'1call':<9} {'epochs':<7} "
        f"{'calls':<6} {'tokens':<8} {'steer':<6}"
    )
    lines = [
        f"backend={report['llm_backend']} model={report.get('model') or '-'} "
        f"token_source={report['token_source']}",
        header,
        "-" * len(header),
    ]
    for entry in report["tasks"]:
        protocol = entry["protocol"]
        onecall = entry.get("baseline_onecall", {})
        iterative = entry.get("baseline_iterative", {})
        onecall_label = (
            "infeas" if onecall and not onecall.get("feasible", True) else str(onecall.get("solved", "-"))
        )
        lines.append(
            f"{entry['name']:<24} {str(protocol['solved']):<6} "
            f"{str(iterative.get('solved', '-')):<5} {onecall_label:<9} "
            f"{protocol['epochs']:<7} {protocol['llm_calls']:<6} "
            f"{protocol['tokens']:<8} "
            f"{protocol.get('distinct_eligible_steering_readers', 0):<6}"
        )
        for pubkey, delta in sorted(protocol["payoffs"].items()):
            lines.append(f"  payoff {pubkey} {delta:+}")
    lines.append(
        f"token_ratio (vs iterative, headline) = {report['token_ratio']}  |  "
        f"vs onecall = {report['token_ratio_vs_onecall']}  |  "
        f"call ratio (legacy, structurally anti-decomposition) = "
        f"{report['cost_ratio_calls_legacy']}"
    )
    lines.append(
        f"tokens: protocol={report['total_tokens_protocol']} "
        f"iterative={report['total_tokens_iterative']} "
        f"est_cost_total=${report['est_cost_total']}"
        + ("  [BUDGET CAPPED]" if report.get("budget_capped") else "")
    )
    return "\n".join(lines)


def format_sweep(sweep: dict[str, Any]) -> str:
    header = (
        f"{'docs':<7} {'workers':<8} {'proto':<6} {'iter':<5} {'1call':<7} "
        f"{'tok_ratio':<10} {'vs_1call':<9} {'steer':<6} {'reuse':<6}"
    )
    lines = [header, "-" * len(header)]
    for cell in sweep["cells"]:
        lines.append(
            f"{cell['doc_class']:<7} {cell['n_workers']:<8} "
            f"{str(cell['solved_protocol']):<6} {str(cell['solved_iterative']):<5} "
            f"{('yes' if cell['onecall_feasible'] else 'INFEAS'):<7} "
            f"{str(cell['token_ratio']):<10} {str(cell['token_ratio_vs_onecall']):<9} "
            f"{cell['distinct_eligible_steering_readers']:<6} "
            f"{cell['redundant_work_avoided']:<6}"
        )
    return "\n".join(lines)
