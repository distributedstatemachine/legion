"""Phase 3.1 Part A invariants - the tests that would have caught the false
baseline: gold facts reach prompts only via documents or the model's own
output, baselines take real multi-call work, the grader can fail, and the
one-call oracle is marked infeasible beyond the context window."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from legion import evaluate
from legion.evaluate import heuristic_fake_complete, run_eval

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASKS = CORPUS / "tasks"
TRAP_FIXTURE = Path(__file__).with_name("fixtures") / "miss_trap.json"


class Spy:
    """Records every (prompt, completion) pair in order."""

    def __init__(self, inner):
        self.inner = inner
        self.events: list[tuple[str, str]] = []

    def __call__(self, prompt: str) -> str:
        text = self.inner(prompt)
        self.events.append((prompt, text))
        return text


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_gold_fact_reaches_a_prompt_except_via_documents_or_model_output(tmp_path):
    # Gold facts may enter prompts only (a) inside their full source document,
    # or (b) because the model itself previously produced them. Any other
    # appearance means the harness leaked the answer key.
    for fixture_path in [TASKS / "night_safety.json", TASKS / "long_deep_archive.json"]:
        fixture = _fixture(fixture_path)
        docs = {
            name: (CORPUS / name).read_text(encoding="utf-8")
            for name in fixture["documents"]
        }
        spy = Spy(heuristic_fake_complete)
        report = run_eval(
            TASKS,
            corpus_dir=CORPUS,
            n_workers=4,
            baseline=True,
            complete=spy,
            out_path=tmp_path / f"{fixture['name']}.json",
            fixture_paths=[fixture_path],
        )
        assert report["tasks"][0]["protocol"]["settled"]
        assert spy.events, "expected LLM traffic"
        for index, (prompt, _completion) in enumerate(spy.events):
            prior_output = "\n".join(text for _, text in spy.events[:index])
            for fact in fixture["gold_facts"]:
                if fact not in prompt:
                    continue
                via_document = any(text in prompt for text in docs.values() if fact in text)
                via_model = fact in prior_output
                assert via_document or via_model, (
                    f"gold fact leaked into a prompt outside its document: {fact[:60]}..."
                )


def test_fake_stub_is_answer_key_free():
    source = inspect.getsource(evaluate.heuristic_fake_complete)
    assert "gold" not in source
    # make_fake_complete must not route fixture gold facts anywhere.
    source = inspect.getsource(evaluate.make_fake_complete)
    assert 'fixture["gold_facts"]' not in source


def test_iterative_baseline_takes_real_multicall_work(tmp_path):
    multi_doc = [p for p in sorted(TASKS.glob("*.json")) if len(_fixture(p)["documents"]) > 1]
    assert multi_doc
    for fixture_path in multi_doc:
        report = run_eval(
            TASKS,
            corpus_dir=CORPUS,
            n_workers=2,
            baseline=True,
            out_path=None,
            fixture_paths=[fixture_path],
        )
        iterative = report["tasks"][0]["baseline_iterative"]
        assert iterative["llm_calls"] > 1, fixture_path.name


def test_iterative_baseline_can_fail_so_the_grader_is_real(tmp_path):
    # The trap fixture's question points at the wrong sentence: a competent
    # but unprivileged extractor misses the gold fact, and the grader says so.
    report = run_eval(
        TRAP_FIXTURE.parent,
        corpus_dir=CORPUS,
        n_workers=2,
        baseline=True,
        out_path=None,
        fixture_paths=[TRAP_FIXTURE],
    )
    (entry,) = report["tasks"]
    assert entry["baseline_iterative"]["llm_calls"] >= 1
    assert entry["baseline_iterative"]["solved"] is False
    assert entry["protocol"]["solved"] is False  # same honest miss, both sides


def test_onecall_baseline_infeasible_beyond_context_window(tmp_path):
    xl = next(TASKS.glob("xl_*.json"))
    report = run_eval(
        TASKS,
        corpus_dir=CORPUS,
        n_workers=2,
        baseline=True,
        out_path=tmp_path / "xl.json",
        fixture_paths=[xl],
    )
    (entry,) = report["tasks"]
    onecall = entry["baseline_onecall"]
    assert onecall["feasible"] is False
    assert onecall["reason"] == "corpus exceeds context window"
    assert onecall["llm_calls"] == 0  # never silently truncated, never issued
    # The protocol and the iterative baseline still run - and still solve.
    assert entry["protocol"]["solved"] is True
    assert entry["baseline_iterative"]["solved"] is True
    # The infeasible oracle is excluded from ratios.
    assert report["token_ratio_vs_onecall"] is None
    assert report["token_ratio"] is not None and report["token_ratio"] > 0


def test_budget_cap_emits_partial_report(tmp_path, monkeypatch):
    monkeypatch.setenv("VSCP_MAX_TOTAL_LLM_CALLS", "3")
    report = run_eval(
        TASKS,
        corpus_dir=CORPUS,
        n_workers=2,
        baseline=True,
        out_path=tmp_path / "capped.json",
        fixture_paths=[TASKS / "night_safety.json"],
    )
    assert report["budget_capped"] is True
    assert report["total_llm_calls_all_runners"] <= 3
    assert (tmp_path / "capped.json").exists()  # partial report still emitted
