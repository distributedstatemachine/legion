# Legion — Phase 3.1 Amendment: Honest Baselines + Real LLM (OpenRouter)

**Version:** 3.1 (amends Phase 3; builds on `distributedstatemachine/legion`, commit `2982809` or later)
**Audience:** an autonomous coding agent inside the repo. Self-contained. Do not regress the 80 passing tests; where a Phase 3 number changes because the baseline was wrong, that is expected — update `REGIME_FINDINGS.md` rather than preserving the old figure.

**Why this exists.** The Phase 3 eval compared the protocol against a baseline that was false in two compounding ways: (1) the fake `complete` stub had a `TASK: BASELINE` branch returning `" ".join(fixture["gold_facts"])` — the answer key, in one call (`legion/evaluate.py` line ~81), so the baseline never *solved* anything, it was *handed* the answer; and (2) even setting the secret aside, a one-call baseline is an infinite-context oracle that does not exist — a real agent loops retrieve→read→reason→synthesize over many calls. Every `cost_ratio` in `REGIME_FINDINGS.md` is therefore measured against a fiction, and it penalizes the protocol. This amendment fixes the comparison and enables real-endpoint runs via OpenRouter.

---

## Part A — Honest baselines

### A1. Kill the gold-fact shortcut (the load-bearing fix)
Remove any branch in `make_fake_complete` (and anywhere else) that returns `fixture["gold_facts"]` — wholesale or joined — in response to a baseline/synthesis prompt. The gold facts may be referenced **only** in grading (`solved = all(fact in answer ...)`), never in any string a model-or-stub returns as an answer. The fake stub must produce baseline answers by the *same per-document extraction path it uses for workers*: read each document, extract its candidate sentence, then synthesize. A fake baseline must be able to *fail* (miss a fact) — if it can't fail, it isn't a baseline.

**Invariant test (`tests/test_realdoc_eval.py`), the test that would have caught this:**
- Wrap the eval's `complete` callable in a spy that records every prompt. Assert that for every baseline and every worker prompt, **no `gold_fact` substring appears in the prompt** the model receives. (Gold facts enter only the grader.)
- Assert `baseline.llm_calls > 1` on every multi-document fixture for both fake and (when enabled) real backends.
- Assert the iterative baseline can score `solved=False` on at least one constructed fixture where extraction is made to miss (e.g. a fixture whose gold fact is not the most-relevant-looking sentence), proving the grader is real and not rubber-stamping.

### A2. Two baselines, both reported
Replace the single `_run_baseline` with two, run and reported side by side:

1. **`baseline_onecall`** (optimistic lower bound): the current single-prompt, all-documents-concatenated call. *Keep it*, but relabel it as the infinite-context oracle reference. On a corpus that exceeds the model's context window it must be recorded as `infeasible` (see A4), not silently truncated.
2. **`baseline_iterative`** (the fair comparator): a competent agent loop with no privileged access —
   - one extraction call per document (retrieve the document, ask for the sentence most relevant to the question),
   - one synthesis call combining the extracted sentences into an answer,
   - optionally one re-query call if synthesis reports an unsupported gap (cap the loop at `n_docs + 2` calls).
   It sees the same documents the protocol's workers can fetch — do **not** hobble it (no withholding documents, no sub-task-width call caps). The credible baseline is the obvious competent approach; the protocol must beat what you would actually deploy.

The grader is byte-identical across protocol, `baseline_onecall`, and `baseline_iterative`; the only variable is method.

### A3. Token-based costing as the primary metric
- Add a `CountingComplete` field `total_tokens` estimated as `ceil(chars / 4)` over prompt **and** completion (document the 4-chars/token approximation in `DECISIONS.md`; replace with the endpoint's real usage numbers when available — see B3).
- New headline metric `token_ratio = protocol_tokens / baseline_iterative_tokens`, reported first. Demote `cost_ratio` (calls) to a clearly-labelled secondary with a one-line note that per-call pricing structurally penalizes any decomposed system and is informative only at equal call granularity.
- Add `cost_per_1k_tokens` (env `VSCP_COST_PER_1K_TOKENS`, default e.g. `0.15`) and report `est_cost_*` from tokens, not calls. Keep `cost_per_call` only for the legacy secondary.

### A4. Context-window handling (makes the decisive experiment measurable)
- Add `VSCP_CONTEXT_WINDOW_TOKENS` (default e.g. `128000`). Before issuing the one-call baseline, if the concatenated prompt's estimated tokens exceed it, record `baseline_onecall = {"feasible": false, "reason": "corpus exceeds context window"}` and exclude it from ratios (the iterative baseline and protocol still run). This is the regime `REGIME_FINDINGS.md` extrapolated to: where the oracle baseline is not costly but *impossible*.
- Add at least one fixture (`corpus/tasks/xl_*.json`) whose corpus deliberately exceeds the default window so the infeasible branch is exercised in CI on the fake path (the fake stub honors the same token estimate and marks one-call infeasible).

### A5. Findings doc
Rewrite `docs/REGIME_FINDINGS.md` against the corrected baselines. Report `token_ratio` vs the **iterative** baseline as the headline, show all three runners per cell, and state plainly where (if anywhere) the protocol now wins — especially on the XL fixture where `baseline_onecall` is infeasible. Preserve the §9 honesty rule: a null result, if that's what the corrected data shows, is stated, not buried. Note explicitly that the Phase 3 numbers were against a false baseline and are superseded.

---

## Part B — Real LLM via OpenRouter

The existing OpenAI-compatible client in `admission.py` (uses `VSCP_LLM_URL` + bearer key) already speaks OpenRouter's protocol. This part wires it end to end and routes worker/baseline calls through it, not just the verifier.

### B1. Unified client
Extract the HTTP logic from `LLMVerifier` into `legion/llm_client.py: openai_chat(messages, *, model, max_tokens, temperature) -> (text, usage)` returning both content and the provider's `usage` block when present. Configuration, all via env:
- `VSCP_LLM=1` enables the real path (unchanged).
- `VSCP_LLM_URL` default `https://openrouter.ai/api/v1/chat/completions`.
- API key from `OPENROUTER_API_KEY` or `VSCP_LLM_API_KEY` or `OPENAI_API_KEY` (first present wins).
- `VSCP_LLM_MODEL` default a documented OpenRouter model string (e.g. `openai/gpt-4o-mini`); `VSCP_VERIFIER_MODEL` may override it for the verifier specifically.
- OpenRouter niceties: send optional `HTTP-Referer` / `X-Title` headers from `VSCP_LLM_REFERER` / `VSCP_LLM_TITLE` when set (OpenRouter ranks/labels by these; harmless if absent).
- Keep: temperature 0, 30 s timeout, ≤ 1 transport retry, never retry a NO/parse failure.
`LLMVerifier` becomes a thin caller of this client (its injection defenses and deterministic quote check are unchanged and must keep passing the injection suite).

### B2. Route workers and baselines through the client
`run_eval`'s real branch builds `complete` from `openai_chat` and injects it into both `LLMWorker`s and both baselines (currently only the verifier path is real). The fake stub stays the default and the CI path. Selection unchanged: real iff `VSCP_LLM=1` and a key is present; otherwise fake.

### B3. Real token accounting
When the provider returns a `usage` block, use its `prompt_tokens` / `completion_tokens` for costing instead of the char/4 estimate; fall back to the estimate only when usage is absent. Record which was used per run in the report (`"token_source": "provider" | "estimated"`).

### B4. Budget and safety rails (real money)
- Per-run hard cap `VSCP_MAX_TOTAL_LLM_CALLS` (default e.g. `500`); on exceed, stop issuing calls, mark the run `budget_capped: true`, and still emit a partial report.
- The existing per-worker `VSCP_MAX_LLM_CALLS_PER_WORKER` stays.
- Never log the API key. A `--dry-run` flag prints the planned model, endpoint, estimated max calls, and estimated max cost, then exits without calling.
- `legion eval` prints the resolved backend, model, and (post-run) actual token usage and estimated dollar cost.

### B5. Tests (all offline / CI-safe)
- `tests/test_llm_client.py`: against a fake transport (monkeypatched `urllib`), assert request shape (model, temperature 0, auth header present, optional referer/title forwarded), `usage` parsing, the single-retry-on-transport-error / no-retry-on-bad-output policy, and that the key never appears in any logged/printed output.
- The injection suite (`tests/test_verifier_injection.py`) must still pass with the verifier delegating to the extracted client.
- A skipped-by-default `tests/test_openrouter_live.py` (runs only when `VSCP_LLM=1` + key present): one real round-trip on the smallest fixture asserting a non-empty answer and a populated `usage` block. Mirrors the existing LLM-smoke-skip convention.

---

## Milestones (strict order; each ends green)
- **P3.1-M1 — Honest baselines** (Part A): gold-fact shortcut removed; dual baselines; token costing primary; context-window infeasibility + XL fixture; the prompt-leak invariant test and `baseline.llm_calls > 1` test; `REGIME_FINDINGS.md` rewritten on fake-path data.
- **P3.1-M2 — OpenRouter** (Part B): unified client; workers + baselines on the real path; real token accounting; budget rails + `--dry-run`; offline client tests + skipped live test.

## Definition of done
`pytest -q` green with ≥ 6 new tests; the prompt-leak test fails if any gold fact reaches a model prompt; both baselines reported per cell with `token_ratio` headline; `legion eval --dry-run` prints plan without calling; `legion eval` with `VSCP_LLM=1` + `OPENROUTER_API_KEY` runs against OpenRouter (verified manually, not in CI); `REGIME_FINDINGS.md` states the corrected result and flags the Phase 3 numbers as superseded; `DECISIONS.md` updated (token approximation, model defaults, baseline definitions); README documents the OpenRouter env vars in ≤ 12 lines.

## Note on what this fixes and doesn't
This corrects the *comparison*, not the *mechanism* — conservation, determinism, incentive-compatibility, and auditability are untouched and their tests must stay green. The expected outcome is that the protocol looks substantially better against an honest iterative baseline than it did against the false one-call oracle, and that on the XL fixture the one-call baseline becomes infeasible while the protocol still settles. If, against the corrected and *competent* iterative baseline, the protocol still loses at all tested scales, that remains the honest finding and belongs in `REGIME_FINDINGS.md` — do not weaken the iterative baseline to manufacture a win.