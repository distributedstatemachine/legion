# Legion — Phase 3 Specification

**Version:** 3.0 (builds on `distributedstatemachine/legion`, commit `627e518` or later — the Phase 2 head)
**Audience:** an autonomous coding agent working *inside the existing repo*. Self-contained. Where silent, choose the simplest option consistent with §7 and append the decision to `docs/DECISIONS.md`. Do not regress any currently passing test (62 at the Phase 2 head); settlement-semantic changes stay behind the existing version flag.

**Premise.** Phase 2 shipped steering v2, a hardened verifier, an equivalence harness, and an eval pipeline that honestly reported the protocol losing to a single-agent baseline by ~7× *at toy scale* (1 kB docs, 2 workers). Phase 3 has one through-line: **find and measure the regime where the protocol's coordination actually pays for itself**, close the gaps the Phase 2 review found, and take the first real step toward distribution by proving the ledger is independently re-derivable. No consensus, no networking between machines beyond local processes, no token/chain integration.

---

## 1. Scope

### In scope
1. **Carryover fixes** from the Phase 2 review (§2): the broken `python -m` CLI, the unbuilt §4A simulation + statistical goldens, and a long-document eval fixture.
2. **The economics regime study** (§3): sweep document length, worker count, and task width; emit the metrics that locate the protocol's break-even point against the baseline; add the eligible-steering-reader metric.
3. **Multi-process operation** (§4): N worker processes + 1 coordinator process against one WAL SQLite ledger, with documented concurrency discipline. Still single-machine, still one trusted coordinator.
4. **Light-client re-derivation** (§5): a standalone verifier that replays the ledger from transfers + claims alone and reconstructs every balance and settlement, byte-identical, in a separate process. This is the cheap first audit that the Phase 1/2 determinism work was building toward.

### Out of scope
Real consensus or BFT; cross-machine networking; P2P gossip; substrate/chain selection; reputation; cross-task identity; Shapley/hypergraph *settlement* (the §4A sim implements Shapley only for its own goldens, never for `legion.settlement`). No new runtime dependencies outside the stdlib except `pytest`/`hypothesis` (dev) — the LLM path stays on `urllib`.

---

## 2. Carryover fixes (do these first; each ends green)

### 2.1 CLI entry point
`python -m legion.cli <cmd>` currently imports and exits 0 silently (no `__main__` guard). Add `if __name__ == "__main__": main()` to `legion/cli.py`. Add `tests/test_cli_entrypoint.py` invoking `python -m legion.cli demo --workers 4 --honest 3 --hoarders 1 --seed 7 --workdir <tmp>` as a subprocess and asserting exit 0 **and** non-empty stdout containing `mean_honest_delta`. The silent-success failure mode is the thing being tested — a no-op must fail the test.

### 2.2 Build §4A — the research simulation and statistical goldens
Phase 2 implemented only the settlement *reference* (`tools/sim/sim.py`, float, settlement-only). Phase 3 adds the full agent-based model and its goldens, exactly as specified in **§4A of the Phase 2 spec v2.1** (world model, the seven strategies HONEST/HOARDER/SPAMMER/RING_SYBIL/RING_BENEF/POISONER/HYBRID, flat + Shapley settlement, `run_many`). Place the agent model in `tools/sim/model.py` (keep the existing `sim.py` settlement reference; `model.py` may import it). Implement `tools/sim/experiments.py` and `tests/test_sim_goldens.py` with the eight statistical-golden bands of §4A.6.

Conformance rule, restated: ordering assertions (the `>`/`<` clauses — honest > hoarder, lone poisoner profitable under Shapley but not flat, all-hybrid ≤ 0.75× all-honest, etc.) **may not be weakened**; numeric bands may be widened only with the measured value recorded in `DECISIONS.md` and called out in the PR. `tools/sim/` still must not import `legion` (CI grep already enforces this; keep it passing).

### 2.3 Long-document eval fixture
Add one fixture task `corpus/tasks/long_*.json` backed by genuinely long documents (each 20–50 kB; 4–8 documents; public-domain or original CC0 text committed to `corpus/`). The question must require evidence from ≥3 documents so the decomposition is real. This is the fixture the regime study (§3) leans on; the existing short fixtures stay as the fast-CI path.

---

## 3. Economics regime study (the core of Phase 3)

**Goal:** turn the single honest "protocol loses 7×" data point into a curve, and identify whether/where a crossover exists. This is measurement, not optimization — do not tune the mechanism to manufacture a win.

### 3.1 New eval metrics
Extend `legion/evaluate.py`'s per-task `protocol` report block (currently `epochs, llm_calls, payoffs, settled, solved`) with:
- `distinct_eligible_steering_readers`: the count from the steering-v2 eligibility pass (promotes the Phase 2 DECISIONS.md observation that steering concentrates wherever search paths cross — on single-doc-per-subtask tasks this was effectively 1; the study measures whether it fans out on multi-doc tasks).
- `redundant_work_avoided`: number of admitted FAIL/CONSTRAINT claims fetched by ≥1 productive reader (a proxy for the reuse the paper credits for its cost savings).
- `peak_parallel_workers`: max workers holding a live lease in any single epoch.
- `verifier_calls` (already counted separately by the harness — surface it in the report).
And per-run top level: `total_llm_calls_protocol`, `total_llm_calls_baseline`, `cost_ratio = est_cost_protocol / est_cost_baseline`.

### 3.2 The sweep
`legion eval --sweep` (or a `tools/regime_sweep.py` driver) runs the eval across the grid:
- document length ∈ {short fixtures, long fixture (§2.3)},
- `n_workers` ∈ {1, 2, 4, 8},
- and reports `cost_ratio`, `solved`, `distinct_eligible_steering_readers`, and `redundant_work_avoided` for each cell.
Writes `regime.json` and prints a grid. On the fake-LLM path this must run in CI (deterministic); with a real endpoint it produces the headline numbers.

### 3.3 Interpretation artifact
Add `docs/REGIME_FINDINGS.md`: a short, honest write-up of what the sweep shows — including, prominently, if no crossover is found at the scales tested. State the conditions under which the protocol would be expected to win (long multi-doc tasks where baseline context cost grows and cross-worker reuse is high) and whether the data supports that expectation. A null result is a valid, required outcome; do not bury it.

### 3.4 Tests
`tests/test_regime_metrics.py`: on the long fixture with the fake LLM, assert `distinct_eligible_steering_readers >= 2` (the multi-doc fan-out the Phase 2 review predicted), `redundant_work_avoided >= 0` is reported, and `cost_ratio` is present and positive. No assertion on crossover direction — that's a finding, not an invariant.

---

## 4. Multi-process operation

**Goal:** prove the architecture survives real concurrency, still single-machine, still one coordinator. This validates the Phase 1 claim that only task-lease acquisition needs serialization while knowledge claims are commutative.

### 4.1 Model
- One **coordinator** process: the sole writer for admission, epoch advance, close, and settlement (its existing `tick()`). It owns those transitions exclusively.
- N **worker** processes: each runs the existing worker loop in its own process, connecting to the same `ledger.db` (WAL). Workers only ever call lease/fetch/submit — never admit/settle.
- Concurrency discipline (document in `DECISIONS.md`, enforce in code): lease acquisition is an atomic `UPDATE ... WHERE status='PENDING' AND ...` guarded by `BEGIN IMMEDIATE`, so exactly one worker wins a contended subtask; claim submission is an independent insert that does not block on other workers; SQLite `busy_timeout` set to a documented value; all writers retry on `SQLITE_BUSY` up to a bounded count.

### 4.2 Harness
`legion run-cluster --workers N --task <fixture> --workdir <dir>` spawns the coordinator and N worker subprocesses (stdlib `multiprocessing` or `subprocess`), runs until the task settles or a timeout, then prints the same per-identity settlement statement as `demo`.

### 4.3 Tests
`tests/test_multiprocess.py` (may be marked slow): spawn the coordinator + 4 worker processes on the short fixture; assert the task settles, conservation holds to the µ, no subtask is completed by two authors (no double-lease), and total BOND in == out. A contention test: two workers targeting the same single available subtask — exactly one acquires the lease, the other backs off cleanly.

---

## 5. Light-client re-derivation

**Goal:** the cheapest possible audit — a second program, sharing no code path with the live engine's write side, that reconstructs the entire economic state from the immutable log and must agree to the µ. This is the determinism work's payoff and the first concrete step toward trust-minimization.

### 5.1 The replayer
`tools/lightclient.py`: given a `ledger.db` (read-only connection), independently:
1. Replays `transfers` in `id` order to reconstruct every identity and pseudo-account balance; asserts they equal the live `identities` / `pseudo_accounts` tables.
2. For every `CLOSED` task with `settlement_applied=1`, rebuilds the snapshot from `claims` + `claim_cite_overrides` + `fetches`, calls `legion.settlement.settle(snapshot, version=task.settlement_version)`, and asserts the resulting transfer multiset equals the `PAYOUT_*`/`BURN`/`FEE_REFUND` transfers actually recorded for that task.
3. Re-runs the deterministic admission checks (signature, ref-tag spans, fetch-gating) for every `ADMITTED` claim and asserts none should have been rejected. (The semantic verifier verdict is *not* re-derivable offline; record this boundary explicitly — the light client checks everything deterministic and flags the verifier as the remaining trust assumption.)

### 5.2 CLI + test
`legion audit <ledger.db>` runs the replayer and prints PASS with counts (identities reconciled, tasks re-settled, claims re-checked) or the first divergence. `tests/test_lightclient.py`: run a full demo, then audit its ledger in a **separate process** (subprocess, fresh interpreter) and assert PASS; then mutate one recorded payout amount in a copy of the DB and assert the audit FAILs with a clear diff. The cross-process requirement is the point — it proves re-derivation doesn't depend on in-memory state.

---

## 6. Milestones (strict order; each ends green)

- **P3-M1 — Carryover** (§2): CLI guard + test; §4A sim model, experiments, and eight statistical goldens; long-document fixture.
- **P3-M2 — Regime study** (§3): new metrics, the sweep driver, `regime.json` in CI on the fake path, `REGIME_FINDINGS.md`.
- **P3-M3 — Multi-process** (§4): cluster harness, concurrency discipline, double-lease and contention tests.
- **P3-M4 — Light client** (§5): replayer, `audit` CLI, cross-process pass + tamper-detection tests.

## 7. Invariants (carried forward — all must still hold)
Conservation to the µ per task; settlement purity and cross-process byte determinism per version; append-only triggers; fetch-gated citations and ADMITTED-only serving; keep-fraction invariance; pseudo-account solvency and global conservation; steering-v2 ring-capture bound and relevance scoping; verifier injection-resistance (deterministic quote check load-bearing); no floats in `legion` settlement; no new claim kinds; `tools/sim/` imports nothing from `legion`.

## 8. Definition of done
`pytest -q` green with ≥ 14 new tests across §2–§5 (including the eight §4A.6 goldens and the cross-process audit); `python -m legion.cli demo` and `legion demo` both work; `legion eval --sweep` writes `regime.json` on the fake-LLM path in CI; `legion run-cluster` settles a task across 4 worker processes; `legion audit` passes on a fresh ledger in a separate process and fails on a tampered one; `docs/REGIME_FINDINGS.md` states the measured crossover (or its honest absence); `docs/DECISIONS.md` updated for every judgment call; README gains a "Phase 3" section (≤ 20 lines) covering the regime finding, multi-process run, and audit command.

## 9. A note on honesty of results
The single most valuable output of this phase is a truthful answer to "does coordinated multi-agent reasoning beat a single agent, and when?" The Phase 2 eval already demonstrated the discipline of reporting a loss. Preserve it: if the sweep shows the protocol never wins at achievable scales, that is the finding, and it belongs in `REGIME_FINDINGS.md` stated plainly. The mechanism's correctness (conservation, determinism, incentive-compatibility, auditability) is independent of whether it is yet *economical*, and Phase 3 is allowed to conclude "correct but not yet economical at scale X" — that is a real, publishable result, not a failure of the work.