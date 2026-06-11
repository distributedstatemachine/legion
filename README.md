# Legion — Verified Shared-Context Protocol (PoC)

A working local prototype of a decentralized knowledge-market protocol derived
from DELM: workers lease subtasks from a task market, submit compact
evidence-bound claims into an admission-gated shared context, and a
deterministic settlement engine splits a `1_000_000 µ` bounty across the
provenance graph at task close. Full specification: [`goal.md`](goal.md);
implementation decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Quick start

```bash
uv sync                 # or: pip install -e . --group dev
uv run pytest -q        # full invariant suite
uv run legion demo --workers 6 --honest 4 --hoarders 2 --K 6 --D 3 --seed 42
```

The demo prints a per-identity statement reconciling to the conservation
invariant, including the steering-pool (γ) flow:

```
bounty_paid_or_burned 1000000
steering_paid 200000
steering_burned 0
mean_honest_delta 250000
mean_hoarder_delta -10000
```

Other commands: `legion settle <workdir> <task_id>` (re-derive a settlement as
pure JSON from the ledger) and `legion inspect <workdir>` (balances and tasks).

## Layout

| Path | What it is |
| --- | --- |
| `legion/store.py` | SQLite schema, append-only triggers, pseudo-account ledger, metered `fetch` |
| `legion/admission.py` | Deterministic ref-tag checks (minimal-span), `MockVerifier` / `LLMVerifier` |
| `legion/settlement.py` | Pure three-pool split (α finisher / β derivation / γ steering), exact integer math |
| `legion/challenges.py` | Under-citation and materiality games, resolved atomically |
| `legion/tasks.py` | Synthetic fact-chain task family, leases, dependency DAG |
| `legion/workers/` | Worker loop, scripted solver (honest + racing hoarder), optional LLM backend |
| `legion/coordinator.py` | Single-writer loop: admissions, epochs, settlement |
| `tests/` | Invariant suite incl. golden demo and an adversary zoo (`test_adversaries.py`) |

## Mechanism in one paragraph

Admitted `FACT` claims earn from the derivation pool via flat backward flow
over `cites` from the winning `ANSWER`; verified negative results (`FAIL`
claims, gated against the answer key's decoys) earn from the steering pool by
readership among productive authors; the finisher takes the premium. Reads are
metered through a fetch log, citations of unfetched claims are rejected, and
under-citation/materiality challenges rewrite the settlement graph through
append-only cite overrides.

## Phase 2

- **Steering v2** (`settle(..., version=2)`, the default): steering weight is no
  longer raw readership. Each productive reader holds a normalized 1,000,000-unit
  endorsement budget, split across the FAIL/CONSTRAINT claims that verifiably
  *informed* its work (fetched before a positively-flowing claim whose docs
  overlap the FAIL's docs). A colluding reader caps its ring's capture at one
  endowment no matter how many ring FAILs it fetches; `version=1` preserves the
  old behavior and old goldens.
- **Settlement-equivalence harness**: `tools/sim/sim.py` (independent float
  reference) and `legion.settlement` must agree to the µ on every scenario in
  `tests/scenarios/` — two implementations of the mechanism auditing each other.
- **Hardened LLM verifier**: nonce-fenced data blocks, strict JSON output, and a
  deterministic quote check (verbatim span substring, 10–300 chars) that holds
  even against a fully compromised model. Adversarial suite in
  `tests/test_verifier_injection.py`.
- **Real-task eval**: `legion eval --tasks corpus/tasks --workers 4 --baseline`
  runs LLM workers against bundled multi-document QA fixtures (including the
  ~36 kB `deep_archive` long-document task) and a single-agent baseline, writing
  `report.json` with cost/accuracy and the `steering_readers` concentration
  metric (uses the real endpoint when `VSCP_LLM=1`, a deterministic gold-fact
  stub otherwise).

## Phase 3

- **Regime study**: `legion eval --sweep` runs the document-length × worker-count
  grid and writes `regime.json`. The finding (`docs/REGIME_FINDINGS.md`): no cost
  crossover at tested scales — per-call cost_ratio 7–29×, but context volume
  (char_ratio) collapses from 3.0× on ~1 kB docs to **1.10×** on the ~130 kB
  six-document task. Correct but not yet economical; the crossover regimes are
  >context-window corpora and cross-task claim reuse.
- **Research sim** (`tools/sim/model.py`): seven-strategy agent model
  (HONEST/HOARDER/SPAMMER/RING_SYBIL/RING_BENEF/POISONER/HYBRID) with flat +
  sampled-Shapley settlement and eight statistical goldens — including: a lone
  poisoner is profitable under naive Shapley coverage but not under flat
  backward flow, and an all-hybrid population earns ≤ 0.75× all-honest welfare.
- **Multi-process**: `legion run-cluster --workers 4 --workdir <dir>` runs N
  worker processes + 1 coordinator against one WAL ledger; lease acquisition is
  atomic (`BEGIN IMMEDIATE`), contention has exactly one winner.
- **Light client**: `legion audit <workdir>` re-derives every balance (via MINT
  genesis transfers), every settlement, and every deterministic admission check
  from the immutable log in a separate process; tampered payouts fail with a
  named divergence. Remaining trust assumption: the semantic verifier verdict.

## Phase 3.1 — honest baselines + OpenRouter

The Phase 3 baseline was false (it was handed the answer key in one call);
`docs/REGIME_FINDINGS.md` is rewritten against two honest runners — a one-call
infinite-context oracle (marked infeasible past the context window) and a fair
iterative agent — with `token_ratio` as the headline. Corrected result: on the
~640 kB XL fixture the one-call baseline is infeasible while the protocol
solves at **1.02×** the iterative baseline's tokens.

Real-endpoint runs (OpenRouter by default):

```bash
export VSCP_LLM=1 OPENROUTER_API_KEY=sk-or-...   # or VSCP_LLM_API_KEY / OPENAI_API_KEY
export VSCP_LLM_MODEL=openai/gpt-4o-mini          # VSCP_VERIFIER_MODEL overrides for the verifier
legion eval --dry-run                             # plan + cost estimate, no calls
legion eval --tasks corpus/tasks --workers 4      # real run; provider usage drives costing
```

Optional: `VSCP_LLM_URL`, `VSCP_LLM_REFERER`/`VSCP_LLM_TITLE` (OpenRouter
headers), `VSCP_CONTEXT_WINDOW_TOKENS` (128000), `VSCP_COST_PER_1K_TOKENS`
(0.15), `VSCP_MAX_TOTAL_LLM_CALLS` (500, hard cap, partial report on exceed).

## Known limitations / TODO

- **Steering collusion** (fixed in Phase 2): settlement v2's reader-normalized,
  relevance-scoped weights bound a ring's steering capture; v1's raw-readership
  rule remains available behind `version=1` for comparison.
- **LLM verifier injection** (hardened in Phase 2): deterministic checks run
  first, untrusted content is structurally fenced, and affirmative verdicts
  require a verbatim span quote. The residual risk is a model that quotes real
  span text while reasoning wrongly — that is a model-quality issue, not an
  injection channel.
- **Single process by design**: all "nodes" share one SQLite DB via the
  coordinator. Distribution is deliberately deferred until the verifier
  economics are validated on real multi-document QA tasks (Phase 3).
- The first-publisher novelty discount (`--priority-weight`) from the spec's
  stretch goal is not implemented.
