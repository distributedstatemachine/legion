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
  runs LLM workers against bundled multi-document QA fixtures and a single-agent
  baseline, writing `report.json` (cost/accuracy; uses the real endpoint when
  `VSCP_LLM=1`, a deterministic gold-fact stub otherwise).

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
