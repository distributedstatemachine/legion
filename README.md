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

## Known limitations / TODO

- **Steering collusion**: γ weight is raw readership among productive authors,
  so a ring member fetching a partner's FAIL still counts; needs a usefulness
  rule stronger than readership (see TODO in `tests/test_adversaries.py`).
- **LLM verifier is prompt-injectable**: claim bodies flow into the prompt
  unquoted. Before any real LLM task: run the deterministic ref-tag check
  first, structurally quote the body, and build an adversarial suite for the
  verifier itself.
- **Single process by design**: all "nodes" share one SQLite DB via the
  coordinator. Distribution is deliberately deferred until the verifier
  economics are validated on real multi-document QA tasks.
- The first-publisher novelty discount (`--priority-weight`) from the spec's
  stretch goal is not implemented.
