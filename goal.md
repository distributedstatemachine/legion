# Verified Shared-Context Protocol — Proof-of-Concept Specification

**Version:** 1.0
**Audience:** an autonomous coding agent. This document is self-contained; do not ask for clarification — where this spec is silent, choose the simplest option consistent with the invariants in §9 and record the decision in `DECISIONS.md`.
**Goal:** a working local prototype of a decentralized knowledge-market protocol derived from DELM (arXiv 2606.10662): workers claim subtasks from a task market, submit compact evidence-bound claims into an admission-gated shared context, and a deterministic settlement engine splits a bounty across the provenance graph at task close.

---

## 1. Scope

### In scope (must build)
1. A persistent **settlement layer** (SQLite): identities, stake, task registry with leases, bounty escrow, payout ledger.
2. An **ephemeral task ledger** per task: append-only claim DAG, evidence store, fetch log, admission gate.
3. **Deterministic admission checks** (ref-tag span verification) plus a pluggable semantic verifier (mock + optional LLM).
4. A **deterministic settlement engine** implementing the three-pool split with flat backward flow (exact math in §6).
5. A **worker runtime** with two interchangeable agent backends: a scripted deterministic solver (for tests/demo) and an LLM-backed solver (optional, behind an env flag).
6. A **challenge stub**: citation-materiality and under-citation challenges resolvable by the verifier interface, with bond slashing.
7. A CLI demo and a full test suite proving the invariants in §9.

### Out of scope (must NOT build)
- Real consensus, networking between machines, or any blockchain integration. All "nodes" are local processes sharing the SQLite DB; a single writer process (the **coordinator**) serializes ledger writes.
- Token economics beyond an integer balance column. No wallets, no chains, no cryptography beyond Ed25519 signatures and SHA-256 hashing.
- The hypergraph/Shapley settlement (record derivation structure per §5.4, but settle with flat flow only).
- Sybil resistance, reputation, or cross-task identity logic.

---

## 2. Tech stack (fixed — do not substitute)

- Python 3.11+, single repo, `pyproject.toml`, installable as `legion`.
- SQLite via the stdlib `sqlite3` (WAL mode). One DB file per run: `ledger.db`.
- `pynacl` for Ed25519 keypairs and signatures; `hashlib.sha256` for content addressing.
- `pytest` + `hypothesis` for tests. `click` for the CLI.
- No web framework. Processes communicate only through the DB (coordinator polls).
- All money is integer **micro-bounty units (µ)**: a task's bounty is `1_000_000 µ`. No floats anywhere in settlement code.

Repository layout:

```
legion/
  __init__.py
  crypto.py        # keys, signing, content hashes
  store.py         # SQLite schema + append-only accessors
  tasks.py         # task registry, leases, dependency DAG
  claims.py        # claim construction, ref-tags, fetch log
  admission.py     # deterministic checks + verifier interface
  settlement.py    # the engine of §6 (pure function, no I/O)
  challenges.py    # materiality / under-citation games
  workers/
    base.py        # worker loop (lease → read → solve → submit)
    scripted.py    # deterministic synthetic-task solver
    llm.py         # optional OpenAI-compatible backend
  coordinator.py   # single-writer process: admissions, epochs, close
  cli.py           # `legion demo`, `legion settle`, `legion inspect`
tests/
docs/DECISIONS.md
```

---

## 3. Synthetic task family (used by demo and tests)

Implement the **fact-chain task** so the whole protocol is exercisable without an LLM:

- A generator `make_task(K, D, seed)` produces: `K` facts `F0..F(K-1)`; fact `Fi` is a random 12-word sentence embedded in a synthetic source document `doc_i.txt` (~600 words of seeded lorem text), at a random offset. Each fact has `D` decoy sentences in the same document that look plausible but are marked wrong by a hidden answer key.
- Solving = submitting an `ANSWER` claim whose body lists all `K` fact sentences verbatim, each supported by a ref-tag into its source document.
- The scripted worker "discovers" facts by querying an oracle function with rate-limited attempts (mirrors the discovery race; one oracle query per turn, uniform over unruled-out candidates). The oracle lives in test/demo code only — the protocol never sees it.

Source documents go into the evidence store (§5.3) at task creation; their hashes are listed in the task record.

---

## 4. Settlement layer

### 4.1 Tables (append-only unless noted)

- `identities(pubkey PK, balance_µ INT)` — balance is mutable; every mutation must be mirrored by a row in `transfers`.
- `transfers(id PK, epoch, from_pubkey, to_pubkey, amount_µ, reason)` — reasons: `ESCROW, FEE, FEE_REFUND, PAYOUT_FINISHER, PAYOUT_DERIVATION, PAYOUT_STEERING, BOND, SLASH, BURN`.
- `tasks(task_id PK, spec_hash, bounty_µ, status, created_epoch, closed_epoch)` — status ∈ `OPEN, CLOSED, EXPIRED`.
- `subtasks(subtask_id PK, task_id, deps_json, status, lease_holder, lease_expiry_epoch)` — status ∈ `PENDING, LEASED, DONE`.
- `claims`, `fetches` — see §5.
- Time is a logical **epoch** counter advanced by the coordinator (one epoch ≈ one polling cycle). No wall-clock logic in protocol code.

### 4.2 Leases

- A worker may lease a subtask iff all `deps` are `DONE`, it posts a lease bond of `5_000 µ`, and it holds no other live lease on the same task.
- Lease duration: 10 epochs. Expiry returns the subtask to `PENDING` and **burns** the bond. Completing the subtask (an admitted claim referencing it) refunds the bond.

---

## 5. Task ledger

### 5.1 Claim record

```json
{
  "claim_id": "sha256 of canonical body",
  "task_id": "...",
  "subtask_id": "... | null",
  "author": "pubkey",
  "kind": "FACT | FAIL | CONSTRAINT | ANSWER",
  "body": "<= 600 chars, the gist",
  "evidence": [{"doc_hash": "...", "ref": {"head": "5 words", "tail": "5 words"}}],
  "cites": ["claim_id", ...],
  "derivations": [["claim_id", ...], ...],
  "sig": "ed25519 over canonical body",
  "epoch_submitted": 0,
  "status": "PENDING | ADMITTED | REJECTED"
}
```

Canonicalization: JSON with sorted keys, UTF-8, no whitespace; `claim_id`, `sig`, `epoch_submitted`, `status` excluded from the hashed body.

### 5.2 Fetch log

Reading is metered: a worker obtains a claim body or an evidence document only through `store.fetch(reader_pubkey, object_id)`, which appends to `fetches(reader, object_id, epoch)`. Workers must not share objects out of band — the scripted workers must be written so all reads go through `fetch`. This log is the sole input to the steering payout and to under-citation challenges.

### 5.3 Evidence store

Content-addressed directory `evidence/` keyed by SHA-256. `put` returns the hash; `get` is metered via the fetch log. Documents are immutable.

### 5.4 Derivations (record-only)

`derivations` is a list of alternative parent-sets (a directed hyperedge per alternative). For this PoC it is **stored and validated for shape** (every id must appear in `cites`; at least one alternative if non-empty) but **never used by settlement**. It exists so the data model doesn't have to migrate when hypergraph attribution ships later.

---

## 6. Admission gate

Run by the coordinator on each `PENDING` claim, in submission order:

1. **Signature + shape**: valid sig, body ≤ 600 chars, ≤ 8 evidence refs, ≤ 16 cites, all cited claim_ids `ADMITTED`, all doc hashes present in the evidence store, author actually fetched every cited claim (check `fetches`) — reject otherwise with a machine-readable reason.
2. **Ref-tag check (deterministic)**: for each evidence ref, `head` and `tail` must each appear verbatim in the referenced document, with `head` ending at an offset strictly before `tail` begins, and the enclosed span ≤ 1200 chars. Pure string matching; no NLP.
3. **Semantic check (pluggable)** via `Verifier.supports(claim, spans) -> bool`:
   - `MockVerifier` (default): for synthetic tasks, returns True iff the claim body's quoted fact sentence appears inside one of its evidence spans and is not a decoy per the task's answer key (the key is passed to the verifier in demo mode only).
   - `LLMVerifier` (optional, `VSCP_LLM=1`): single completion asking "is every assertion in BODY supported by SPANS — answer YES/NO"; any non-YES → reject.
4. **Fee**: deduct `ADMISSION_FEE = 10_000 µ` from the author at submission regardless of outcome (reason `FEE`).
5. Admitted claims are immutable and immediately visible. The **first** admitted `ANSWER` that the verifier accepts as solving the task spec freezes the ledger and schedules settlement for `epoch + CHALLENGE_WINDOW` (default 5 epochs).

---

## 7. Settlement engine (exact, deterministic)

`settle(ledger_snapshot) -> list[Transfer]` must be a **pure function**: same snapshot bytes → identical transfer list. Property-tested (§9).

Constants: `ALPHA = 350_000 µ`, `BETA = 450_000 µ`, `GAMMA = 200_000 µ` (sum = bounty), `DELTA_NUM/DELTA_DEN = 1/2`.

1. **Finisher premium**: `ALPHA` to the answer's author.
2. **Derivation pool** (`BETA`), flat backward flow over `cites` from the answer:
   - Maintain integer `inflow[claim_id]`; start `inflow[answer] = BETA`.
   - Process claims in reverse-topological order from the answer. For node n with parents `P = cites(n)`: if `P` is empty, n keeps all inflow. Else n keeps `inflow - pass`, where `pass = inflow * DELTA_NUM // DELTA_DEN`; distribute `pass` among parents by equal integer division, assigning the remainder one µ at a time to parents in ascending `claim_id` order (largest-remainder with deterministic tie-break).
   - Claims outside the answer's ancestry receive nothing from this pool.
3. **Steering pool** (`GAMMA`): let `A` = set of authors of claims with positive derivation flow, plus the finisher. For each admitted `FAIL`/`CONSTRAINT` claim c, weight `w(c) = |readers(c) ∩ A|` from the fetch log (count each reader once). Split `GAMMA` proportionally by `w` with the same largest-remainder rule, tie-break ascending `claim_id`. If all weights are zero, `BURN` the pool.
4. **Fee refunds**: for every admitted claim whose total earned flow (derivation + steering) ≥ `ADMISSION_FEE`, transfer `FEE_REFUND` of the full fee to its author. Rejected claims never refund.
5. **Conservation**: `ALPHA + BETA + GAMMA` paid out + burned must equal the bounty exactly; assert this inside `settle`.

---

## 8. Challenges (stub, but functional)

During the challenge window any identity may file:
- **Under-citation**: "answer/claim X used fetched-but-uncited claim Y." Resolver: deterministic — upheld iff `fetches` shows the author fetched Y before submitting X and X's body contains ≥ 6 consecutive words also present in Y's body. Upheld → Y is inserted into X's `cites` for settlement, challenger receives a `25_000 µ` bond slashed from X's author.
- **Materiality**: "citation of Y by X contributed nothing." Resolver: the `Verifier` is asked whether X's body is fully supported without Y's span; YES → citation removed for settlement, X's author slashed `25_000 µ` to the challenger; NO → the challenger's own `25_000 µ` bond goes to X's author.
Settlement runs on the post-challenge graph.

---

## 9. Invariants and acceptance tests (the agent's definition of done)

All must pass under `pytest -q`; items marked (H) are `hypothesis` property tests over randomized ledgers.

1. **Conservation (H)**: for any closed task, Σ payouts + Σ burns = bounty; no identity balance ever goes negative.
2. **Determinism (H)**: serializing the ledger, reloading, and re-running `settle` yields byte-identical transfer lists across 3 runs and across two different process invocations.
3. **Append-only**: any UPDATE/DELETE on `claims`, `transfers`, or `fetches` outside whitelisted status transitions raises; enforced with SQLite triggers and tested.
4. **Keep-fraction invariance (H)**: adding an extra citation to a synthetic node never changes that node's own derivation payout (the structural ring-resistance property).
5. **Uncited-earns-nothing**: an admitted claim outside the answer ancestry with zero productive FAIL readership receives 0 µ and no fee refund.
6. **Ref-tag soundness**: fuzz 200 mutated evidence refs (reordered head/tail, off-by-one truncation, cross-document spans); all must be rejected deterministically.
7. **Fetch-gating**: a claim citing an unfetched claim is rejected; a worker that bypasses `fetch` cannot produce admissible citations (enforce by making bodies retrievable only through `fetch`).
8. **End-to-end demo**: `legion demo --workers 6 --honest 4 --hoarders 2 --K 6 --D 3 --seed 42` must (a) close the task within 200 epochs, (b) produce a settlement where mean honest payoff > mean hoarder payoff, and (c) print a per-identity statement reconciling to invariant 1. Seeded, so the assertion is exact and committed as a golden file.
9. **Challenge round-trip**: a scripted under-citation scenario (hoarder finisher omits a fetched claim) is detected by a scripted challenger, the citation is inserted, and the omitted author's payout strictly increases vs. the unchallenged counterfactual.
10. **LLM path smoke test** (skipped when `VSCP_LLM` unset): one claim admission via the LLM verifier completes and both YES and NO branches are reachable with crafted inputs.

---

## 10. Milestones (implement strictly in order; each ends with green tests)

- **M0 — skeleton**: repo, schema, crypto, append-only triggers. Tests: 1 (balances), 3.
- **M1 — claims + admission**: evidence store, ref-tags, fetch log, MockVerifier, coordinator loop. Tests: 6, 7.
- **M2 — settlement**: pure engine + golden vectors (hand-compute one 7-claim example in a test, with the integer remainders worked out in comments). Tests: 1, 2, 4, 5.
- **M3 — workers + demo**: scripted solver, leases, CLI, end-to-end. Test: 8.
- **M4 — challenges**: both games. Test: 9.
- **M5 — LLM backend (optional)**: `workers/llm.py`, `LLMVerifier`. Test: 10. If no API key is available, implement against the interface and leave the test skipped.

Stretch (only after M5, never at its expense): a `--priority-weight` settlement flag implementing first-publisher novelty discount over recorded `derivations`, plus a regression showing duplicate claims earn < 10% of originals under it.

## 11. Non-goals reminder for the agent

Do not add: networking, async frameworks, ORMs, docker, configuration systems, or any blockchain library. Do not "improve" the settlement math — its exact integer semantics are the deliverable. If a test in §9 seems too strict, the test is right and the implementation is wrong.