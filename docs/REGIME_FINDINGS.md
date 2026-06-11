# Regime Findings (Phase 3.1 — corrected baselines)

> **Supersedes the Phase 3 numbers.** The Phase 3 eval compared against a
> baseline that was false twice over: the fake stub's baseline branch returned
> the answer key verbatim in one call, and a one-call oracle is an
> infinite-context fiction. Every Phase 3 `cost_ratio` measured the protocol
> against that fiction and penalized it. The figures below are measured
> against honest, answer-key-free runners; the prompt-leak invariant test now
> makes the old failure impossible.

Runners, identical grader (`solved = all gold facts in answer`):
- **protocol** — LLM workers + hardened verifier + settlement.
- **baseline_iterative** — the fair comparator: a competent agent loop, one
  extraction per document + one synthesis, ≤ n_docs+2 calls, no privileged access.
- **baseline_onecall** — the infinite-context oracle reference; recorded
  `INFEASIBLE` (never truncated) when the corpus exceeds
  `VSCP_CONTEXT_WINDOW_TOKENS` (default 128k).

Headline metric: **token_ratio = protocol_tokens / iterative_tokens** (per-call
ratios are a legacy secondary; per-call pricing structurally penalizes any
decomposed system). Fake-path (deterministic) data; tokens estimated at 4
chars/token until provider usage replaces them on real runs.

| docs (per-doc size) | workers | token_ratio vs iterative | vs onecall | onecall   |
|---------------------|---------|--------------------------|------------|-----------|
| short (~1 kB)       | 1       | 2.09                     | 2.38       | feasible  |
| short               | 8       | 6.50                     | 7.38       | feasible  |
| long (~22 kB × 6)   | 1       | 1.08                     | 1.11       | feasible  |
| long                | 8       | 2.28                     | 2.36       | feasible  |
| **xl (~80 kB × 8)** | **1**   | **1.02**                 | —          | **INFEASIBLE** |
| xl                  | 2       | 1.15                     | —          | INFEASIBLE |
| xl                  | 8       | 1.90                     | —          | INFEASIBLE |

All cells: protocol and iterative baseline both solve. Full grid (incl.
steering-reader fan-out and reuse counts): `legion eval --sweep` → `regime.json`.

## What the corrected data shows

1. **The decisive regime exists and is now measured.** On the ~640 kB XL
   corpus the one-call oracle is *infeasible* — not expensive, impossible —
   while the protocol settles and solves. Against the honest iterative
   baseline the protocol's token overhead at task width 1 is **2.1%**
   (1.02×): effective parity, with verified evidence-bound claims, exact
   provenance payouts, and an independently auditable ledger as the surplus.
2. **No strict token win yet, stated plainly.** In no cell does token_ratio
   drop below 1.0. The protocol's verifier and steering machinery cost real
   tokens, and a lone competent agent reading each document once is the same
   token floor the protocol pays. The honest claim is *parity at scale with
   verification included*, not *cheaper*.
3. **Overhead is the price of width, not of size.** token_ratio rises with
   worker count (triage, FAIL traffic, verifier checks per claim) and falls
   with document size in every class. Worker count should be sized to task
   width; parallelism buys wall-clock epochs and steering fan-out (0 → 3
   eligible readers), not token savings on a single question.
4. **Where a strict win should appear** (untested here, Phase 4 candidates):
   amortization — multiple questions over one admitted claim graph (the
   iterative baseline re-reads the corpus per question; the protocol's
   admitted FACTs are reusable), and corpora past what even an agentic
   context-window can practically page through.

**Conclusion: the protocol is feasibility-dominant over the one-call oracle at
XL scale and token-parity (1.02×) with a competent iterative agent at task
width 1, with verification and auditability as the differential. A strict
token win was not observed at any tested scale — that remains the honest
finding.**
