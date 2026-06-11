# Regime Findings (Phase 3, §3)

Measured on the fake-LLM path (deterministic; real-endpoint numbers will move,
the structure should not). Grid: document class × worker count, all cells
solved. Full data: `legion eval --sweep` → `regime.json`.

| docs  | workers | cost_ratio (calls) | char_ratio (context) | steering readers | reuse |
|-------|---------|--------------------|----------------------|------------------|-------|
| short | 1       | 7.0                | 3.03                 | 0                | 0     |
| short | 2       | 9.0                | 3.95                 | 0                | 0     |
| short | 4       | 13.0               | 5.78                 | 2                | 6     |
| short | 8       | 21.0               | 9.47                 | 2                | 18    |
| long  | 1       | 15.0               | 1.10                 | 0                | 0     |
| long  | 2       | 17.0               | 1.27                 | 0                | 0     |
| long  | 4       | 21.0               | 1.62                 | 2                | 2     |
| long  | 8       | 29.0               | 2.33                 | 3                | 6     |

## The headline, stated plainly

**No cost crossover was found at the scales tested.** The protocol never beats
the single-agent baseline on either metric in this grid. That is the finding;
it is not buried.

Two distinct results inside that:

1. **Per-call pricing is structurally unwinnable.** The baseline is one call
   by construction; the protocol pays per extraction, triage, synthesis, and
   verifier check (7–29 calls). Under flat per-call pricing the protocol can
   never win — this is an artifact of the pricing model, not evidence about
   coordination. Any real conclusion must be drawn from token/context volume.
2. **Context volume tells the interesting story.** On short documents the
   protocol ships 3–9.5× the baseline's characters. On the ~130 kB
   six-document task it collapses to **1.10×** at one worker — near parity —
   because the documents dominate both sides and the protocol reads each
   document approximately once. The protocol's overhead (verifier spans,
   triage, synthesis) is roughly constant while the baseline's cost grows
   linearly with corpus size.

## Where the protocol would be expected to win, and does the data support it?

The expectation: long multi-document tasks where the baseline's context cost
grows with the corpus and cross-worker reuse is high. The data **supports the
trend but not yet the crossover**: char_ratio falls from 3.03 → 1.10 as
documents grow ~20×, steering eligibility fans out from the finisher alone (0
counted pre-answer in single-worker cells) to 3 distinct readers at 8 workers,
and redundant-work-avoided becomes positive exactly where parallelism does.
Extrapolating the trend, the crossover regimes are: (a) corpora larger than a
single context window, where the baseline is not merely costlier but
*infeasible*; (b) multiple questions over one corpus, where admitted claims
amortize across tasks — neither is exercised by the current single-question
fixtures. Per-worker char_ratio also rises with worker count (more triage and
verifier calls), so worker count should be sized to task width, not maximized.

**Conclusion: correct but not yet economical at the scales tested.** The
mechanism's correctness (conservation, determinism, ring-resistance,
auditability) stands independently; the economics require either >context-
window corpora or cross-task claim reuse, both Phase 4 candidates.
