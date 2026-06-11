# Implementation Decisions

- Admission fees are deducted inside `Store.submit_claim`. This satisfies the requirement that the fee is charged at submission regardless of admission outcome and keeps rejected claims accountable even before the coordinator polls.
- Challenge graph changes are append-only citation overrides in `claim_cite_overrides`, not mutations to admitted `claims`. Settlement snapshots apply the latest override so the challenge graph is post-resolution while the base claim ledger stays immutable.
- Payouts are transferred from pseudo escrow accounts such as `ESCROW:<task_id>`. The escrow debit from the sponsor is recorded at task creation, but pseudo accounts are not identities because the spec only requires identity balances.
- The synthetic fact task uses independent fact subtasks plus an answer subtask depending on all facts. This exercises leases and the dependency DAG while allowing multiple local workers to race on discovery.
- The optional LLM path is implemented behind the verifier interface with an OpenAI-compatible HTTP hook. Tests skip it when `VSCP_LLM` is unset and use a crafted completion function when enabled.
