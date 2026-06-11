"""Light-client re-derivation: reconstruct the entire economic state from the
immutable log and assert it agrees with the live tables to the µ.

Shares no code path with the live engine's write side: it opens the DB
read-only, replays `transfers` with its own arithmetic, rebuilds settlement
snapshots with its own SQL, and re-runs only *pure* functions from the engine
(`legion.settlement.settle`, `legion.admission.resolve_ref_span`,
`legion.crypto.*`).

Trust boundary, stated explicitly: the semantic verifier's verdict (Mock or
LLM) is NOT re-derivable offline, and neither is the lease state observed at
admission time. The light client checks everything deterministic - balances,
settlement math, signatures, hashes, ref-tag spans, fetch-gating - and flags
the verifier as the remaining trust assumption.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legion import crypto
from legion import claims as claim_helpers
from legion.admission import resolve_ref_span
from legion.settlement import settle

PSEUDO_PREFIXES = ("ESCROW:", "BOND:", "FEE_POOL", "BURN")
BOUNTY_REASONS = {"PAYOUT_FINISHER", "PAYOUT_DERIVATION", "PAYOUT_STEERING", "BURN"}


@dataclass
class AuditResult:
    identities_reconciled: int = 0
    pseudo_reconciled: int = 0
    tasks_resettled: int = 0
    claims_rechecked: int = 0
    divergences: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.divergences

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASS: {self.identities_reconciled} identities + "
                f"{self.pseudo_reconciled} pseudo accounts reconciled, "
                f"{self.tasks_resettled} tasks re-settled, "
                f"{self.claims_rechecked} admitted claims re-checked.\n"
                "Trust boundary: semantic verifier verdicts and lease state at "
                "admission are not re-derivable offline."
            )
        return "FAIL: " + self.divergences[0]


def _is_pseudo(account: str | None) -> bool:
    return account is not None and account.startswith(PSEUDO_PREFIXES)


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _replay_balances(conn: sqlite3.Connection, result: AuditResult) -> None:
    balances: dict[str, int] = defaultdict(int)
    pseudo: dict[str, int] = defaultdict(int)
    for row in conn.execute(
        'SELECT from_pubkey, to_pubkey, "amount_µ" AS amount FROM transfers ORDER BY id'
    ):
        src, dst, amount = row["from_pubkey"], row["to_pubkey"], int(row["amount"])
        if src is not None:
            (pseudo if _is_pseudo(src) else balances)[src] -= amount
        if dst is not None:
            (pseudo if _is_pseudo(dst) else balances)[dst] += amount

    for row in conn.execute('SELECT pubkey, "balance_µ" AS balance FROM identities'):
        replayed = balances.pop(row["pubkey"], 0)
        if replayed != int(row["balance"]):
            result.divergences.append(
                f"identity {row['pubkey'][:10]} live balance {row['balance']} != replayed {replayed}"
            )
        result.identities_reconciled += 1
    for pubkey, replayed in balances.items():
        if replayed != 0:
            result.divergences.append(f"replayed balance for unknown identity {pubkey[:10]}")
    for row in conn.execute('SELECT account, "balance_µ" AS balance FROM pseudo_accounts'):
        replayed = pseudo.pop(row["account"], 0)
        if replayed != int(row["balance"]):
            result.divergences.append(
                f"pseudo {row['account']} live balance {row['balance']} != replayed {replayed}"
            )
        result.pseudo_reconciled += 1
    for account, replayed in pseudo.items():
        if replayed != 0:
            result.divergences.append(f"replayed balance for untracked pseudo {account}")


def _rebuild_snapshot(conn: sqlite3.Connection, task_row: sqlite3.Row) -> dict[str, Any]:
    task_id = task_row["task_id"]
    claims: dict[str, Any] = {}
    for row in conn.execute(
        "SELECT * FROM claims WHERE task_id = ? AND status = 'ADMITTED' ORDER BY claim_id",
        (task_id,),
    ):
        override = conn.execute(
            "SELECT cites_json FROM claim_cite_overrides WHERE claim_id = ? ORDER BY id DESC LIMIT 1",
            (row["claim_id"],),
        ).fetchone()
        claims[row["claim_id"]] = {
            "claim_id": row["claim_id"],
            "task_id": row["task_id"],
            "author": row["author"],
            "kind": row["kind"],
            "body": row["body"],
            "evidence": json.loads(row["evidence_json"]),
            "cites": json.loads(override["cites_json"] if override else row["cites_json"]),
            "epoch_submitted": row["epoch_submitted"],
            "status": row["status"],
        }
    fetches = [
        dict(row)
        for row in conn.execute("SELECT reader, object_id, epoch FROM fetches ORDER BY id")
    ]
    return {
        "task_id": task_id,
        "bounty_µ": int(task_row["bounty_µ"]),
        "answer_claim_id": task_row["answer_claim_id"],
        "settlement_version": task_row["settlement_version"] or 2,
        "claims": claims,
        "fetches": fetches,
    }


def _recheck_settlements(conn: sqlite3.Connection, result: AuditResult) -> None:
    expected_refunds: Counter = Counter()
    for task_row in conn.execute(
        "SELECT * FROM tasks WHERE status = 'CLOSED' AND settlement_applied = 1 ORDER BY task_id"
    ).fetchall():
        task_id = task_row["task_id"]
        snapshot = _rebuild_snapshot(conn, task_row)
        computed: Counter = Counter()
        for transfer in settle(snapshot):
            if transfer.reason in BOUNTY_REASONS:
                computed[(transfer.reason, transfer.to_pubkey, transfer.amount_mu)] += 1
            elif transfer.reason == "FEE_REFUND":
                expected_refunds[(transfer.to_pubkey, transfer.amount_mu)] += 1
        recorded: Counter = Counter()
        for row in conn.execute(
            'SELECT reason, to_pubkey, "amount_µ" AS amount FROM transfers '
            "WHERE from_pubkey = ? ORDER BY id",
            (f"ESCROW:{task_id}",),
        ):
            recorded[(row["reason"], row["to_pubkey"], int(row["amount"]))] += 1
        if computed != recorded:
            missing = computed - recorded
            extra = recorded - computed
            result.divergences.append(
                f"task {task_id}: settlement mismatch (missing {dict(missing)}, extra {dict(extra)})"
            )
        result.tasks_resettled += 1

    recorded_refunds: Counter = Counter()
    for row in conn.execute(
        'SELECT to_pubkey, "amount_µ" AS amount FROM transfers WHERE reason = \'FEE_REFUND\''
    ):
        recorded_refunds[(row["to_pubkey"], int(row["amount"]))] += 1
    if expected_refunds != recorded_refunds:
        result.divergences.append(
            f"fee refunds mismatch (computed {dict(expected_refunds)}, recorded {dict(recorded_refunds)})"
        )


def _recheck_admissions(conn: sqlite3.Connection, evidence_dir: Path, result: AuditResult) -> None:
    status_by_id = {
        row["claim_id"]: row["status"] for row in conn.execute("SELECT claim_id, status FROM claims")
    }
    task_by_id = {
        row["claim_id"]: row["task_id"] for row in conn.execute("SELECT claim_id, task_id FROM claims")
    }
    first_fetch: dict[tuple[str, str], int] = {}
    for row in conn.execute("SELECT reader, object_id, epoch FROM fetches ORDER BY id"):
        key = (row["reader"], row["object_id"])
        if key not in first_fetch or row["epoch"] < first_fetch[key]:
            first_fetch[key] = row["epoch"]

    for row in conn.execute("SELECT * FROM claims WHERE status = 'ADMITTED' ORDER BY rowid"):
        claim = {
            "task_id": row["task_id"],
            "subtask_id": row["subtask_id"],
            "author": row["author"],
            "kind": row["kind"],
            "body": row["body"],
            "evidence": json.loads(row["evidence_json"]),
            "cites": json.loads(row["cites_json"]),
            "derivations": json.loads(row["derivations_json"]),
        }
        claim_id = row["claim_id"]

        def diverge(reason: str) -> None:
            result.divergences.append(f"claim {claim_id[:12]} should have been rejected: {reason}")

        payload = crypto.canonical_claim_bytes(claim)
        if crypto.sha256_bytes(payload) != claim_id:
            diverge("claim_id does not hash the canonical body")
        elif not crypto.verify(row["author"], payload, row["sig"]):
            diverge("bad signature")
        if len(row["body"]) > 600:
            diverge("body too long")
        if row["kind"] == "ANSWER" and not row["body"].startswith("ANSWER: "):
            diverge("bad answer body")
        if len(claim["evidence"]) > 8 or len(claim["cites"]) > 16:
            diverge("shape limits exceeded")
        if not claim_helpers.validate_derivations_shape(claim["cites"], claim["derivations"]):
            diverge("bad derivations shape")
        for cited in claim["cites"]:
            if status_by_id.get(cited) != "ADMITTED":
                diverge(f"cites unadmitted {cited[:12]}")
            elif task_by_id.get(cited) != row["task_id"]:
                diverge(f"cross-task cite {cited[:12]}")
            elif first_fetch.get((row["author"], cited), 1 << 62) > row["epoch_submitted"]:
                diverge(f"cited {cited[:12]} without a prior fetch")
        for evidence_ref in claim["evidence"]:
            doc_path = evidence_dir / evidence_ref.get("doc_hash", "")
            if not doc_path.exists():
                diverge("missing evidence document")
                continue
            document = doc_path.read_text(encoding="utf-8")
            if resolve_ref_span(document, evidence_ref.get("ref", {})) is None:
                diverge("evidence ref does not resolve")
        result.claims_rechecked += 1


def audit(path: str | Path) -> AuditResult:
    path = Path(path)
    if path.is_dir():
        db_path, evidence_dir = path / "ledger.db", path / "evidence"
    else:
        db_path, evidence_dir = path, path.parent / "evidence"
    result = AuditResult()
    conn = _connect_readonly(db_path)
    try:
        _replay_balances(conn, result)
        _recheck_settlements(conn, result)
        _recheck_admissions(conn, evidence_dir, result)
    finally:
        conn.close()
    return result
