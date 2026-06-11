from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from legion import crypto
from legion.admission_constants import ADMISSION_FEE, LEASE_BOND


TRANSFER_REASONS = {
    "ESCROW",
    "FEE",
    "FEE_REFUND",
    "PAYOUT_FINISHER",
    "PAYOUT_DERIVATION",
    "PAYOUT_STEERING",
    "BOND",
    "SLASH",
    "BURN",
}


class Store:
    def __init__(self, root: str | Path, db_name: str = "ledger.db") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)
        self.db_path = self.root / db_name
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO meta(key, value) VALUES('epoch', '0');

            CREATE TABLE IF NOT EXISTS identities(
              pubkey TEXT PRIMARY KEY,
              "balance_µ" INTEGER NOT NULL CHECK("balance_µ" >= 0)
            );

            CREATE TABLE IF NOT EXISTS pseudo_accounts(
              account TEXT PRIMARY KEY,
              "balance_µ" INTEGER NOT NULL CHECK("balance_µ" >= 0)
            );

            CREATE TABLE IF NOT EXISTS transfers(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              epoch INTEGER NOT NULL,
              from_pubkey TEXT,
              to_pubkey TEXT,
              "amount_µ" INTEGER NOT NULL CHECK("amount_µ" >= 0),
              reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks(
              task_id TEXT PRIMARY KEY,
              spec_hash TEXT NOT NULL,
              "bounty_µ" INTEGER NOT NULL CHECK("bounty_µ" >= 0),
              status TEXT NOT NULL CHECK(status IN ('OPEN', 'CLOSED', 'EXPIRED')),
              created_epoch INTEGER NOT NULL,
              closed_epoch INTEGER,
              settlement_epoch INTEGER,
              settlement_applied INTEGER NOT NULL DEFAULT 0,
              settlement_version INTEGER,
              answer_claim_id TEXT,
              spec_json TEXT NOT NULL,
              answer_key_json TEXT
            );

            CREATE TABLE IF NOT EXISTS subtasks(
              subtask_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              deps_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('PENDING', 'LEASED', 'DONE')),
              lease_holder TEXT,
              lease_expiry_epoch INTEGER
            );

            CREATE TABLE IF NOT EXISTS evidence_docs(
              doc_hash TEXT PRIMARY KEY,
              task_id TEXT,
              name TEXT,
              bytes_len INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claims(
              claim_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              subtask_id TEXT,
              author TEXT NOT NULL REFERENCES identities(pubkey),
              kind TEXT NOT NULL CHECK(kind IN ('FACT', 'FAIL', 'CONSTRAINT', 'ANSWER')),
              body TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              cites_json TEXT NOT NULL,
              derivations_json TEXT NOT NULL,
              sig TEXT NOT NULL,
              epoch_submitted INTEGER NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('PENDING', 'ADMITTED', 'REJECTED')),
              reject_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS claim_cite_overrides(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              claim_id TEXT NOT NULL REFERENCES claims(claim_id),
              cites_json TEXT NOT NULL,
              epoch INTEGER NOT NULL,
              reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fetches(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              reader TEXT NOT NULL REFERENCES identities(pubkey),
              object_id TEXT NOT NULL,
              epoch INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS challenges(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind TEXT NOT NULL CHECK(kind IN ('UNDER_CITATION', 'MATERIALITY')),
              challenger TEXT NOT NULL REFERENCES identities(pubkey),
              target_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
              related_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
              epoch INTEGER NOT NULL,
              status TEXT NOT NULL,
              upheld INTEGER
            );

            CREATE TRIGGER IF NOT EXISTS claims_no_delete
            BEFORE DELETE ON claims
            BEGIN
              SELECT RAISE(ABORT, 'claims are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS claims_status_only_update
            BEFORE UPDATE ON claims
            WHEN NOT (
              OLD.status = 'PENDING'
              AND NEW.status IN ('ADMITTED', 'REJECTED')
              AND OLD.claim_id IS NEW.claim_id
              AND OLD.task_id IS NEW.task_id
              AND OLD.subtask_id IS NEW.subtask_id
              AND OLD.author IS NEW.author
              AND OLD.kind IS NEW.kind
              AND OLD.body IS NEW.body
              AND OLD.evidence_json IS NEW.evidence_json
              AND OLD.cites_json IS NEW.cites_json
              AND OLD.derivations_json IS NEW.derivations_json
              AND OLD.sig IS NEW.sig
              AND OLD.epoch_submitted IS NEW.epoch_submitted
              AND OLD.reject_reason IS NULL
            )
            BEGIN
              SELECT RAISE(ABORT, 'claims are append-only except status transitions');
            END;

            CREATE TRIGGER IF NOT EXISTS transfers_no_update
            BEFORE UPDATE ON transfers
            BEGIN
              SELECT RAISE(ABORT, 'transfers are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS transfers_no_delete
            BEFORE DELETE ON transfers
            BEGIN
              SELECT RAISE(ABORT, 'transfers are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS fetches_no_update
            BEFORE UPDATE ON fetches
            BEGIN
              SELECT RAISE(ABORT, 'fetches are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS fetches_no_delete
            BEFORE DELETE ON fetches
            BEGIN
              SELECT RAISE(ABORT, 'fetches are append-only');
            END;
            """
        )
        self.conn.commit()

    def epoch(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'epoch'").fetchone()
        return int(row["value"])

    def advance_epoch(self, by: int = 1) -> int:
        next_epoch = self.epoch() + by
        self.conn.execute("UPDATE meta SET value = ? WHERE key = 'epoch'", (str(next_epoch),))
        self.conn.commit()
        return next_epoch

    def create_identity(self, pubkey: str, balance: int = 0) -> None:
        self.conn.execute(
            'INSERT OR IGNORE INTO identities(pubkey, "balance_µ") VALUES(?, ?)',
            (pubkey, balance),
        )
        self.conn.commit()

    def balance(self, pubkey: str) -> int:
        row = self.conn.execute(
            'SELECT "balance_µ" FROM identities WHERE pubkey = ?', (pubkey,)
        ).fetchone()
        if row is None:
            raise KeyError(pubkey)
        return int(row["balance_µ"])

    def balances(self) -> dict[str, int]:
        rows = self.conn.execute(
            'SELECT pubkey, "balance_µ" FROM identities ORDER BY pubkey'
        ).fetchall()
        return {row["pubkey"]: int(row["balance_µ"]) for row in rows}

    @staticmethod
    def _is_pseudo(pubkey: str | None) -> bool:
        return pubkey is None or pubkey.startswith(("ESCROW:", "BOND:", "FEE_POOL", "BURN"))

    def pseudo_balances(self) -> dict[str, int]:
        rows = self.conn.execute(
            'SELECT account, "balance_µ" FROM pseudo_accounts ORDER BY account'
        ).fetchall()
        return {row["account"]: int(row["balance_µ"]) for row in rows}

    def _debit_pseudo(self, account: str, amount: int) -> None:
        row = self.conn.execute(
            'SELECT "balance_µ" FROM pseudo_accounts WHERE account = ?', (account,)
        ).fetchone()
        current = int(row["balance_µ"]) if row is not None else 0
        if current < amount:
            raise ValueError(f"insufficient pseudo balance for {account}")
        self.conn.execute(
            'UPDATE pseudo_accounts SET "balance_µ" = "balance_µ" - ? WHERE account = ?',
            (amount, account),
        )

    def _credit_pseudo(self, account: str, amount: int) -> None:
        self.conn.execute(
            'INSERT INTO pseudo_accounts(account, "balance_µ") VALUES(?, ?) '
            'ON CONFLICT(account) DO UPDATE SET "balance_µ" = "balance_µ" + excluded."balance_µ"',
            (account, amount),
        )

    def _apply_transfer(
        self,
        epoch: int,
        from_pubkey: str | None,
        to_pubkey: str | None,
        amount: int,
        reason: str,
    ) -> None:
        if reason not in TRANSFER_REASONS:
            raise ValueError(f"unknown transfer reason: {reason}")
        if amount < 0:
            raise ValueError("negative transfer amount")
        if from_pubkey is not None:
            if self._is_pseudo(from_pubkey):
                self._debit_pseudo(from_pubkey, amount)
            else:
                current = self.balance(from_pubkey)
                if current < amount:
                    raise ValueError(f"insufficient balance for {from_pubkey}")
                self.conn.execute(
                    'UPDATE identities SET "balance_µ" = "balance_µ" - ? WHERE pubkey = ?',
                    (amount, from_pubkey),
                )
        if to_pubkey is not None:
            if self._is_pseudo(to_pubkey):
                self._credit_pseudo(to_pubkey, amount)
            else:
                self.conn.execute(
                    'INSERT OR IGNORE INTO identities(pubkey, "balance_µ") VALUES(?, 0)',
                    (to_pubkey,),
                )
                self.conn.execute(
                    'UPDATE identities SET "balance_µ" = "balance_µ" + ? WHERE pubkey = ?',
                    (amount, to_pubkey),
                )
        self.conn.execute(
            'INSERT INTO transfers(epoch, from_pubkey, to_pubkey, "amount_µ", reason) '
            "VALUES(?, ?, ?, ?, ?)",
            (epoch, from_pubkey, to_pubkey, amount, reason),
        )

    def add_transfer(
        self,
        *,
        from_pubkey: str | None,
        to_pubkey: str | None,
        amount: int,
        reason: str,
        epoch: int | None = None,
    ) -> None:
        with self.conn:
            self._apply_transfer(
                self.epoch() if epoch is None else epoch,
                from_pubkey,
                to_pubkey,
                amount,
                reason,
            )

    def put_evidence(self, text: str, task_id: str | None = None, name: str | None = None) -> str:
        doc_hash = crypto.content_hash(text)
        path = self.evidence_dir / doc_hash
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        self.conn.execute(
            "INSERT OR IGNORE INTO evidence_docs(doc_hash, task_id, name, bytes_len) "
            "VALUES(?, ?, ?, ?)",
            (doc_hash, task_id, name, len(text.encode("utf-8"))),
        )
        self.conn.commit()
        return doc_hash

    def evidence_exists(self, doc_hash: str) -> bool:
        if not (self.evidence_dir / doc_hash).exists():
            return False
        row = self.conn.execute(
            "SELECT 1 FROM evidence_docs WHERE doc_hash = ?", (doc_hash,)
        ).fetchone()
        return row is not None

    def read_evidence_unmetered(self, doc_hash: str) -> str:
        if not self.evidence_exists(doc_hash):
            raise KeyError(doc_hash)
        return (self.evidence_dir / doc_hash).read_text(encoding="utf-8")

    def fetch(self, reader_pubkey: str, object_id: str) -> str:
        self.create_identity(reader_pubkey, 0)
        if self.evidence_exists(object_id):
            value = self.read_evidence_unmetered(object_id)
        else:
            row = self.conn.execute(
                "SELECT body, status FROM claims WHERE claim_id = ?", (object_id,)
            ).fetchone()
            # Only ADMITTED claim bodies are servable: PENDING/REJECTED bodies
            # would otherwise leak pre-admission information to readers.
            if row is None or row["status"] != "ADMITTED":
                raise KeyError(object_id)
            value = row["body"]
        self.conn.execute(
            "INSERT INTO fetches(reader, object_id, epoch) VALUES(?, ?, ?)",
            (reader_pubkey, object_id, self.epoch()),
        )
        self.conn.commit()
        return value

    def has_fetched(self, reader: str, object_id: str, before_epoch: int | None = None) -> bool:
        if before_epoch is None:
            row = self.conn.execute(
                "SELECT 1 FROM fetches WHERE reader = ? AND object_id = ? LIMIT 1",
                (reader, object_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM fetches WHERE reader = ? AND object_id = ? AND epoch <= ? LIMIT 1",
                (reader, object_id, before_epoch),
            ).fetchone()
        return row is not None

    def create_task(
        self,
        *,
        task_id: str,
        spec_hash: str,
        bounty: int,
        spec: dict[str, Any],
        answer_key: dict[str, Any] | None = None,
        sponsor_pubkey: str | None = None,
    ) -> None:
        with self.conn:
            if sponsor_pubkey is not None:
                self._apply_transfer(
                    self.epoch(), sponsor_pubkey, f"ESCROW:{task_id}", bounty, "ESCROW"
                )
            self.conn.execute(
                'INSERT INTO tasks(task_id, spec_hash, "bounty_µ", status, created_epoch, '
                "spec_json, answer_key_json) VALUES(?, ?, ?, 'OPEN', ?, ?, ?)",
                (
                    task_id,
                    spec_hash,
                    bounty,
                    self.epoch(),
                    crypto.canonical_json(spec),
                    crypto.canonical_json(answer_key) if answer_key is not None else None,
                ),
            )

    def create_subtask(self, subtask_id: str, task_id: str, deps: Iterable[str]) -> None:
        self.conn.execute(
            "INSERT INTO subtasks(subtask_id, task_id, deps_json, status) "
            "VALUES(?, ?, ?, 'PENDING')",
            (subtask_id, task_id, crypto.canonical_json(sorted(deps))),
        )
        self.conn.commit()

    def task_row(self, task_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    def task_spec(self, task_id: str) -> dict[str, Any]:
        return json.loads(self.task_row(task_id)["spec_json"])

    def answer_key(self, task_id: str) -> dict[str, Any] | None:
        encoded = self.task_row(task_id)["answer_key_json"]
        return json.loads(encoded) if encoded else None

    def submit_claim(self, claim: dict[str, Any]) -> str:
        task = self.task_row(claim["task_id"])
        if task["status"] != "OPEN":
            raise ValueError("task ledger is frozen")
        epoch = self.epoch()
        claim = dict(claim)
        claim["epoch_submitted"] = epoch
        claim["status"] = "PENDING"
        with self.conn:
            self.conn.execute(
                "INSERT INTO claims(claim_id, task_id, subtask_id, author, kind, body, "
                "evidence_json, cites_json, derivations_json, sig, epoch_submitted, status, reject_reason) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim["claim_id"],
                    claim["task_id"],
                    claim.get("subtask_id"),
                    claim["author"],
                    claim["kind"],
                    claim["body"],
                    crypto.canonical_json(claim.get("evidence", [])),
                    crypto.canonical_json(claim.get("cites", [])),
                    crypto.canonical_json(claim.get("derivations", [])),
                    claim["sig"],
                    epoch,
                    "PENDING",
                    None,
                ),
            )
            self._apply_transfer(epoch, claim["author"], "FEE_POOL", ADMISSION_FEE, "FEE")
        return claim["claim_id"]

    def parse_claim_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "claim_id": row["claim_id"],
            "task_id": row["task_id"],
            "subtask_id": row["subtask_id"],
            "author": row["author"],
            "kind": row["kind"],
            "body": row["body"],
            "evidence": json.loads(row["evidence_json"]),
            "cites": json.loads(row["cites_json"]),
            "derivations": json.loads(row["derivations_json"]),
            "sig": row["sig"],
            "epoch_submitted": row["epoch_submitted"],
            "status": row["status"],
            "reject_reason": row["reject_reason"],
        }

    def claim(self, claim_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return self.parse_claim_row(row)

    def pending_claims(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE status = 'PENDING' ORDER BY rowid"
        ).fetchall()
        return [self.parse_claim_row(row) for row in rows]

    def admitted_claims(self, task_id: str | None = None) -> list[dict[str, Any]]:
        if task_id is None:
            rows = self.conn.execute(
                "SELECT * FROM claims WHERE status = 'ADMITTED' ORDER BY claim_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM claims WHERE status = 'ADMITTED' AND task_id = ? ORDER BY claim_id",
                (task_id,),
            ).fetchall()
        return [self.parse_claim_row(row) for row in rows]

    def claims_for_subtask(self, subtask_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE subtask_id = ? AND status = 'ADMITTED' ORDER BY epoch_submitted",
            (subtask_id,),
        ).fetchall()
        return [self.parse_claim_row(row) for row in rows]

    def set_claim_status(self, claim_id: str, status: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE claims SET status = ?, reject_reason = ? WHERE claim_id = ?",
            (status, reason, claim_id),
        )
        self.conn.commit()

    def admit_claim(self, claim_id: str) -> None:
        claim = self.claim(claim_id)
        self.set_claim_status(claim_id, "ADMITTED", None)
        if claim.get("subtask_id"):
            self.complete_subtask(claim["subtask_id"], claim["author"])

    def reject_claim(self, claim_id: str, reason: str) -> None:
        self.set_claim_status(claim_id, "REJECTED", reason)

    def close_task_for_answer(
        self,
        task_id: str,
        answer_claim_id: str,
        challenge_window: int,
        settlement_version: int = 2,
    ) -> None:
        row = self.task_row(task_id)
        if row["status"] != "OPEN":
            return
        epoch = self.epoch()
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET status = 'CLOSED', closed_epoch = ?, settlement_epoch = ?, "
                "settlement_version = ?, answer_claim_id = ? WHERE task_id = ?",
                (epoch, epoch + challenge_window, settlement_version, answer_claim_id, task_id),
            )
            # Refund every live lease bond on this task: the close freezes the
            # ledger, so a holder mid-work (e.g. an honest answer-lease holder
            # beaten by a racing unleased answer) must not have its bond
            # stranded and later burned by expiry.
            leased = self.conn.execute(
                "SELECT subtask_id, lease_holder FROM subtasks "
                "WHERE task_id = ? AND status = 'LEASED'",
                (task_id,),
            ).fetchall()
            for sub in leased:
                self._apply_transfer(
                    epoch, f"BOND:{sub['subtask_id']}", sub["lease_holder"], LEASE_BOND, "BOND"
                )
                self.conn.execute(
                    "UPDATE subtasks SET status = 'PENDING', lease_holder = NULL, "
                    "lease_expiry_epoch = NULL WHERE subtask_id = ?",
                    (sub["subtask_id"],),
                )

    def mark_settlement_applied(self, task_id: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET settlement_applied = 1 WHERE task_id = ?", (task_id,)
        )
        self.conn.commit()

    def subtask(self, subtask_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM subtasks WHERE subtask_id = ?", (subtask_id,)
        ).fetchone()
        if row is None:
            raise KeyError(subtask_id)
        return row

    def complete_subtask(self, subtask_id: str, author: str) -> None:
        row = self.subtask(subtask_id)
        if row["status"] == "DONE":
            return
        if row["lease_holder"] != author:
            raise ValueError("only the lease holder can complete a subtask")
        with self.conn:
            self.conn.execute(
                "UPDATE subtasks SET status = 'DONE', lease_holder = NULL, lease_expiry_epoch = NULL "
                "WHERE subtask_id = ?",
                (subtask_id,),
            )
            self._apply_transfer(
                self.epoch(), f"BOND:{subtask_id}", author, LEASE_BOND, "BOND"
            )

    def set_cites_override(self, claim_id: str, cites: list[str], reason: str) -> None:
        self.conn.execute(
            "INSERT INTO claim_cite_overrides(claim_id, cites_json, epoch, reason) VALUES(?, ?, ?, ?)",
            (claim_id, crypto.canonical_json(sorted(dict.fromkeys(cites))), self.epoch(), reason),
        )
        self.conn.commit()

    def latest_cites(self, claim: dict[str, Any]) -> list[str]:
        row = self.conn.execute(
            "SELECT cites_json FROM claim_cite_overrides WHERE claim_id = ? ORDER BY id DESC LIMIT 1",
            (claim["claim_id"],),
        ).fetchone()
        if row is None:
            return list(claim["cites"])
        return json.loads(row["cites_json"])

    def fetch_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT reader, object_id, epoch FROM fetches ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def transfer_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            'SELECT id, epoch, from_pubkey, to_pubkey, "amount_µ", reason FROM transfers ORDER BY id'
        ).fetchall()
        return [
            {
                "id": row["id"],
                "epoch": row["epoch"],
                "from_pubkey": row["from_pubkey"],
                "to_pubkey": row["to_pubkey"],
                "amount_µ": row["amount_µ"],
                "reason": row["reason"],
            }
            for row in rows
        ]

    def snapshot(self, task_id: str) -> dict[str, Any]:
        task = self.task_row(task_id)
        claims = {}
        for claim in self.admitted_claims(task_id):
            claim = dict(claim)
            claim["cites"] = self.latest_cites(claim)
            claims[claim["claim_id"]] = {
                "claim_id": claim["claim_id"],
                "task_id": claim["task_id"],
                "author": claim["author"],
                "kind": claim["kind"],
                "body": claim["body"],
                "evidence": claim["evidence"],
                "cites": claim["cites"],
                "epoch_submitted": claim["epoch_submitted"],
                "status": claim["status"],
            }
        return {
            "task_id": task_id,
            "bounty_µ": int(task["bounty_µ"]),
            "answer_claim_id": task["answer_claim_id"],
            "settlement_version": task["settlement_version"] or 2,
            "claims": claims,
            "fetches": self.fetch_rows(),
        }

    def apply_settlement_transfers(self, task_id: str, transfers: Iterable[Any]) -> None:
        with self.conn:
            for transfer in transfers:
                amount = transfer.amount_mu if hasattr(transfer, "amount_mu") else transfer["amount_µ"]
                reason = transfer.reason if hasattr(transfer, "reason") else transfer["reason"]
                from_pubkey = (
                    transfer.from_pubkey if hasattr(transfer, "from_pubkey") else transfer["from_pubkey"]
                )
                to_pubkey = transfer.to_pubkey if hasattr(transfer, "to_pubkey") else transfer["to_pubkey"]
                if from_pubkey is None and reason.startswith("PAYOUT"):
                    from_pubkey = f"ESCROW:{task_id}"
                if reason == "BURN" and from_pubkey is None:
                    from_pubkey = f"ESCROW:{task_id}"
                self._apply_transfer(self.epoch(), from_pubkey, to_pubkey, amount, reason)
            self.conn.execute(
                "UPDATE tasks SET settlement_applied = 1 WHERE task_id = ?", (task_id,)
            )
