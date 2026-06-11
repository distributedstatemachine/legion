from __future__ import annotations

from legion.admission import AdmissionGate, default_verifier
from legion import settlement
from legion import tasks


class Coordinator:
    def __init__(self, store, verifier=None) -> None:
        self.store = store
        self.verifier = verifier or default_verifier()
        self.gate = AdmissionGate(store, self.verifier)

    def tick(self) -> None:
        tasks.expire_leases(self.store)
        self.gate.process_pending()
        self.settle_due_tasks()
        self.store.advance_epoch()

    def settle_due_tasks(self) -> list[str]:
        rows = self.store.conn.execute(
            "SELECT * FROM tasks WHERE status = 'CLOSED' AND settlement_applied = 0 "
            "AND settlement_epoch <= ? ORDER BY task_id",
            (self.store.epoch(),),
        ).fetchall()
        settled: list[str] = []
        for row in rows:
            snapshot = self.store.snapshot(row["task_id"])
            transfers = settlement.settle(snapshot)
            self.store.apply_settlement_transfers(row["task_id"], transfers)
            settled.append(row["task_id"])
        return settled
