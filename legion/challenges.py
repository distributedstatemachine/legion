from __future__ import annotations

from typing import Any

from legion.admission import Verifier, resolve_ref_span
from legion.admission_constants import CHALLENGE_BOND


def _six_word_overlap(a: str, b: str) -> bool:
    a_words = a.split()
    b_words = b.split()
    if len(a_words) < 6 or len(b_words) < 6:
        return False
    phrases = {" ".join(b_words[i : i + 6]) for i in range(len(b_words) - 5)}
    return any(" ".join(a_words[i : i + 6]) in phrases for i in range(len(a_words) - 5))


class ChallengeEngine:
    def __init__(self, store, verifier: Verifier) -> None:
        self.store = store
        self.verifier = verifier

    def _assert_challenge_window(self, target: dict[str, Any]) -> None:
        task = self.store.task_row(target["task_id"])
        if task["status"] != "CLOSED":
            raise ValueError("challenges are only allowed after task close")
        if task["settlement_epoch"] is None or self.store.epoch() > task["settlement_epoch"]:
            raise ValueError("challenge window has closed")

    def file_under_citation(
        self, challenger: str, target_claim_id: str, omitted_claim_id: str
    ) -> bool:
        target = self.store.claim(target_claim_id)
        self._assert_challenge_window(target)
        omitted = self.store.claim(omitted_claim_id)
        with self.store.conn:
            self.store._apply_transfer(
                self.store.epoch(), challenger, f"BOND:challenge:{target_claim_id}", CHALLENGE_BOND, "BOND"
            )
            cur = self.store.conn.execute(
                "INSERT INTO challenges(kind, challenger, target_claim_id, related_claim_id, epoch, status) "
                "VALUES('UNDER_CITATION', ?, ?, ?, ?, 'PENDING')",
                (challenger, target_claim_id, omitted_claim_id, self.store.epoch()),
            )
            challenge_id = cur.lastrowid
        upheld = (
            omitted_claim_id not in target["cites"]
            and self.store.has_fetched(target["author"], omitted_claim_id, target["epoch_submitted"])
            and _six_word_overlap(target["body"], omitted["body"])
        )
        if upheld:
            cites = sorted([*target["cites"], omitted_claim_id])
            self.store.set_cites_override(target_claim_id, cites, "UNDER_CITATION")
            self.store.add_transfer(
                from_pubkey=f"BOND:challenge:{target_claim_id}",
                to_pubkey=challenger,
                amount=CHALLENGE_BOND,
                reason="BOND",
            )
            self.store.add_transfer(
                from_pubkey=target["author"],
                to_pubkey=challenger,
                amount=CHALLENGE_BOND,
                reason="SLASH",
            )
        else:
            self.store.add_transfer(
                from_pubkey=f"BOND:challenge:{target_claim_id}",
                to_pubkey=None,
                amount=CHALLENGE_BOND,
                reason="BURN",
            )
        self.store.conn.execute(
            "UPDATE challenges SET status = 'RESOLVED', upheld = ? WHERE id = ?",
            (1 if upheld else 0, challenge_id),
        )
        self.store.conn.commit()
        return upheld

    def file_materiality(
        self, challenger: str, target_claim_id: str, cited_claim_id: str
    ) -> bool:
        target = self.store.claim(target_claim_id)
        self._assert_challenge_window(target)
        cited = self.store.claim(cited_claim_id)
        with self.store.conn:
            self.store._apply_transfer(
                self.store.epoch(), challenger, f"BOND:challenge:{target_claim_id}", CHALLENGE_BOND, "BOND"
            )
            cur = self.store.conn.execute(
                "INSERT INTO challenges(kind, challenger, target_claim_id, related_claim_id, epoch, status) "
                "VALUES('MATERIALITY', ?, ?, ?, ?, 'PENDING')",
                (challenger, target_claim_id, cited_claim_id, self.store.epoch()),
            )
            challenge_id = cur.lastrowid
        spans = self._support_spans_excluding(target, cited)
        upheld = cited_claim_id in target["cites"] and self.verifier.supports(target, spans)
        if upheld:
            cites = [claim_id for claim_id in target["cites"] if claim_id != cited_claim_id]
            self.store.set_cites_override(target_claim_id, cites, "MATERIALITY")
            self.store.add_transfer(
                from_pubkey=f"BOND:challenge:{target_claim_id}",
                to_pubkey=challenger,
                amount=CHALLENGE_BOND,
                reason="BOND",
            )
            self.store.add_transfer(
                from_pubkey=target["author"],
                to_pubkey=challenger,
                amount=CHALLENGE_BOND,
                reason="SLASH",
            )
        else:
            self.store.add_transfer(
                from_pubkey=f"BOND:challenge:{target_claim_id}",
                to_pubkey=target["author"],
                amount=CHALLENGE_BOND,
                reason="BOND",
            )
        self.store.conn.execute(
            "UPDATE challenges SET status = 'RESOLVED', upheld = ? WHERE id = ?",
            (1 if upheld else 0, challenge_id),
        )
        self.store.conn.commit()
        return upheld

    def _support_spans_excluding(
        self, target: dict[str, Any], excluded: dict[str, Any]
    ) -> list[str]:
        spans: list[str] = []
        excluded_docs = {ref["doc_hash"] for ref in excluded.get("evidence", [])}
        for evidence_ref in target.get("evidence", []):
            if evidence_ref["doc_hash"] in excluded_docs:
                continue
            document = self.store.read_evidence_unmetered(evidence_ref["doc_hash"])
            span = resolve_ref_span(document, evidence_ref["ref"])
            if span is not None:
                spans.append(span)
        for cited_id in target.get("cites", []):
            if cited_id == excluded["claim_id"]:
                continue
            spans.append(self.store.claim(cited_id)["body"])
        return spans
