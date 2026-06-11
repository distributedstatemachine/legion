from __future__ import annotations

from typing import Any

from legion import crypto
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
    """Resolves challenges atomically.

    The verdict is computed read-only first; all state changes (bond, override,
    payouts, challenge row) then commit in a single transaction so a failure
    cannot leave an override applied with a stranded bond. If an upheld slash
    exceeds the target author's balance the slash is capped at that balance
    (documented in docs/DECISIONS.md) rather than wedging the challenge.

    Deliberate asymmetry: a failed under-citation bond is BURNED (the claim was
    deterministic to check, frivolous filings are pure spam), while a failed
    materiality bond pays the target author (the challenger consumed verifier
    work and impugned a real citation).
    """

    def __init__(self, store, verifier: Verifier) -> None:
        self.store = store
        self.verifier = verifier

    def _assert_challenge_window(self, target: dict[str, Any]) -> None:
        task = self.store.task_row(target["task_id"])
        if task["status"] != "CLOSED":
            raise ValueError("challenges are only allowed after task close")
        if task["settlement_epoch"] is None or self.store.epoch() > task["settlement_epoch"]:
            raise ValueError("challenge window has closed")

    def _resolve(
        self,
        *,
        kind: str,
        challenger: str,
        target: dict[str, Any],
        related_claim_id: str,
        upheld: bool,
        override_cites: list[str],
        failed_bond_to: str | None,
    ) -> bool:
        store = self.store
        epoch = store.epoch()
        bond_account = f"BOND:challenge:{target['claim_id']}"
        with store.conn:
            store._apply_transfer(epoch, challenger, bond_account, CHALLENGE_BOND, "BOND")
            store.conn.execute(
                "INSERT INTO challenges(kind, challenger, target_claim_id, related_claim_id, "
                "epoch, status, upheld) VALUES(?, ?, ?, ?, ?, 'RESOLVED', ?)",
                (kind, challenger, target["claim_id"], related_claim_id, epoch, 1 if upheld else 0),
            )
            if upheld:
                store.conn.execute(
                    "INSERT INTO claim_cite_overrides(claim_id, cites_json, epoch, reason) "
                    "VALUES(?, ?, ?, ?)",
                    (
                        target["claim_id"],
                        crypto.canonical_json(sorted(dict.fromkeys(override_cites))),
                        epoch,
                        kind,
                    ),
                )
                store._apply_transfer(epoch, bond_account, challenger, CHALLENGE_BOND, "BOND")
                slash = min(store.balance(target["author"]), CHALLENGE_BOND)
                if slash > 0:
                    store._apply_transfer(epoch, target["author"], challenger, slash, "SLASH")
            elif failed_bond_to is None:
                store._apply_transfer(epoch, bond_account, None, CHALLENGE_BOND, "BURN")
            else:
                store._apply_transfer(epoch, bond_account, failed_bond_to, CHALLENGE_BOND, "BOND")
        return upheld

    def file_under_citation(
        self, challenger: str, target_claim_id: str, omitted_claim_id: str
    ) -> bool:
        target = self.store.claim(target_claim_id)
        self._assert_challenge_window(target)
        omitted = self.store.claim(omitted_claim_id)
        current_cites = self.store.latest_cites(target)
        upheld = (
            omitted_claim_id not in current_cites
            and self.store.has_fetched(target["author"], omitted_claim_id, target["epoch_submitted"])
            and _six_word_overlap(target["body"], omitted["body"])
        )
        return self._resolve(
            kind="UNDER_CITATION",
            challenger=challenger,
            target=target,
            related_claim_id=omitted_claim_id,
            upheld=upheld,
            override_cites=[*current_cites, omitted_claim_id],
            failed_bond_to=None,
        )

    def file_materiality(
        self, challenger: str, target_claim_id: str, cited_claim_id: str
    ) -> bool:
        target = self.store.claim(target_claim_id)
        self._assert_challenge_window(target)
        cited = self.store.claim(cited_claim_id)
        current_cites = self.store.latest_cites(target)
        spans = self._support_spans_excluding(target, cited)
        upheld = cited_claim_id in current_cites and self.verifier.supports(target, spans)
        return self._resolve(
            kind="MATERIALITY",
            challenger=challenger,
            target=target,
            related_claim_id=cited_claim_id,
            upheld=upheld,
            override_cites=[cid for cid in current_cites if cid != cited_claim_id],
            failed_bond_to=target["author"],
        )

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
