from __future__ import annotations

from typing import Any

from legion import crypto


CLAIM_KINDS = {"FACT", "FAIL", "CONSTRAINT", "ANSWER"}


def normalize_cites(cites: list[str] | tuple[str, ...] | None) -> list[str]:
    return sorted(dict.fromkeys(cites or ()))


def normalize_derivations(derivations: list[list[str]] | None) -> list[list[str]]:
    if not derivations:
        return []
    normalized = [normalize_cites(parent_set) for parent_set in derivations]
    normalized.sort(key=lambda ids: tuple(ids))
    return normalized


def build_claim(
    *,
    private_key: str,
    author: str,
    task_id: str,
    kind: str,
    body: str,
    evidence: list[dict[str, Any]] | None = None,
    cites: list[str] | None = None,
    derivations: list[list[str]] | None = None,
    subtask_id: str | None = None,
    epoch_submitted: int = 0,
) -> dict[str, Any]:
    if kind not in CLAIM_KINDS:
        raise ValueError(f"unknown claim kind: {kind}")
    claim: dict[str, Any] = {
        "task_id": task_id,
        "subtask_id": subtask_id,
        "author": author,
        "kind": kind,
        "body": body,
        "evidence": evidence or [],
        "cites": normalize_cites(cites),
        "derivations": normalize_derivations(derivations),
    }
    payload = crypto.canonical_claim_bytes(claim)
    claim["claim_id"] = crypto.sha256_bytes(payload)
    claim["sig"] = crypto.sign(private_key, payload)
    claim["epoch_submitted"] = epoch_submitted
    claim["status"] = "PENDING"
    return claim


def words(text: str) -> list[str]:
    return text.split()


def ref_for_sentence(sentence: str) -> dict[str, dict[str, str]]:
    parts = words(sentence)
    if len(parts) < 10:
        raise ValueError("ref tags require at least 10 words")
    return {"ref": {"head": " ".join(parts[:5]), "tail": " ".join(parts[-5:])}}


def evidence_ref(doc_hash: str, sentence: str) -> dict[str, Any]:
    ref = ref_for_sentence(sentence)
    return {"doc_hash": doc_hash, "ref": ref["ref"]}


def validate_derivations_shape(cites: list[str], derivations: list[list[str]]) -> bool:
    if not derivations:
        return True
    cited = set(cites)
    return all(parent_set and set(parent_set) <= cited for parent_set in derivations)
