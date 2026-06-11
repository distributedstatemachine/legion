from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey


EXCLUDED_CLAIM_FIELDS = {"claim_id", "sig", "epoch_submitted", "status", "reject_reason"}


@dataclass(frozen=True)
class Keypair:
    private_key: str
    pubkey: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_claim_body(claim: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in claim.items() if k not in EXCLUDED_CLAIM_FIELDS}


def canonical_claim_bytes(claim: dict[str, Any]) -> bytes:
    return canonical_bytes(canonical_claim_body(claim))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return sha256_bytes(data)


def generate_keypair() -> Keypair:
    signing_key = SigningKey.generate()
    return Keypair(
        private_key=signing_key.encode().hex(),
        pubkey=signing_key.verify_key.encode().hex(),
    )


def keypair_from_seed(seed: str | bytes) -> Keypair:
    if isinstance(seed, str):
        seed = seed.encode("utf-8")
    signing_key = SigningKey(hashlib.sha256(seed).digest())
    return Keypair(
        private_key=signing_key.encode().hex(),
        pubkey=signing_key.verify_key.encode().hex(),
    )


def signing_key_from_hex(private_key: str) -> SigningKey:
    return SigningKey(bytes.fromhex(private_key))


def sign(private_key: str, payload: bytes) -> str:
    return signing_key_from_hex(private_key).sign(payload).signature.hex()


def verify(pubkey: str, payload: bytes, signature: str) -> bool:
    try:
        VerifyKey(bytes.fromhex(pubkey)).verify(payload, bytes.fromhex(signature))
    except (BadSignatureError, ValueError):
        return False
    return True
