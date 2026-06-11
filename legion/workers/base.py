from __future__ import annotations

from dataclasses import dataclass

from legion.crypto import Keypair


@dataclass
class Worker:
    name: str
    keypair: Keypair
    role: str

    @property
    def pubkey(self) -> str:
        return self.keypair.pubkey

    def step(self, store, task_id: str) -> None:
        raise NotImplementedError
