"""Deterministically generate the long-document eval fixture.

The committed files under `corpus/long/` are the output of this script (seeded,
byte-stable); rerun it only if you change the generator, then re-commit the
text. Long documents (~12 kB each, three per task) put the eval into the
regime the architecture is actually for - multi-document tasks where a single
context-window pass is expensive - instead of the ~1 kB fixtures where the
single-agent baseline trivially wins.

Usage: python -m tools.make_long_fixture
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG_DIR = ROOT / "corpus" / "long"
FIXTURE_PATH = ROOT / "corpus" / "tasks" / "deep_archive.json"

SEASONS = ["thaw", "planting", "high summer", "harvest", "frosts", "deep winter"]
OFFICIALS = ["warden", "clerk", "surveyor", "almoner", "toll-keeper", "master carpenter"]
PLACES = [
    "north gate", "salt store", "rope walk", "fish shambles", "tide mill",
    "beacon hill", "long wharf", "grain exchange", "ferry stairs", "chandlery",
]
EVENTS = [
    "a dispute over weights and measures settled by the assize",
    "the arrival of a coastal trader carrying tar and cordage",
    "repairs to the culvert after the spring flood",
    "an inventory of stores taken before the winter closure",
    "the apprenticeship of two boys to the cooper",
    "a fine levied for unswept chimneys in the lower lanes",
    "the purchase of lamp oil at the autumn fair",
    "a census of carts passing the toll bar in a single week",
    "the reglazing of the council chamber windows",
    "an outbreak of murrain among the common herd, soon contained",
]
LEDGERS = ["common ledger", "toll book", "harbour roll", "assize register"]

# Hand-picked gold facts (verbatim sentences embedded mid-document).
DOCS = {
    "harbor_chronicle.txt": (
        "The harbormaster's ledger records that the deep-water quay was rebuilt "
        "on oak pilings after the great storm season destroyed the original "
        "timber framework."
    ),
    "observatory_log.txt": (
        "The observatory's principal telescope lost three nights of observation "
        "each month to lamplight from the new quarter until the council ordered "
        "shielded street lanterns."
    ),
    "mill_registry.txt": (
        "The grain mill converted from waterwheel to turbine drive in a single "
        "winter, doubling its throughput while halving the toll charged to "
        "upland farmers."
    ),
}

QUESTION = (
    "According to the town archives, what major infrastructure changes did the "
    "town make, and what prompted each of them?"
)


def _entry(rng: random.Random, year: int, index: int) -> str:
    return (
        f"Entry {index}, year {year}, season of {rng.choice(SEASONS)}: "
        f"the {rng.choice(OFFICIALS)} of the {rng.choice(PLACES)} reported "
        f"{rng.choice(EVENTS)}, and the {rng.choice(OFFICIALS)} entered a sum of "
        f"{rng.randrange(2, 480)} shillings in the {rng.choice(LEDGERS)}."
    )


def _document(name: str, gold: str, seed: int, target_bytes: int = 12_000) -> str:
    rng = random.Random(seed)
    entries: list[str] = []
    year, index = 1701 + seed % 40, 1
    while sum(len(e) + 1 for e in entries) < target_bytes:
        entries.append(_entry(rng, year, index))
        index += 1
        if index % 9 == 0:
            year += 1
    midpoint = len(entries) // 2
    entries.insert(midpoint, gold)
    title = name.replace("_", " ").removesuffix(".txt").title()
    return f"{title}\n\n" + "\n".join(entries) + "\n"


def main() -> None:
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    documents, gold_facts = [], []
    for seed, (name, gold) in enumerate(sorted(DOCS.items())):
        (LONG_DIR / name).write_text(_document(name, gold, seed), encoding="utf-8")
        documents.append(f"long/{name}")
        gold_facts.append(gold)
    FIXTURE_PATH.write_text(
        json.dumps(
            {
                "name": "deep_archive",
                "question": QUESTION,
                "documents": documents,
                "gold_facts": gold_facts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in DOCS:
        print(name, (LONG_DIR / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
