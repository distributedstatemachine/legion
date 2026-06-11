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
XL_DIR = ROOT / "corpus" / "xl"
FIXTURE_PATH = ROOT / "corpus" / "tasks" / "long_deep_archive.json"
XL_FIXTURE_PATH = ROOT / "corpus" / "tasks" / "xl_town_archive.json"

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
    "aqueduct_survey.txt": (
        "The surveyor's report concluded that the wooden aqueduct leaked a "
        "third of its flow, and the council voted to relay the whole course in "
        "fired clay pipe before the next dry summer."
    ),
    "bridge_accounts.txt": (
        "The bridge accounts show that the central span was raised by four feet "
        "and rebuilt in dressed stone after two winters in which ice floes "
        "carried away the old wooden arches."
    ),
    "granary_minutes.txt": (
        "The granary minutes record that a second fireproof storehouse was "
        "raised in brick on the east bank after the great fire consumed half "
        "the common grain reserve."
    ),
}

# Questions are crafted so each gold sentence is the top keyword-overlap
# sentence within its own document (the answer-key-free heuristic stub and a
# competent model should both find them; a vague question would honestly miss).
QUESTION = (
    "What prompted the town to rebuild the quay on oak pilings, order shielded "
    "lanterns for the telescope, convert the waterwheel mill to turbine drive, "
    "relay the leaking aqueduct in clay pipe, raise the bridge span in dressed "
    "stone, and add a fireproof brick storehouse?"
)

XL_DOCS = dict(DOCS) | {
    "walls_ledger.txt": (
        "The walls ledger notes that the eastern rampart was widened to carry "
        "a paved patrol walk after smugglers scaled the old narrow parapet "
        "unseen on moonless nights."
    ),
    "wells_survey.txt": (
        "The wells survey concluded that the shared aquifer had dropped below "
        "the old pump intakes, and the town sank three deeper boreholes lined "
        "with iron rings."
    ),
}

XL_QUESTION = (
    QUESTION[:-1]
    + ", widen the eastern rampart with a paved patrol walk, and sink deeper "
    "boreholes lined with iron rings into the aquifer?"
)


def _entry(rng: random.Random, year: int, index: int) -> str:
    return (
        f"Entry {index}, year {year}, season of {rng.choice(SEASONS)}: "
        f"the {rng.choice(OFFICIALS)} of the {rng.choice(PLACES)} reported "
        f"{rng.choice(EVENTS)}, and the {rng.choice(OFFICIALS)} entered a sum of "
        f"{rng.randrange(2, 480)} shillings in the {rng.choice(LEDGERS)}."
    )


def _document(name: str, gold: str, seed: int, target_bytes: int = 22_000) -> str:
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


def _emit(
    out_dir: Path,
    fixture_path: Path,
    fixture_name: str,
    question: str,
    docs: dict[str, str],
    rel_prefix: str,
    target_bytes: int,
    seed_offset: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    documents, gold_facts = [], []
    for seed, (name, gold) in enumerate(sorted(docs.items())):
        text = _document(name, gold, seed + seed_offset, target_bytes=target_bytes)
        (out_dir / name).write_text(text, encoding="utf-8")
        documents.append(f"{rel_prefix}/{name}")
        gold_facts.append(gold)
    fixture_path.write_text(
        json.dumps(
            {
                "name": fixture_name,
                "question": question,
                "documents": documents,
                "gold_facts": gold_facts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in sorted(docs):
        print(name, (out_dir / name).stat().st_size, "bytes")


def main() -> None:
    _emit(LONG_DIR, FIXTURE_PATH, "deep_archive", QUESTION, DOCS, "long", 22_000)
    # XL: the corpus deliberately exceeds the default 128k-token context window
    # (8 docs x ~80 kB ~= 160k estimated tokens) so the one-call baseline's
    # infeasible branch is exercised in CI on the fake path.
    _emit(
        XL_DIR,
        XL_FIXTURE_PATH,
        "xl_town_archive",
        XL_QUESTION,
        XL_DOCS,
        "xl",
        80_000,
        seed_offset=100,
    )


if __name__ == "__main__":
    main()
