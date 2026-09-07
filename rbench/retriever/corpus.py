"""Corpus and query data model.

The `provenance` field is unused in Phase 1 but is part of the schema now so the
Phase 3 provenance-weighted defense does not require a data migration later.
Provenance is a coarse trust class: 'internal' (verified) > 'external' (ingested
from an untrusted source, e.g. a scraped page or received email) > 'unknown'.
Attacker-planted documents are 'external' by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PROVENANCE_LEVELS = ("internal", "external", "unknown")


@dataclass
class Doc:
    doc_id: str
    text: str
    provenance: str = "unknown"          # one of PROVENANCE_LEVELS
    is_poison: bool = False              # bookkeeping flag for the attack/eval phases
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_LEVELS:
            raise ValueError(
                f"provenance {self.provenance!r} not in {PROVENANCE_LEVELS}"
            )


@dataclass
class Query:
    query_id: str
    text: str
    # ids of docs that are genuinely relevant (ground truth), for honest-retrieval eval
    relevant_ids: tuple[str, ...] = ()


def load_json_corpus(path: str | Path) -> tuple[list[Doc], list[Query]]:
    """Load a corpus + queries from a JSON file with keys 'docs' and 'queries'."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [
        Doc(
            doc_id=d["doc_id"],
            text=d["text"],
            provenance=d.get("provenance", "unknown"),
            is_poison=d.get("is_poison", False),
            meta=d.get("meta", {}),
        )
        for d in data["docs"]
    ]
    queries = [
        Query(
            query_id=q["query_id"],
            text=q["text"],
            relevant_ids=tuple(q.get("relevant_ids", [])),
        )
        for q in data["queries"]
    ]
    return docs, queries


def load_toy() -> tuple[list[Doc], list[Query]]:
    """The bundled synthetic corpus used for the Phase 1 smoke test."""
    from .corpus import load_json_corpus  # self import keeps path logic in one place

    from ..config import DATA_DIR

    return load_json_corpus(DATA_DIR / "toy_corpus.json")
