"""Load a real BEIR benchmark into the rbench Doc/Query model.

Downloads the official BEIR zip once (corpus.jsonl / queries.jsonl / qrels/*.tsv)
and parses it with the standard library only — no ir_datasets / HF datasets, so it
behaves identically on Windows and on Colab. BEIR is the standard suite the
retrieval-barrier paper uses; SciFact (~5k scientific abstracts) is the small default,
NFCorpus and FiQA are also single-GPU friendly.

Honest corpus docs are marked provenance='internal' (trusted knowledge base); the
attack injects its own provenance='external' poison doc separately — which is what
the Phase 3 provenance defense keys on.

Sampling keeps every doc relevant to a sampled query (so honest retrieval is
meaningful), then tops up with random other docs to `max_docs`, keeping the corpus a
hard, dense competition while tractable.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile

import numpy as np

from ..config import DATA_DIR
from .corpus import Doc, Query

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
CACHE = DATA_DIR / "beir"


def _ensure(name: str) -> "object":
    """Download+extract the BEIR dataset if not cached; return its extracted dir."""
    from pathlib import Path

    dest = CACHE / name
    if (dest / "corpus.jsonl").exists():
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE / f"{name}.zip"
    url = BEIR_URL.format(name=name)
    print(f"  downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "rbench/0.1"})
    with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
        f.write(r.read())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE)          # extracts into CACHE/<name>/
    zip_path.unlink(missing_ok=True)
    if not (dest / "corpus.jsonl").exists():
        raise FileNotFoundError(f"unexpected BEIR layout under {dest}")
    return dest


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(
    name: str = "scifact",
    split: str = "test",
    max_docs: int | None = 5000,
    max_queries: int | None = 50,
    seed: int = 20260907,
) -> tuple[list[Doc], list[Query]]:
    d = _ensure(name)
    rng = np.random.default_rng(seed)

    q_text = {str(q["_id"]): q["text"] for q in _read_jsonl(d / "queries.jsonl")}

    # qrels: <split>.tsv with header "query-id\tcorpus-id\tscore"
    qrels_file = d / "qrels" / f"{split}.tsv"
    if not qrels_file.exists():  # fall back to whatever split exists
        qrels_file = next((d / "qrels").glob("*.tsv"))
    qrels: dict[str, set[str]] = {}
    with open(qrels_file, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            if int(float(score)) > 0:
                qrels.setdefault(qid, set()).add(did)

    usable = sorted(qid for qid in q_text if qrels.get(qid))
    if max_queries and len(usable) > max_queries:
        sel = rng.choice(len(usable), size=max_queries, replace=False)
        usable = [usable[i] for i in sorted(sel)]

    needed = set().union(*(qrels[qid] for qid in usable)) if usable else set()

    all_docs: dict[str, str] = {}
    for row in _read_jsonl(d / "corpus.jsonl"):
        did = str(row["_id"])
        title = (row.get("title") or "").strip()
        body = (row.get("text") or "").strip()
        all_docs[did] = f"{title}. {body}" if title else body

    keep = set(needed)
    others = [x for x in all_docs if x not in keep]
    if max_docs and len(keep) < max_docs:
        room = min(max_docs - len(keep), len(others))
        sel = rng.choice(len(others), size=room, replace=False)
        keep |= {others[i] for i in sel}
    elif not max_docs:
        keep |= set(others)

    docs = [Doc(doc_id=x, text=all_docs[x], provenance="internal") for x in keep if all_docs[x]]
    kept_ids = {dd.doc_id for dd in docs}
    queries = [
        Query(query_id=qid, text=q_text[qid],
              relevant_ids=tuple(r for r in qrels[qid] if r in kept_ids))
        for qid in usable
    ]
    queries = [q for q in queries if q.relevant_ids]
    return docs, queries
