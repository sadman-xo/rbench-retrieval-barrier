"""The BEIR loader must give the same docs, in the same order, in every process.

assign_provenance draws one random number per doc IN ORDER, so the doc order fixes
which docs are 'external'. The loader once built its doc list from a Python set of
string ids, and string hashing changes per process (PYTHONHASHSEED): every run got a
different external labeling with the same count. This test loads a small fake BEIR
dataset in two processes with different hash seeds and compares the labels.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import sys
from pathlib import Path
import rbench.retriever.beir_loader as bl
from rbench.defense.provenance import assign_provenance
bl.CACHE = Path(sys.argv[1])
docs, queries = bl.load_beir("fake", max_docs=60, max_queries=10)
for d in assign_provenance(docs):
    print(d.doc_id, d.provenance)
print("queries", " ".join(q.query_id for q in queries))
"""


def write_fake_beir(root: Path, n_docs: int = 200, n_queries: int = 30) -> None:
    d = root / "fake"
    (d / "qrels").mkdir(parents=True)
    with open(d / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i in range(n_docs):
            f.write(json.dumps({"_id": f"doc{i}", "title": "", "text": f"text {i}"}) + "\n")
    with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
        for i in range(n_queries):
            f.write(json.dumps({"_id": f"q{i}", "text": f"query {i}"}) + "\n")
    with open(d / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for i in range(n_queries):
            f.write(f"q{i}\tdoc{(7 * i) % n_docs}\t1\n")


def load_in_process(cache: Path, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(cache)], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_doc_order_and_labels_same_in_every_process(tmp_path):
    write_fake_beir(tmp_path)
    runs = [load_in_process(tmp_path, s) for s in ("1", "2", "3")]
    assert runs[0].count("\n") == 61            # 60 docs + the query line
    assert runs[0] == runs[1] == runs[2]
