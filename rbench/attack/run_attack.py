"""Phase 2 — reproduce the retrieval-barrier attack (toy or real BEIR corpus).

For each target query we optimize a trigger (black-box CEM) so the poisoned
document becomes relevant, inject it, and measure whether it reaches top-k. We
compare three poison variants against two retrieval pipelines:

  variants:   none   (payload only, no trigger)
              naive  (trigger = the query text, the paper's repeat-the-query baseline)
              cem    (CEM-optimized trigger)
  pipelines:  dense-only        (the base paper's setting)
              hybrid + rerank   (dense+BM25+cross-encoder)  <-- the gap-#1 comparison

Primary metric: Retrieval Success Rate (RSR@k) = fraction of target queries whose
poison doc lands in the top-k.

Run:  python -m rbench.attack.run_attack --dataset scifact --device auto
      python -m rbench.attack.run_attack --dataset toy --device auto
      python -m rbench.attack.run_attack --dataset scifact --max-docs 5000 --max-queries 50 \
             --device auto --iters 40 --samples 96

The honest corpus is indexed ONCE; each poison trial only re-embeds the single
poison doc (see RetrievalPipeline.search_with_extra), so BEIR-scale corpora are fine.

Phase-2 assumption: the attacker knows a representative target query. Phase 4 drops
this — it optimizes ONE trigger over a query distribution and tests on held-out queries.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict

import numpy as np

from ..config import RESULTS_DIR, RetrievalConfig, resolve_device
from ..retriever import Doc, Embedder, RetrievalPipeline, load_toy
from ..retriever.beir_loader import load_beir
from .cem import CEMConfig, optimize_tokens

# Inert placeholder — represents where the malicious instruction would sit. Kept
# non-actionable on purpose (defensive research on synthetic/benchmark data).
PAYLOAD = "[PAYLOAD-PLACEHOLDER: non-actionable lab marker representing an injected instruction]"
POISON_ID = "poison-inj"

_WORD = re.compile(r"[a-z]{3,}")


def build_vocab(docs, queries, max_size: int = 500) -> list[str]:
    """Trigger vocabulary: query words first (topically on-target), then the most
    frequent corpus words, up to max_size. Black-box: derived only from observable
    text, no model internals."""
    seen: set[str] = set()
    vocab: list[str] = []
    for q in queries:
        for w in _WORD.findall(q.text.lower()):
            if w not in seen:
                seen.add(w)
                vocab.append(w)
    cnt: Counter = Counter()
    for d in docs:
        cnt.update(set(_WORD.findall(d.text.lower())))  # document frequency
    for w, _ in cnt.most_common():
        if len(vocab) >= max_size:
            break
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    return vocab[:max_size]


def make_poison(text: str) -> Doc:
    return Doc(doc_id=POISON_ID, text=text, provenance="external", is_poison=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy", help="toy | scifact | nfcorpus | fiqa | beir/...")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-hybrid", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)

    if args.dataset == "toy":
        docs, queries = load_toy()
    else:
        print(f"loading BEIR '{args.dataset}' (downloads+caches on first run)...")
        docs, queries = load_beir(args.dataset, max_docs=args.max_docs, max_queries=args.max_queries)

    base_docs = [d for d in docs if not d.is_poison]   # honest corpus only
    print(f"dataset={args.dataset} device={device} k={args.k} | "
          f"corpus={len(base_docs)} docs | queries={len(queries)}")
    print(f"CEM (token-level): trigger_len={args.trigger_len} n_iters={args.iters} n_samples={args.samples}\n")

    embedder = Embedder(RetrievalConfig().embedder_name, device=device, normalize=True)

    pipes: dict[str, RetrievalPipeline] = {
        "dense-only": RetrievalPipeline(embedder, RetrievalConfig(hybrid=False, top_k=args.k, device=device)),
    }
    if not args.no_hybrid:
        pipes["hybrid+rerank"] = RetrievalPipeline(
            embedder, RetrievalConfig(hybrid=True, rerank=True, top_k=args.k, device=device))
    for pipe in pipes.values():
        pipe.index_base(base_docs)   # embed honest corpus once per pipeline

    # honest-retrieval sanity: is this a working, healthy retrieval setup?
    dense_pipe = pipes["dense-only"]
    honest_recall = statistics.mean(
        sum(1 for r in q.relevant_ids if dense_pipe.search_base(q.query_id, q.text).contains(r))
        / len(q.relevant_ids) for q in queries) if queries else float("nan")
    print(f"honest recall@{args.k} (dense) = {honest_recall:.3f}\n")

    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters, n_samples=args.samples)
    variants = ["none", "naive", "cem"]
    results: dict[str, dict[str, list]] = {p: {v: [] for v in variants} for p in pipes}
    triggers: dict[str, str] = {}

    for i, q in enumerate(queries, 1):
        target = embedder.encode_one(q.text)

        def score_fn(cands: list[str]) -> np.ndarray:
            embs = embedder.encode([f"{t}. {PAYLOAD}" for t in cands])
            return embs @ target

        cem_res = optimize_tokens(embedder.tokenizer, score_fn, cem_cfg, device=device)
        triggers[q.query_id] = cem_res.best_trigger
        poison_texts = {
            "none": PAYLOAD,
            "naive": f"{q.text} {PAYLOAD}",
            "cem": f"{cem_res.best_trigger}. {PAYLOAD}",
        }
        for pname, pipe in pipes.items():
            for v in variants:
                res = pipe.search_with_extra(q.query_id, q.text, make_poison(poison_texts[v]))
                results[pname][v].append((q.query_id, res.contains(POISON_ID), res.rank_of(POISON_ID)))
        if i % 10 == 0 or i == len(queries):
            print(f"  [{i}/{len(queries)}] queries attacked")

    def rsr(rows) -> float:
        return sum(1 for _, hit, _ in rows if hit) / len(rows)

    print("\n" + "=" * 60)
    print(f"Retrieval Success Rate (RSR@{args.k})  -  poison doc in top-{args.k}")
    print("=" * 60)
    print(f"{'variant':<16}" + "".join(f"{p:>18}" for p in pipes))
    for v in variants:
        print(f"{v:<16}" + "".join(f"{rsr(results[p][v]) * 100:>16.0f}%" for p in pipes))
    print("\n(none = no trigger; naive = repeat the query; cem = optimized trigger)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {"dataset": args.dataset, "device": device, "k": args.k,
                   "corpus_size": len(base_docs), "n_queries": len(queries),
                   "cem": asdict(cem_cfg), "search_space": "token-level",
                   "honest_recall_at_k": honest_recall},
        "triggers": triggers,
        "rsr": {p: {v: rsr(results[p][v]) for v in variants} for p in pipes},
        "per_query": {p: {v: [{"query": qid, "in_topk": hit, "rank": rank}
                             for qid, hit, rank in results[p][v]] for v in variants} for p in pipes},
    }
    tag = args.dataset.replace("/", "_")
    path = RESULTS_DIR / f"phase2_attack_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
