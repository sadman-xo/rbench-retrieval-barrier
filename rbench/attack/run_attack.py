"""Phase 2 — reproduce the retrieval-barrier attack.

For each target query we optimize a trigger (black-box CEM) so the poisoned
document becomes relevant, insert it, re-index, and measure whether it reaches
top-k. We compare three poison variants against two retrieval pipelines:

  variants:   none   (payload only, no trigger)
              naive  (trigger = the query text, the paper's repeat-the-query baseline)
              cem    (CEM-optimized trigger)
  pipelines:  dense-only        (the base paper's setting)
              hybrid + rerank   (dense+BM25+cross-encoder)  <-- the gap-#1 comparison

Primary metric: Retrieval Success Rate (RSR@k) = fraction of target queries whose
poison doc lands in the top-k.

Run:  python -m rbench.attack.run_attack --device auto
      python -m rbench.attack.run_attack --device auto --iters 40 --samples 96

Phase-2 assumption: the attacker knows a representative target query. Phase 4 drops
this — it optimizes ONE trigger over a query distribution and tests on held-out queries.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict

import numpy as np

from ..config import RESULTS_DIR, RetrievalConfig, resolve_device
from ..retriever import Doc, Embedder, RetrievalPipeline, load_toy
from .cem import CEMConfig, optimize

# Inert placeholder — represents where the malicious instruction would sit. Kept
# non-actionable on purpose (defensive research on synthetic data).
PAYLOAD = "[PAYLOAD-PLACEHOLDER: non-actionable lab marker representing an injected instruction]"

_WORD = re.compile(r"[a-z]{3,}")


def build_vocab(docs, queries, max_size: int = 400) -> list[str]:
    """Vocabulary the trigger is drawn from: alphabetic words (len>=3) seen in the
    corpus and queries. A shared vocab is standard for reproducing this attack; it
    lets CEM discover topically-relevant words without any model internals."""
    seen: dict[str, None] = {}
    for d in docs:
        for w in _WORD.findall(d.text.lower()):
            seen.setdefault(w, None)
    for q in queries:
        for w in _WORD.findall(q.text.lower()):
            seen.setdefault(w, None)
    vocab = list(seen.keys())
    return vocab[:max_size]


def poison_corpus(base_docs, poison_id: str, poison_text: str) -> list[Doc]:
    """Return a copy of the corpus with the poison doc's text set to `poison_text`."""
    out = []
    for d in base_docs:
        if d.doc_id == poison_id:
            out.append(Doc(doc_id=d.doc_id, text=poison_text, provenance=d.provenance,
                           is_poison=True, meta=d.meta))
        else:
            out.append(d)
    return out


def evaluate(pipe: RetrievalPipeline, docs, poison_id: str, query) -> tuple[bool, int | None]:
    pipe.index(docs)
    res = pipe.search(query.query_id, query.text)
    return res.contains(poison_id), res.rank_of(poison_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5, help="top_k for retrieval success")
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-hybrid", action="store_true", help="skip the hybrid+rerank pipeline")
    args = ap.parse_args()

    device = resolve_device(args.device)
    docs, queries = load_toy()
    poison_id = next(d.doc_id for d in docs if d.is_poison)
    vocab = build_vocab(docs, queries)
    print(f"device={device} | k={args.k} | corpus={len(docs)} docs | vocab={len(vocab)} words")
    print(f"CEM: trigger_len={args.trigger_len} n_iters={args.iters} n_samples={args.samples}\n")

    embedder = Embedder(RetrievalConfig().embedder_name, device=device, normalize=True)

    # pipelines built once (reranker model stays loaded); re-indexed per variant
    pipes: dict[str, RetrievalPipeline] = {
        "dense-only": RetrievalPipeline(embedder, RetrievalConfig(hybrid=False, top_k=args.k, device=device)),
    }
    if not args.no_hybrid:
        pipes["hybrid+rerank"] = RetrievalPipeline(
            embedder, RetrievalConfig(hybrid=True, rerank=True, top_k=args.k, device=device))

    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters, n_samples=args.samples)
    variants = ["none", "naive", "cem"]
    # results[pipeline][variant] = list of (query_id, in_topk, rank)
    results: dict[str, dict[str, list]] = {p: {v: [] for v in variants} for p in pipes}
    triggers: dict[str, str] = {}

    for q in queries:
        target = embedder.encode_one(q.text)

        def score_fn(cands: list[str]) -> np.ndarray:
            texts = [f"{t}. {PAYLOAD}" for t in cands]
            embs = embedder.encode(texts)
            return embs @ target  # cosine (unit vectors)

        cem_res = optimize(vocab, score_fn, cem_cfg)
        triggers[q.query_id] = cem_res.best_trigger

        poison_texts = {
            "none": PAYLOAD,
            "naive": f"{q.text} {PAYLOAD}",
            "cem": f"{cem_res.best_trigger}. {PAYLOAD}",
        }
        for pname, pipe in pipes.items():
            for v in variants:
                hit, rank = evaluate(pipe, poison_corpus(docs, poison_id, poison_texts[v]), poison_id, q)
                results[pname][v].append((q.query_id, hit, rank))
        print(f"  {q.query_id:12s} cem_trigger: \"{cem_res.best_trigger}\"")

    # ---- aggregate + print ----
    def rsr(rows) -> float:
        return sum(1 for _, hit, _ in rows if hit) / len(rows)

    print("\n" + "=" * 60)
    print(f"Retrieval Success Rate (RSR@{args.k})  -  poison doc in top-{args.k}")
    print("=" * 60)
    header = f"{'variant':<16}" + "".join(f"{p:>18}" for p in pipes)
    print(header)
    for v in variants:
        line = f"{v:<16}"
        for p in pipes:
            line += f"{rsr(results[p][v]) * 100:>16.0f}%"
        print(line)
    print("\n(none = no trigger; naive = repeat the query; cem = optimized trigger)")

    # ---- save ----
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {"device": device, "k": args.k, "cem": asdict(cem_cfg), "vocab_size": len(vocab)},
        "triggers": triggers,
        "rsr": {p: {v: rsr(results[p][v]) for v in variants} for p in pipes},
        "per_query": {p: {v: [{"query": qid, "in_topk": hit, "rank": rank}
                             for qid, hit, rank in results[p][v]] for v in variants} for p in pipes},
    }
    path = RESULTS_DIR / "phase2_attack.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
