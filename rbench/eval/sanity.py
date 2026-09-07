"""Phase 1 smoke test.

Run:  python -m rbench.eval.sanity            # dense-only
      python -m rbench.eval.sanity --hybrid   # dense + BM25 + reranker

It verifies three things:
  1. The pipeline indexes and answers queries.
  2. Honest retrieval works: relevant docs land in top_k (recall / MRR).
  3. The retrieval barrier is intact BEFORE any attack: the untriggered poison
     doc does NOT reach top_k for honest queries. (Phase 2 will break this.)
"""
from __future__ import annotations

import argparse
import statistics

from ..config import RetrievalConfig, resolve_device
from ..retriever import Embedder, RetrievalPipeline, load_toy
from .metrics import mrr, recall_at_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true", help="use dense+BM25+reranker path")
    ap.add_argument("--no-rerank", action="store_true", help="hybrid fusion without reranking")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (auto picks GPU if present)")
    args = ap.parse_args()

    cfg = RetrievalConfig(hybrid=args.hybrid, rerank=not args.no_rerank,
                          device=resolve_device(args.device))
    print(f"config: {cfg.describe()} | device={cfg.device}")

    docs, queries = load_toy()
    print(f"corpus: {len(docs)} docs, {len(queries)} queries")

    embedder = Embedder(cfg.embedder_name, device=cfg.device, normalize=cfg.normalize_embeddings)
    print(f"embedder dim: {embedder.dim}")

    pipe = RetrievalPipeline(embedder, cfg).index(docs)

    poison_ids = {d.doc_id for d in docs if d.is_poison}
    recalls, mrrs = [], []
    barrier_intact = True

    print("\nper-query results (top_k):")
    for q in queries:
        res = pipe.search(q.query_id, q.text)
        r = recall_at_k(res, q.relevant_ids)
        m = mrr(res, q.relevant_ids)
        recalls.append(r)
        mrrs.append(m)
        poison_hit = any(res.contains(p) for p in poison_ids)
        if poison_hit:
            barrier_intact = False
        top = ", ".join(f"{i}({s:.3f})" for i, s in zip(res.ranked_ids, res.scores))
        print(f"  {q.query_id:12s} recall={r:.2f} mrr={m:.2f} poison_in_topk={poison_hit}")
        print(f"      -> {top}")

    print("\nsummary:")
    print(f"  mean recall@{cfg.top_k} = {statistics.mean(recalls):.3f}")
    print(f"  mean MRR         = {statistics.mean(mrrs):.3f}")
    print(f"  retrieval barrier intact (no poison in top_k pre-attack): {barrier_intact}")

    ok = statistics.mean(recalls) > 0.5 and barrier_intact
    print(f"\nPHASE 1 SMOKE TEST: {'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
