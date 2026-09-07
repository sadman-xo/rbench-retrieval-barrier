"""Phase 3 — provenance-weighted defense vs. the (static) CEM attacker.

Story in three numbers, swept over the penalty strength beta:
  * honest recall@k  — utility: does the defense hurt legitimate retrieval?
  * RSR@k (cem)      — security: does it demote the poison?
  * RSR@k (naive)    — security against the trivial query-echo poison too

Setup that keeps it honest (see rbench/defense/provenance.py): a fraction of the
HONEST corpus is legitimately 'external', so the defense cannot just exclude all
external docs without paying a recall cost. beta=0 is the no-defense baseline.

The attacker here is STATIC (optimizes embedding similarity only, unaware of the
defense). Phase 4 makes it defense-aware (adaptive, at the query-distribution level).

Run:  python -m rbench.defense.run_defense --dataset scifact --device auto
      python -m rbench.defense.run_defense --dataset toy --device auto --hybrid
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import numpy as np

from ..config import RESULTS_DIR, RetrievalConfig, resolve_device
from ..retriever import Embedder, RetrievalPipeline, load_toy
from ..retriever.beir_loader import load_beir
from ..attack.cem import CEMConfig, optimize
from ..attack.run_attack import PAYLOAD, POISON_ID, build_vocab, make_poison
from .provenance import assign_provenance

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def honest_recall(pipe: RetrievalPipeline, queries) -> float:
    vals = []
    for q in queries:
        res = pipe.search_base(q.query_id, q.text)
        vals.append(sum(1 for r in q.relevant_ids if res.contains(r)) / len(q.relevant_ids))
    return statistics.mean(vals) if vals else float("nan")


def rsr(pipe: RetrievalPipeline, queries, poison_texts: dict) -> float:
    hits = 0
    for q in queries:
        res = pipe.search_with_extra(q.query_id, q.text, make_poison(poison_texts[q.query_id]))
        hits += int(res.contains(POISON_ID))
    return hits / len(queries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--external-frac", type=float, default=0.3,
                    help="fraction of the honest corpus labeled legitimately 'external'")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--hybrid", action="store_true", help="evaluate on hybrid+rerank instead of dense-only")
    args = ap.parse_args()

    device = resolve_device(args.device)
    if args.dataset == "toy":
        docs, queries = load_toy()
    else:
        print(f"loading BEIR '{args.dataset}' ...")
        docs, queries = load_beir(args.dataset, max_docs=args.max_docs, max_queries=args.max_queries)

    base = assign_provenance([d for d in docs if not d.is_poison], external_frac=args.external_frac)
    n_ext = sum(1 for d in base if d.provenance == "external")
    vocab = build_vocab(base, queries)
    pipeline_name = "hybrid+rerank" if args.hybrid else "dense-only"
    print(f"dataset={args.dataset} device={device} k={args.k} pipeline={pipeline_name}")
    print(f"corpus={len(base)} docs ({n_ext} external / {len(base)-n_ext} internal) | "
          f"queries={len(queries)} | vocab={len(vocab)}\n")

    embedder = Embedder(RetrievalConfig().embedder_name, device=device, normalize=True)
    cfg = RetrievalConfig(hybrid=args.hybrid, rerank=args.hybrid, top_k=args.k, device=device)
    pipe = RetrievalPipeline(embedder, cfg).index_base(base)

    # static attacker: optimize a trigger per query once (defense-unaware)
    print("optimizing CEM triggers (static attacker)...")
    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters, n_samples=args.samples)
    cem_texts, naive_texts = {}, {}
    for q in queries:
        target = embedder.encode_one(q.text)

        def score_fn(cands):
            return embedder.encode([f"{t}. {PAYLOAD}" for t in cands]) @ target

        best = optimize(vocab, score_fn, cem_cfg).best_trigger
        cem_texts[q.query_id] = f"{best}. {PAYLOAD}"
        naive_texts[q.query_id] = f"{q.text} {PAYLOAD}"

    # sweep beta: beta=0 -> defense off
    print("\n" + "=" * 66)
    print(f"Provenance defense sweep ({pipeline_name}, external_frac={args.external_frac})")
    print("=" * 66)
    print(f"{'beta':>6}{'honest_recall':>16}{'RSR_naive':>14}{'RSR_cem':>12}")
    rows = []
    for beta in BETAS:
        cfg.provenance_defense = beta > 0
        cfg.provenance_penalty = beta
        rec = honest_recall(pipe, queries)
        r_naive = rsr(pipe, queries, naive_texts)
        r_cem = rsr(pipe, queries, cem_texts)
        rows.append({"beta": beta, "honest_recall": rec, "rsr_naive": r_naive, "rsr_cem": r_cem})
        tag = "  (no defense)" if beta == 0 else ""
        print(f"{beta:>6.2f}{rec:>16.3f}{r_naive*100:>13.0f}%{r_cem*100:>11.0f}%{tag}")

    print("\nHigher beta = stronger penalty on external docs: RSR falls (security up),")
    print("honest recall falls too (utility cost). The gap is the defense's value;")
    print("Phase 4's adaptive attacker will try to claw RSR back up at fixed beta.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"config": {"dataset": args.dataset, "device": device, "k": args.k,
                      "pipeline": pipeline_name, "external_frac": args.external_frac,
                      "corpus_size": len(base), "n_external": n_ext, "n_queries": len(queries),
                      "cem": asdict(cem_cfg)},
           "sweep": rows}
    tag = args.dataset.replace("/", "_")
    path = RESULTS_DIR / f"phase3_defense_{tag}_{pipeline_name.replace('+','')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
