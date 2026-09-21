"""Phase 4 — adaptive attacker vs. provenance defense.

Evaluates whether the provenance defense holds when the attacker knows the
defense exists and optimizes against it.  Two experiments in one run:

Experiment 1  Per-query comparison (fair head-to-head with Phase 3)
              For each test query, run CEM with TWO objectives:
                static  — raw embedding similarity (defense-unaware)
                adaptive — post-defense score (defense-aware)
              Evaluate both through the defended pipeline.

Experiment 2  T2 universal trigger (realistic threat model)
              Train ONE trigger on a distribution of training queries.
              Evaluate on HELD-OUT test queries through the defended pipeline.
              Run this for both static and adaptive objectives.

The summary table tells the three-column story:
  no defense  →  defense vs. static  →  defense vs. adaptive

Run:
  python -m rbench.attack.run_adaptive --dataset scifact --device auto
  python -m rbench.attack.run_adaptive --dataset scifact --beta 0.5 --device auto
  python -m rbench.attack.run_adaptive --dataset toy --device auto
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from ..config import RESULTS_DIR, RetrievalConfig, resolve_device
from ..retriever import Doc, Embedder, RetrievalPipeline, load_toy
from ..retriever.beir_loader import load_beir
from ..defense.provenance import assign_provenance
from .cem import CEMConfig, optimize_tokens
from .adaptive import (
    make_perquery_score_fn,
    make_static_score_fn,
    make_t2_score_fn,
    make_static_t2_score_fn,
)

PAYLOAD = "[PAYLOAD-PLACEHOLDER: non-actionable lab marker representing an injected instruction]"
POISON_ID = "poison-adaptive"


def make_poison(text: str) -> Doc:
    return Doc(doc_id=POISON_ID, text=text, provenance="external", is_poison=True)


def eval_trigger(pipe, queries, trigger_text, poison_id=POISON_ID):
    """Evaluate one trigger across queries.  Return per-query hit list."""
    results = []
    for q in queries:
        poison = make_poison(f"{trigger_text}. {PAYLOAD}")
        res = pipe.search_with_extra(q.query_id, q.text, poison)
        results.append((q.query_id, res.contains(poison_id), res.rank_of(poison_id)))
    return results


def rsr(rows) -> float:
    return sum(1 for _, hit, _ in rows if hit) / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: adaptive attacker vs. provenance defense")
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5, help="defense penalty strength")
    ap.add_argument("--external-frac", type=float, default=0.3)
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="fraction of queries used for T2 training (rest = held-out test)")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    device = resolve_device(args.device)

    # ── load dataset ───────────────────────────────────────────────────
    if args.dataset == "toy":
        docs, queries = load_toy()
    else:
        print(f"loading BEIR '{args.dataset}' ...")
        docs, queries = load_beir(args.dataset, max_docs=args.max_docs,
                                  max_queries=args.max_queries)

    base = assign_provenance([d for d in docs if not d.is_poison],
                             external_frac=args.external_frac, seed=args.seed)
    n_ext = sum(1 for d in base if d.provenance == "external")
    pipe_name = "hybrid+rerank" if args.hybrid else "dense-only"

    # ── train / test split ─────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(queries))
    n_train = max(1, int(len(queries) * args.train_frac))
    train_q = [queries[i] for i in perm[:n_train]]
    test_q = [queries[i] for i in perm[n_train:]]
    if not test_q:
        test_q = train_q

    print(f"dataset={args.dataset}  device={device}  k={args.k}  "
          f"beta={args.beta}  pipeline={pipe_name}")
    print(f"corpus={len(base)} docs ({n_ext} external)  |  "
          f"queries: {len(train_q)} train, {len(test_q)} test")
    print(f"CEM: trigger_len={args.trigger_len}  iters={args.iters}  "
          f"samples={args.samples}\n")

    # ── set up embedder and pipeline ───────────────────────────────────
    embedder = Embedder(RetrievalConfig().embedder_name, device=device, normalize=True)
    trust_w = RetrievalConfig().trust_weights
    trust_ext = trust_w.get("external", 0.0)

    cfg_def = RetrievalConfig(
        hybrid=args.hybrid, rerank=args.hybrid, top_k=args.k, device=device,
        provenance_defense=True, provenance_penalty=args.beta)
    cfg_nodef = RetrievalConfig(
        hybrid=args.hybrid, rerank=args.hybrid, top_k=args.k, device=device,
        provenance_defense=False)

    pipe_def = RetrievalPipeline(embedder, cfg_def).index_base(base)
    pipe_nodef = RetrievalPipeline(embedder, cfg_nodef).index_base(base)

    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters,
                        n_samples=args.samples, seed=args.seed)

    # ── pre-encode queries and base scores ─────────────────────────────
    base_emb = pipe_def._base_emb
    all_q_vecs = {q.query_id: embedder.encode_one(q.text) for q in queries}
    all_base_scores = {qid: base_emb @ qv for qid, qv in all_q_vecs.items()}

    # ==================================================================
    # Experiment 1 — per-query: static vs. adaptive (on test queries)
    # ==================================================================
    print("=" * 66)
    print("Experiment 1: per-query  static vs. adaptive  (test set)")
    print("=" * 66)

    pq_static_rows = []
    pq_adaptive_rows = []
    pq_naive_rows = []
    pq_nodef_rows = []

    for i, q in enumerate(test_q, 1):
        qv = all_q_vecs[q.query_id]
        bs = all_base_scores[q.query_id]

        # static objective (raw similarity)
        sfn_static = make_static_score_fn(embedder, qv, payload=PAYLOAD)
        trig_static = optimize_tokens(
            embedder.tokenizer, sfn_static, cem_cfg, device=device).best_trigger

        # adaptive objective (post-defense score)
        sfn_adapt = make_perquery_score_fn(
            embedder, qv, bs, args.beta, trust_ext, payload=PAYLOAD)
        trig_adapt = optimize_tokens(
            embedder.tokenizer, sfn_adapt, cem_cfg, device=device).best_trigger

        # naive trigger (query echo)
        trig_naive = q.text

        # evaluate all through defended pipeline
        for trig, rows in [(trig_static, pq_static_rows),
                           (trig_adapt, pq_adaptive_rows),
                           (trig_naive, pq_naive_rows)]:
            poison = make_poison(f"{trig}. {PAYLOAD}")
            res = pipe_def.search_with_extra(q.query_id, q.text, poison)
            rows.append((q.query_id, res.contains(POISON_ID), res.rank_of(POISON_ID)))

        # no-defense baseline
        poison_nd = make_poison(f"{trig_static}. {PAYLOAD}")
        res_nd = pipe_nodef.search_with_extra(q.query_id, q.text, poison_nd)
        pq_nodef_rows.append((q.query_id, res_nd.contains(POISON_ID),
                              res_nd.rank_of(POISON_ID)))

        if i % 5 == 0 or i == len(test_q):
            print(f"  [{i}/{len(test_q)}] test queries done")

    print(f"\n{'setting':<32}  RSR@{args.k}")
    print("-" * 46)
    print(f"{'no defense (static CEM)':<32}  {rsr(pq_nodef_rows)*100:5.1f}%")
    print(f"{'defense + naive':<32}  {rsr(pq_naive_rows)*100:5.1f}%")
    print(f"{'defense + static CEM':<32}  {rsr(pq_static_rows)*100:5.1f}%")
    print(f"{'defense + adaptive CEM':<32}  {rsr(pq_adaptive_rows)*100:5.1f}%")

    # ==================================================================
    # Experiment 2 — T2: one trigger on query distribution (held-out)
    # ==================================================================
    print("\n" + "=" * 66)
    print("Experiment 2: T2 universal trigger  (train -> held-out test)")
    print("=" * 66)

    train_qvecs = np.stack([all_q_vecs[q.query_id] for q in train_q])
    train_bscores = [all_base_scores[q.query_id] for q in train_q]

    # T2 static (raw similarity averaged across train queries)
    print("  training T2 static trigger ...")
    sfn_t2_static = make_static_t2_score_fn(embedder, train_qvecs, payload=PAYLOAD)
    t2_static_res = optimize_tokens(
        embedder.tokenizer, sfn_t2_static, cem_cfg, device=device)
    t2_trig_static = t2_static_res.best_trigger
    print(f"    best score: {t2_static_res.best_score:.4f}")

    # T2 adaptive (post-defense score averaged across train queries)
    print("  training T2 adaptive trigger ...")
    sfn_t2_adapt = make_t2_score_fn(
        embedder, train_qvecs, train_bscores, args.beta, trust_ext, payload=PAYLOAD)
    t2_adapt_res = optimize_tokens(
        embedder.tokenizer, sfn_t2_adapt, cem_cfg, device=device)
    t2_trig_adapt = t2_adapt_res.best_trigger
    print(f"    best score: {t2_adapt_res.best_score:.4f}")

    # evaluate both on held-out test queries
    print("  evaluating on held-out test queries ...")

    t2_static_def = eval_trigger(pipe_def, test_q, t2_trig_static)
    t2_static_nodef = eval_trigger(pipe_nodef, test_q, t2_trig_static)
    t2_adapt_def = eval_trigger(pipe_def, test_q, t2_trig_adapt)
    t2_adapt_nodef = eval_trigger(pipe_nodef, test_q, t2_trig_adapt)

    print(f"\n{'setting':<40}  RSR@{args.k}")
    print("-" * 54)
    print(f"{'T2 static,  no defense':<40}  {rsr(t2_static_nodef)*100:5.1f}%")
    print(f"{'T2 static,  defense (b=' + str(args.beta) + ')':<40}  {rsr(t2_static_def)*100:5.1f}%")
    print(f"{'T2 adaptive, no defense':<40}  {rsr(t2_adapt_nodef)*100:5.1f}%")
    print(f"{'T2 adaptive, defense (b=' + str(args.beta) + ')':<40}  {rsr(t2_adapt_def)*100:5.1f}%")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 66)
    print("Summary -- the three-column story")
    print("=" * 66)
    print(f"{'setting':<36}  {'no defense':>12}  {'defense':>12}")
    print("-" * 64)
    print(f"{'per-query naive':<36}  {'--':>12}  "
          f"{rsr(pq_naive_rows)*100:>11.1f}%")
    print(f"{'per-query static CEM':<36}  "
          f"{rsr(pq_nodef_rows)*100:>11.1f}%  "
          f"{rsr(pq_static_rows)*100:>11.1f}%")
    print(f"{'per-query adaptive CEM':<36}  {'--':>12}  "
          f"{rsr(pq_adaptive_rows)*100:>11.1f}%")
    print(f"{'T2 static  (held-out)':<36}  "
          f"{rsr(t2_static_nodef)*100:>11.1f}%  "
          f"{rsr(t2_static_def)*100:>11.1f}%")
    print(f"{'T2 adaptive (held-out)':<36}  "
          f"{rsr(t2_adapt_nodef)*100:>11.1f}%  "
          f"{rsr(t2_adapt_def)*100:>11.1f}%")

    print(f"\nbeta={args.beta}  k={args.k}  pipeline={pipe_name}  "
          f"external_frac={args.external_frac}")

    # ── save results ───────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "dataset": args.dataset, "device": device, "k": args.k,
            "beta": args.beta, "pipeline": pipe_name,
            "external_frac": args.external_frac,
            "corpus_size": len(base), "n_external": n_ext,
            "n_train": len(train_q), "n_test": len(test_q),
            "train_frac": args.train_frac,
            "cem": asdict(cem_cfg), "search_space": "token-level",
        },
        "perquery": {
            "no_defense_static": rsr(pq_nodef_rows),
            "defense_naive": rsr(pq_naive_rows),
            "defense_static": rsr(pq_static_rows),
            "defense_adaptive": rsr(pq_adaptive_rows),
            "per_query_static": [
                {"query": qid, "in_topk": h, "rank": r}
                for qid, h, r in pq_static_rows],
            "per_query_adaptive": [
                {"query": qid, "in_topk": h, "rank": r}
                for qid, h, r in pq_adaptive_rows],
        },
        "t2": {
            "trigger_static": t2_trig_static,
            "trigger_adaptive": t2_trig_adapt,
            "static_nodef": rsr(t2_static_nodef),
            "static_def": rsr(t2_static_def),
            "adaptive_nodef": rsr(t2_adapt_nodef),
            "adaptive_def": rsr(t2_adapt_def),
            "per_query_static": [
                {"query": qid, "in_topk": h, "rank": r}
                for qid, h, r in t2_static_def],
            "per_query_adaptive": [
                {"query": qid, "in_topk": h, "rank": r}
                for qid, h, r in t2_adapt_def],
        },
    }
    tag = args.dataset.replace("/", "_")
    path = RESULTS_DIR / f"phase4_adaptive_{tag}_{pipe_name.replace('+', '')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
