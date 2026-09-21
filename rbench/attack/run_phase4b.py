"""Phase 4b -- stronger-attacker ablation: CEM vs. greedy coordinate search.

Shows the provenance defense is attack-agnostic by running TWO different
optimizers through the defended pipeline.  If both fail, the defense's
robustness is not an artifact of CEM's search limitations.

For each test query, we optimize a trigger with CEM and with GCS, evaluate
each through the defended pipeline, and report RSR + total query budget.

Run:
  python -m rbench.attack.run_phase4b --dataset scifact --device auto
  python -m rbench.attack.run_phase4b --dataset toy --device auto
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
from .gcs import GCSConfig, QueryCounter, optimize_gcs
from .adaptive import make_perquery_score_fn, make_static_score_fn

PAYLOAD = "[PAYLOAD-PLACEHOLDER: non-actionable lab marker representing an injected instruction]"
POISON_ID = "poison-4b"


def make_poison(text: str) -> Doc:
    return Doc(doc_id=POISON_ID, text=text, provenance="external", is_poison=True)


def rsr(rows) -> float:
    return sum(1 for _, hit, _ in rows if hit) / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 4b: CEM vs. GCS attacker ablation")
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--external-frac", type=float, default=0.3)
    ap.add_argument("--trigger-len", type=int, default=12)
    # CEM params
    ap.add_argument("--cem-iters", type=int, default=40)
    ap.add_argument("--cem-samples", type=int, default=96)
    # GCS params
    ap.add_argument("--gcs-passes", type=int, default=5)
    ap.add_argument("--gcs-candidates", type=int, default=500)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    device = resolve_device(args.device)

    # -- load dataset -------------------------------------------------------
    if args.dataset == "toy":
        docs, queries = load_toy()
    else:
        print(f"loading BEIR '{args.dataset}' ...")
        docs, queries = load_beir(args.dataset, max_docs=args.max_docs,
                                  max_queries=args.max_queries)

    base = assign_provenance([d for d in docs if not d.is_poison],
                             external_frac=args.external_frac, seed=args.seed)
    n_ext = sum(1 for d in base if d.provenance == "external")

    # same train/test split as Phase 4 for comparability
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(queries))
    n_train = max(1, int(len(queries) * args.train_frac))
    test_q = [queries[i] for i in perm[n_train:]]
    if not test_q:
        test_q = [queries[i] for i in perm[:n_train]]

    print(f"dataset={args.dataset}  device={device}  k={args.k}  beta={args.beta}")
    print(f"corpus={len(base)} docs ({n_ext} external)  |  test queries={len(test_q)}")
    print(f"CEM:  iters={args.cem_iters}  samples={args.cem_samples}")
    print(f"GCS:  passes={args.gcs_passes}  candidates/pos={args.gcs_candidates}")
    cem_budget = args.cem_iters * args.cem_samples
    gcs_budget = args.gcs_passes * args.trigger_len * args.gcs_candidates
    print(f"query budget: CEM ~{cem_budget:,}  |  GCS ~{gcs_budget:,}\n")

    # -- set up pipeline -----------------------------------------------------
    embedder = Embedder(RetrievalConfig().embedder_name, device=device,
                        normalize=True)
    trust_ext = RetrievalConfig().trust_weights.get("external", 0.0)

    cfg_def = RetrievalConfig(
        hybrid=False, top_k=args.k, device=device,
        provenance_defense=True, provenance_penalty=args.beta)
    cfg_nodef = RetrievalConfig(
        hybrid=False, top_k=args.k, device=device,
        provenance_defense=False)

    pipe_def = RetrievalPipeline(embedder, cfg_def).index_base(base)
    pipe_nodef = RetrievalPipeline(embedder, cfg_nodef).index_base(base)

    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.cem_iters,
                        n_samples=args.cem_samples, seed=args.seed)
    gcs_cfg = GCSConfig(trigger_len=args.trigger_len, n_passes=args.gcs_passes,
                        candidates_per_pos=args.gcs_candidates, seed=args.seed)

    base_emb = pipe_def._base_emb

    # -- per-query evaluation ------------------------------------------------
    cem_static_rows, cem_adapt_rows, cem_nodef_rows = [], [], []
    gcs_static_rows, gcs_adapt_rows, gcs_nodef_rows = [], [], []
    cem_total_queries, gcs_total_queries = 0, 0
    cem_best_scores, gcs_best_scores = [], []

    print("=" * 66)
    print("Per-query: CEM vs. GCS  (test set)")
    print("=" * 66)

    for i, q in enumerate(test_q, 1):
        qv = embedder.encode_one(q.text)
        bs = base_emb @ qv

        # -- static score functions with query counters ----------------------
        cem_counter = QueryCounter(
            make_static_score_fn(embedder, qv, payload=PAYLOAD))
        gcs_counter = QueryCounter(
            make_static_score_fn(embedder, qv, payload=PAYLOAD))

        # -- adaptive score functions ----------------------------------------
        cem_adapt_counter = QueryCounter(
            make_perquery_score_fn(embedder, qv, bs, args.beta, trust_ext,
                                   payload=PAYLOAD))
        gcs_adapt_counter = QueryCounter(
            make_perquery_score_fn(embedder, qv, bs, args.beta, trust_ext,
                                   payload=PAYLOAD))

        # -- optimize --------------------------------------------------------
        cem_res = optimize_tokens(embedder.tokenizer, cem_counter, cem_cfg,
                                  device=device)
        gcs_res = optimize_gcs(embedder.tokenizer, gcs_counter, gcs_cfg,
                               device=device)

        cem_adapt_res = optimize_tokens(embedder.tokenizer, cem_adapt_counter,
                                        cem_cfg, device=device)
        gcs_adapt_res = optimize_gcs(embedder.tokenizer, gcs_adapt_counter,
                                      gcs_cfg, device=device)

        cem_total_queries += cem_counter.count + cem_adapt_counter.count
        gcs_total_queries += gcs_counter.count + gcs_adapt_counter.count
        cem_best_scores.append(cem_res.best_score)
        gcs_best_scores.append(gcs_res.best_score)

        # -- evaluate through pipelines --------------------------------------
        for trig, rows in [
            (cem_res.best_trigger, cem_static_rows),
            (gcs_res.best_trigger, gcs_static_rows),
            (cem_adapt_res.best_trigger, cem_adapt_rows),
            (gcs_adapt_res.best_trigger, gcs_adapt_rows),
        ]:
            poison = make_poison(f"{trig}. {PAYLOAD}")
            res = pipe_def.search_with_extra(q.query_id, q.text, poison)
            rows.append((q.query_id, res.contains(POISON_ID),
                         res.rank_of(POISON_ID)))

        # no-defense baseline (static triggers only)
        for trig, rows in [
            (cem_res.best_trigger, cem_nodef_rows),
            (gcs_res.best_trigger, gcs_nodef_rows),
        ]:
            poison = make_poison(f"{trig}. {PAYLOAD}")
            res = pipe_nodef.search_with_extra(q.query_id, q.text, poison)
            rows.append((q.query_id, res.contains(POISON_ID),
                         res.rank_of(POISON_ID)))

        if i % 5 == 0 or i == len(test_q):
            print(f"  [{i}/{len(test_q)}] queries done")

    # -- results table -------------------------------------------------------
    avg_cem = np.mean(cem_best_scores)
    avg_gcs = np.mean(gcs_best_scores)

    print(f"\n{'optimizer':<10} {'objective':<12} {'RSR@' + str(args.k):>8}"
          f"  {'queries':>10}  {'avg_score':>10}")
    print("-" * 58)
    print(f"{'CEM':<10} {'static':<12} {rsr(cem_static_rows)*100:>7.1f}%"
          f"  {cem_total_queries//2:>10,}  {avg_cem:>10.4f}")
    print(f"{'CEM':<10} {'adaptive':<12} {rsr(cem_adapt_rows)*100:>7.1f}%"
          f"  {'':>10}  {'':>10}")
    print(f"{'GCS':<10} {'static':<12} {rsr(gcs_static_rows)*100:>7.1f}%"
          f"  {gcs_total_queries//2:>10,}  {avg_gcs:>10.4f}")
    print(f"{'GCS':<10} {'adaptive':<12} {rsr(gcs_adapt_rows)*100:>7.1f}%"
          f"  {'':>10}  {'':>10}")

    print(f"\nNo-defense baseline (static objective):")
    print(f"  CEM: {rsr(cem_nodef_rows)*100:.1f}%    GCS: {rsr(gcs_nodef_rows)*100:.1f}%")

    gcs_vs_cem = avg_gcs - avg_cem
    print(f"\nGCS avg raw score vs CEM: {'+' if gcs_vs_cem >= 0 else ''}"
          f"{gcs_vs_cem:.4f}  "
          f"({'GCS finds higher-scoring triggers' if gcs_vs_cem > 0 else 'CEM finds higher-scoring triggers'})")

    print(f"\nbeta={args.beta}  k={args.k}  pipeline=dense-only"
          f"  external_frac={args.external_frac}")

    # -- save ----------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "dataset": args.dataset, "device": device, "k": args.k,
            "beta": args.beta, "external_frac": args.external_frac,
            "corpus_size": len(base), "n_external": n_ext,
            "n_test": len(test_q),
            "cem": asdict(cem_cfg), "gcs": asdict(gcs_cfg),
        },
        "results": {
            "cem_static_rsr": rsr(cem_static_rows),
            "cem_adaptive_rsr": rsr(cem_adapt_rows),
            "cem_nodef_rsr": rsr(cem_nodef_rows),
            "cem_queries": cem_total_queries,
            "cem_avg_score": float(avg_cem),
            "gcs_static_rsr": rsr(gcs_static_rows),
            "gcs_adaptive_rsr": rsr(gcs_adapt_rows),
            "gcs_nodef_rsr": rsr(gcs_nodef_rows),
            "gcs_queries": gcs_total_queries,
            "gcs_avg_score": float(avg_gcs),
        },
        "per_query": {
            "cem_static": [{"query": q, "in_topk": h, "rank": r}
                           for q, h, r in cem_static_rows],
            "gcs_static": [{"query": q, "in_topk": h, "rank": r}
                           for q, h, r in gcs_static_rows],
        },
    }
    tag = args.dataset.replace("/", "_")
    path = RESULTS_DIR / f"phase4b_gcs_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
