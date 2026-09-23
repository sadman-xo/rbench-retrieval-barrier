"""Phase 4c -- does the provenance defense survive an UNBOUNDED similarity?

Hypothesis (raised 2026-09-22).  Phase 4 found the defense structurally robust, but
every run used unit-norm (cosine) embeddings, where a poison's raw score is <= 1.
Dot-product retrievers (Contriever, TAS-B, DPR, ANCE) do not normalize: the score is
|e_d| |q| cos(theta), and an attacker can raise it by inflating |e_d|.  For beta < 1
the defended poison score (1 - beta) * s_p + beta * m grows without limit in s_p, so
norm inflation should break the soft defense.  Proof: rbench/defense/certify.py.

Prior work.  Zhong et al. (EMNLP 2023, arXiv:2310.19156) already showed that white-box
HotFlip passages against Contriever have large norms (their Fig. 4), and that clipping
ALL passage norms blunts the attack at some retrieval cost (their Table 5).  So norm
inflation and norm clipping are NOT new here.  New here: (1) the interaction with the
provenance penalty, (2) a BLACK-BOX token-level CEM attacker (the source paper's
threat model), (3) a cap on UNTRUSTED docs only, so trusted docs keep their scores,
(4) a per-query certificate that holds for any poison text.

Grid.  Every trigger is evaluated on every beta in --betas x {cap off, cap on}:
  cap off, beta = 0   : no defense
  cap off, beta > 0   : penalty only    <- the hypothesis says this breaks
  cap on,  beta = 0   : cap only        (Zhong-style clip, untrusted docs only)
  cap on,  beta > 0   : penalty + cap   <- the proposed fix

Attackers (per-query on the test split; T2 = train split -> held-out test split):
  naive     query text as the trigger (per-query only)
  static    maximize the raw score e_p . q -- defense-unaware, so it rewards norm
  adaptive  maximize the post-defense score WITH the cap applied, at --beta
            (defense-aware; the attacker that matters for the cap-on cells)
  echo+static / echo+adaptive
            the query text as a FIXED prefix, CEM chooses only the suffix, with
            the static / adaptive objective (per-query only; rbench/attack/echo.py).
            The first SciFact run showed the echo beats uniform-start CEM, so this
            starts from the strongest point; under a dot product the suffix can
            also add norm. The plain echo is kept if no suffix beats it.
An adaptive attacker against the penalty alone is not run: CEM uses only the RANK
of scores, and that objective is a monotone function of the raw score for beta < 1,
so its run is identical to the static one.

Also reported: embedding-norm statistics (do black-box triggers inflate the norm?),
honest recall and the recall of relevant EXTERNAL docs (does the defense still let
external evidence in, or is it hard exclusion?), and the certified fraction per cell
(1 - certified = the highest RSR ANY attacker can reach on those queries).

Dense-only: the certificate and the norm argument are about the dense score.

Run:
  python -m rbench.attack.run_norm --dataset scifact --device auto                # Contriever, dot
  python -m rbench.attack.run_norm --dataset scifact --device auto --normalize    # same model, cosine
  python -m rbench.attack.run_norm --dataset scifact --device auto \\
      --embedder sentence-transformers/msmarco-distilbert-base-tas-b              # 2nd dot model
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, replace

import numpy as np

from ..config import RESULTS_DIR, RetrievalConfig, resolve_device
from ..retriever import Doc, Embedder, RetrievalPipeline, load_toy
from ..retriever.beir_loader import load_beir
from ..defense.provenance import assign_provenance
from ..defense.certify import certify_query
from .cem import CEMConfig, optimize_tokens
from .echo import optimize_echo_suffix
from .adaptive import (
    make_perquery_score_fn,
    make_static_score_fn,
    make_static_t2_score_fn,
    make_t2_score_fn,
)

PAYLOAD = "[PAYLOAD-PLACEHOLDER: non-actionable lab marker representing an injected instruction]"
POISON_ID = "poison-norm"
DEFAULT_EMBEDDER = "facebook/contriever-msmarco"   # BEIR-standard dot-product retriever
FLOAT_SLACK = 1e-5    # float32 unit/clipped norms can exceed 1 or the cap by ~1e-7


def make_poison(text: str) -> Doc:
    return Doc(doc_id=POISON_ID, text=text, provenance="external", is_poison=True)


def mean(xs) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4c: norm inflation vs. provenance defense")
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--normalize", action="store_true",
                    help="unit-normalize embeddings (cosine) -- the control arm")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--betas", default="0,0.25,0.5,0.75,1.0",
                    help="comma-separated penalty strengths for the evaluation grid")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="penalty strength the adaptive attacker optimizes against")
    ap.add_argument("--cap-pct", type=float, default=99.0,
                    help="norm cap = this percentile of trusted-doc norms")
    ap.add_argument("--external-frac", type=float, default=0.3)
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    device = resolve_device(args.device)
    betas = tuple(sorted({float(b) for b in args.betas.split(",") if b.strip()} | {args.beta}))
    sim = "cosine" if args.normalize else "dot"

    # ── data, provenance, train/test split (same recipe as Phase 4) ────
    if args.dataset == "toy":
        docs, queries = load_toy()
    else:
        print(f"loading BEIR '{args.dataset}' ...")
        docs, queries = load_beir(args.dataset, max_docs=args.max_docs,
                                  max_queries=args.max_queries)
    base = assign_provenance([d for d in docs if not d.is_poison],
                             external_frac=args.external_frac, seed=args.seed)
    prov = {d.doc_id: d.provenance for d in base}
    n_ext = sum(1 for d in base if d.provenance == "external")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(queries))
    n_train = max(1, int(len(queries) * args.train_frac))
    train_q = [queries[i] for i in perm[:n_train]]
    test_q = [queries[i] for i in perm[n_train:]] or train_q

    print(f"dataset={args.dataset}  embedder={args.embedder}  sim={sim}  device={device}")
    print(f"corpus={len(base)} docs ({n_ext} external)  |  queries: "
          f"{len(train_q)} train, {len(test_q)} test  |  k={args.k}")
    print(f"betas={betas}  attack beta={args.beta}  cap=p{args.cap_pct:g} of trusted norms")
    print(f"CEM: trigger_len={args.trigger_len}  iters={args.iters}  samples={args.samples}\n")

    # ── one index, then a (beta, cap) grid of views over it ────────────
    embedder = Embedder(args.embedder, device=device, normalize=args.normalize)
    base_cfg = RetrievalConfig(embedder_name=args.embedder, top_k=args.k, device=device,
                               normalize_embeddings=args.normalize)
    root = RetrievalPipeline(embedder, base_cfg).index_base(base)
    cells: dict[tuple[float, bool], RetrievalPipeline] = {}
    for cap in (False, True):
        for beta in betas:
            cfg = replace(base_cfg, provenance_defense=beta > 0, provenance_penalty=beta,
                          norm_cap_pct=args.cap_pct if cap else None)
            cells[(beta, cap)] = root.with_config(cfg)
    attack_pipe = cells[(args.beta, True)]            # what the adaptive attacker targets
    norm_cap = attack_pipe._norm_cap
    trusted = np.array([root._trust(d) >= 1.0 for d in base])
    trust_ext = base_cfg.trust_weights.get("external", 0.0)

    q_vecs = {q.query_id: embedder.encode_one(q.text) for q in queries}

    def max_poison_norm(cap: bool) -> float:
        lim = 1.0 if args.normalize else float("inf")
        if cap:
            lim = min(lim, norm_cap)
        return lim * (1.0 + FLOAT_SLACK)

    def certified_frac(beta: float, cap: bool, qs) -> float:
        pipe = cells[(beta, cap)]
        return mean(certify_query(pipe._base_emb @ q_vecs[q.query_id], trusted,
                                  q_vecs[q.query_id], args.k, beta, max_poison_norm(cap))[0]
                    for q in qs)

    # ── 1. embedding norms ─────────────────────────────────────────────
    norms = np.linalg.norm(root._base_emb_raw, axis=1)
    q_norms = np.array([np.linalg.norm(v) for v in q_vecs.values()])

    def nstats(x):
        if len(x) == 0:
            return {"mean": float("nan"), "p50": float("nan"),
                    "p99": float("nan"), "max": float("nan")}
        return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
                "p99": float(np.percentile(x, 99)), "max": float(x.max())}

    norm_stats = {"trusted": nstats(norms[trusted]), "external": nstats(norms[~trusted]),
                  "queries": nstats(q_norms), "cap": norm_cap}
    print("=" * 72)
    print("1. Embedding norms (raw, before any cap)")
    print("=" * 72)
    for name in ("trusted", "external", "queries"):
        s = norm_stats[name]
        print(f"  {name:<9} mean {s['mean']:.3f}  p50 {s['p50']:.3f}  "
              f"p99 {s['p99']:.3f}  max {s['max']:.3f}")
    print(f"  cap (p{args.cap_pct:g} of trusted) = {norm_cap:.3f}")

    # ── 2. utility and certificate per cell (all queries) ──────────────
    def utility(pipe):
        rec, ext_rec = [], []
        for q in queries:
            res = pipe.search_base(q.query_id, q.text)
            rel = q.relevant_ids
            rec.append(sum(res.contains(r) for r in rel) / len(rel))
            ext = [r for r in rel if prov.get(r) == "external"]
            if ext:
                ext_rec.append(sum(res.contains(r) for r in ext) / len(ext))
        return mean(rec), mean(ext_rec)

    print("\n" + "=" * 72)
    print(f"2. Utility and certificate ({len(queries)} queries)")
    print("=" * 72)
    print("   ext_recall = recall of relevant EXTERNAL docs (0 = hard exclusion)")
    print("   certified  = share of queries where NO poison text can reach top-k")
    print(f"{'cap':>5}{'beta':>7}{'recall':>9}{'ext_recall':>12}{'certified':>11}")
    util_rows = []
    for (beta, cap), pipe in cells.items():
        rec, ext_rec = utility(pipe)
        cert = certified_frac(beta, cap, queries)
        util_rows.append({"beta": beta, "cap": cap, "recall": rec,
                          "ext_recall": ext_rec, "certified": cert})
        print(f"{'on' if cap else 'off':>5}{beta:>7.2f}{rec:>9.3f}{ext_rec:>12.3f}"
              f"{cert*100:>10.0f}%")

    # ── 3. attacks ─────────────────────────────────────────────────────
    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters,
                        n_samples=args.samples, seed=args.seed)
    tok = embedder.tokenizer

    print("\n" + "=" * 72)
    print("3. Optimizing triggers")
    print("=" * 72)
    triggers: dict[str, dict[str, str]] = {
        name: {} for name in ("naive", "static", "adaptive", "echo+static", "echo+adaptive")}
    suffix_won: dict[str, list[bool]] = {"echo+static": [], "echo+adaptive": []}
    for i, q in enumerate(test_q, 1):
        qv = q_vecs[q.query_id]
        static_fn = make_static_score_fn(embedder, qv, payload=PAYLOAD)
        adaptive_fn = make_perquery_score_fn(embedder, qv, attack_pipe._base_emb @ qv, args.beta,
                                             trust_ext, payload=PAYLOAD, norm_cap=norm_cap)
        triggers["naive"][q.query_id] = q.text
        triggers["static"][q.query_id] = optimize_tokens(
            tok, static_fn, cem_cfg, device=device).best_trigger
        triggers["adaptive"][q.query_id] = optimize_tokens(
            tok, adaptive_fn, cem_cfg, device=device).best_trigger
        for name, fn in (("echo+static", static_fn), ("echo+adaptive", adaptive_fn)):
            trig, won = optimize_echo_suffix(tok, fn, q.text, cem_cfg, device=device)
            triggers[name][q.query_id] = trig
            suffix_won[name].append(won)
        if i % 5 == 0 or i == len(test_q):
            print(f"  per-query: [{i}/{len(test_q)}] test queries done")

    train_qvecs = np.stack([q_vecs[q.query_id] for q in train_q])
    print("  training T2 static trigger ...")
    t2_static = optimize_tokens(
        tok, make_static_t2_score_fn(embedder, train_qvecs, payload=PAYLOAD),
        cem_cfg, device=device).best_trigger
    print("  training T2 adaptive trigger ...")
    t2_adaptive = optimize_tokens(
        tok, make_t2_score_fn(embedder, train_qvecs,
                              [attack_pipe._base_emb @ q_vecs[q.query_id] for q in train_q],
                              args.beta, trust_ext, payload=PAYLOAD, norm_cap=norm_cap),
        cem_cfg, device=device).best_trigger
    for name, trig in (("T2 static", t2_static), ("T2 adaptive", t2_adaptive)):
        triggers[name] = {q.query_id: trig for q in test_q}

    # ── 4. poison norms: does a black-box trigger inflate |e_p|? ────────
    print("\n" + "=" * 72)
    print("4. Poison embeddings (raw norm relative to the cap; cosine to the query)")
    print("=" * 72)
    print(f"{'attacker':<14}{'norm/cap mean':>15}{'max':>8}{'above cap':>11}{'cos mean':>10}")
    poison_stats = {}
    for name, by_q in triggers.items():
        embs = embedder.encode([f"{by_q[q.query_id]}. {PAYLOAD}" for q in test_q])
        qs = np.stack([q_vecs[q.query_id] for q in test_q])
        pn = np.linalg.norm(embs, axis=1)
        cos = (embs * qs).sum(1) / np.maximum(pn * np.linalg.norm(qs, axis=1), 1e-12)
        ratio = pn / norm_cap
        poison_stats[name] = {"norm_over_cap": ratio.tolist(), "cos": cos.tolist()}
        print(f"{name:<14}{ratio.mean():>15.3f}{ratio.max():>8.3f}"
              f"{(ratio > 1 + FLOAT_SLACK).mean()*100:>10.0f}%{cos.mean():>10.3f}")
    for name, won in suffix_won.items():
        print(f"  {name}: the CEM suffix beat the bare echo on {sum(won)}/{len(won)} queries")

    # ── 5. RSR over the grid ───────────────────────────────────────────
    def rsr_grid(by_q):
        return {cell: mean(pipe.search_with_extra(
                    q.query_id, q.text, make_poison(f"{by_q[q.query_id]}. {PAYLOAD}")
                ).contains(POISON_ID) for q in test_q)
                for cell, pipe in cells.items()}

    grids = {name: rsr_grid(by_q) for name, by_q in triggers.items()}
    bound = {cell: 1.0 - certified_frac(cell[0], cell[1], test_q) for cell in cells}

    print("\n" + "=" * 72)
    print(f"5. RSR@{args.k} on {len(test_q)} held-out test queries  "
          f"({sim}, {args.embedder.split('/')[-1]})")
    print("=" * 72)
    violations = []
    for cap in (False, True):
        print(f"\n  {'cap ' + ('ON' if cap else 'OFF'):<14}"
              + "".join(f"{'b=' + format(b, 'g'):>8}" for b in betas))
        for name, grid in grids.items():
            print(f"  {name:<14}" + "".join(f"{grid[(b, cap)]*100:>7.0f}%" for b in betas))
            violations += [(name, b, cap) for b in betas
                           if grid[(b, cap)] > bound[(b, cap)] + 1e-9]
        print(f"  {'max any':<14}" + "".join(f"{bound[(b, cap)]*100:>7.0f}%" for b in betas)
              + "   <- 1 - certified: the ceiling for ANY attacker")
    if violations:
        print(f"\n  WARNING: RSR above the certified ceiling in {violations} -- a bug.")

    print("\nHow to read this:")
    print("  * cap OFF, b in (0,1): the hypothesis. High static RSR here with sim=dot but not")
    print("    with --normalize means norm inflation breaks the penalty alone.")
    print("  * cap ON, b=attack beta, rows 'adaptive' and 'echo+adaptive': the fix against a")
    print("    defense-aware attacker. 'echo+static' vs 'naive' with cap OFF under sim=dot is")
    print("    the cleanest norm test: a gain there that the cap removes is norm inflation.")
    print("  * b=1 is hard exclusion: safe always, but ext_recall in table 2 shows its cost.")

    # ── save ───────────────────────────────────────────────────────────
    def cell_key(cell):
        return f"beta={cell[0]:g},cap={'on' if cell[1] else 'off'}"

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "dataset": args.dataset, "embedder": args.embedder, "sim": sim,
            "device": device, "k": args.k, "betas": list(betas), "attack_beta": args.beta,
            "cap_pct": args.cap_pct, "external_frac": args.external_frac,
            "corpus_size": len(base), "n_external": n_ext,
            "n_train": len(train_q), "n_test": len(test_q),
            "cem": asdict(cem_cfg), "search_space": "token-level", "pipeline": "dense-only",
        },
        "norms": norm_stats,
        "utility": util_rows,
        "rsr": {name: {cell_key(c): v for c, v in grid.items()} for name, grid in grids.items()},
        "max_any_attacker_test": {cell_key(c): v for c, v in bound.items()},
        "poison_stats": poison_stats,
        "echo_suffix_won": {name: won for name, won in suffix_won.items()},
        "triggers": {"per_query_static": triggers["static"],
                     "per_query_adaptive": triggers["adaptive"],
                     "echo_static": triggers["echo+static"],
                     "echo_adaptive": triggers["echo+adaptive"],
                     "t2_static": t2_static, "t2_adaptive": t2_adaptive},
        "certificate_violations": [list(v) for v in violations],
    }
    tag = args.dataset.replace("/", "_")
    model_tag = args.embedder.split("/")[-1]
    path = RESULTS_DIR / f"phase4c_norm_{tag}_{model_tag}_{sim}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
