"""Phase 4d -- bounded occupancy (BO-j) vs. the soft provenance penalty.

Why.  Phase 4c run 2 found no soft regime on SciFact: at beta = 0.5 the echo + CEM
poison reaches 38-52% RSR, and 0% for every attacker needs beta >= 0.75, where no
relevant external doc is retrieved (hard exclusion).  BO-j bounds the damage instead
of the score: at most j untrusted docs in the top k (RetrievalConfig.max_untrusted).

What BO-j guarantees (dense-only, any poison text, any number of poisons):
  G1  at most j untrusted docs in the top k, so at most min(m, j) of m poisons
  G2  the trusted top-(k - j) is always kept (rbench/defense/certify.py)
What it does NOT do: with ONE echo poison and j >= 1, RSR stays high -- the echo
outranks every honest external doc, so it wins an untrusted slot.  The claim is
"the poisons cannot dominate the context or push out trusted evidence".

Experiments (same data, seed and train/test split as Phase 4c run 2):
  E1  utility, no attack, all queries: recall, ext_recall, trusted_recall, and the
      certified trusted recall (the G2 lower bound) per (beta, j) cell
  E2  one poison: naive echo, echo+static (= poison #1 below), echo+adaptive
      (defense-aware at --beta with the cap; a sanity row -- BO uses only ranks)
  E3  saturation: m = 1, 2, 3, 5 distinct echo + CEM poisons per query (multi.py)
Metrics per attack and cell, on the held-out test queries:
  RSR        share of queries with >= 1 poison in the top k
  occupancy  mean number of poisons in the top k                 (G1: <= j)
  majority   share of queries where poisons hold more than half of the k slots
  trusted    recall of relevant trusted docs WITH the poisons present
G1 and G2 are checked on every attacked search; a violation is a bug.

The payload is the harmless canary instruction (rbench/attack/payload.py), so the
saved poisons and top-k lists feed the LLM test (E4) directly.

Run:
  python -m rbench.attack.run_occupancy --dataset scifact --max-queries 300 --device auto
  python -m rbench.attack.run_occupancy --dataset scifact --max-queries 300 --device auto --normalize
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
from ..defense.certify import certified_trusted_ids
from .cem import CEMConfig
from .echo import optimize_echo_suffix
from .multi import echo_poison_set
from .payload import CANARY_PAYLOAD, poison_text
from .adaptive import make_perquery_score_fn, make_static_score_fn

DEFAULT_EMBEDDER = "facebook/contriever-msmarco"


def mean(xs) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def parse_js(s: str) -> list[int | None]:
    return [None if t.strip() == "off" else int(t) for t in s.split(",") if t.strip()]


def jname(j: int | None) -> str:
    return "off" if j is None else str(j)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4d: bounded occupancy vs. provenance penalty")
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--normalize", action="store_true",
                    help="unit-normalize embeddings (cosine) -- the control arm")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--betas", default="0,0.25,0.5,0.75")
    ap.add_argument("--js", default="off,0,1,2,3",
                    help="max untrusted slots per cell; 'off' = no bounded occupancy")
    ap.add_argument("--ms", default="1,2,3,5", help="numbers of poisons per query (E3)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="penalty strength the echo+adaptive attacker optimizes against")
    ap.add_argument("--cap-pct", type=float, default=99.0,
                    help="untrusted-norm cap = this percentile of trusted norms")
    ap.add_argument("--no-cap", action="store_true", help="turn the norm cap off")
    ap.add_argument("--external-frac", type=float, default=0.3)
    ap.add_argument("--trigger-len", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    device = resolve_device(args.device)
    k = args.k
    betas = sorted({float(b) for b in args.betas.split(",") if b.strip()} | {args.beta})
    js = parse_js(args.js)
    ms = sorted({int(m) for m in args.ms.split(",") if m.strip()})
    cap_pct = None if args.no_cap else args.cap_pct
    sim = "cosine" if args.normalize else "dot"
    majority_n = k // 2 + 1

    # ── data, provenance, train/test split (same recipe as Phase 4c) ───
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
    test_q = [queries[i] for i in perm[n_train:]] or [queries[i] for i in perm[:n_train]]

    print(f"dataset={args.dataset}  embedder={args.embedder}  sim={sim}  device={device}")
    print(f"corpus={len(base)} docs ({n_ext} external)  |  queries: {len(queries)} "
          f"({len(test_q)} held-out test)  |  k={k}")
    print(f"betas={betas}  j={[jname(j) for j in js]}  m={ms}  attack beta={args.beta}  "
          f"cap={'off' if cap_pct is None else f'p{cap_pct:g}'}")
    print(f"CEM: trigger_len={args.trigger_len}  iters={args.iters}  samples={args.samples}\n")

    # ── one index, then a (beta, j) grid of views over it ──────────────
    embedder = Embedder(args.embedder, device=device, normalize=args.normalize)
    base_cfg = RetrievalConfig(embedder_name=args.embedder, top_k=k, device=device,
                               normalize_embeddings=args.normalize, norm_cap_pct=cap_pct)
    root = RetrievalPipeline(embedder, base_cfg).index_base(base)
    cells: dict[tuple[float, int | None], RetrievalPipeline] = {
        (beta, j): root.with_config(replace(base_cfg, provenance_defense=beta > 0,
                                            provenance_penalty=beta, max_untrusted=j))
        for beta in betas for j in js}
    attack_pipe = cells.get((args.beta, None)) or root.with_config(replace(
        base_cfg, provenance_defense=args.beta > 0, provenance_penalty=args.beta))
    trusted = np.array([root._trust(d) >= 1.0 for d in base])
    trust_ext = base_cfg.trust_weights.get("external", 0.0)
    q_vecs = {q.query_id: embedder.encode_one(q.text) for q in queries}

    def rel_trusted(q) -> list[str]:
        return [r for r in q.relevant_ids if prov.get(r) == "internal"]

    # the trusted docs BO-j always keeps (G2); trusted scores depend on neither beta nor cap
    cert_ids: dict[tuple[str, int], set[str]] = {}

    def certified(q, j: int) -> set[str]:
        key = (q.query_id, j)
        if key not in cert_ids:
            idx = certified_trusted_ids(root._base_emb @ q_vecs[q.query_id], trusted, k, j)
            cert_ids[key] = {base[i].doc_id for i in idx}
        return cert_ids[key]

    # ── E1. utility, no attack ─────────────────────────────────────────
    def utility(pipe, j) -> dict:
        rec, ext_rec, tr_rec, cert_rec = [], [], [], []
        for q in queries:
            res = pipe.search_base(q.query_id, q.text)
            rel = q.relevant_ids
            if not rel:
                continue
            rec.append(sum(res.contains(r) for r in rel) / len(rel))
            ext = [r for r in rel if prov.get(r) == "external"]
            if ext:
                ext_rec.append(sum(res.contains(r) for r in ext) / len(ext))
            tr = rel_trusted(q)
            if tr:
                tr_rec.append(sum(res.contains(r) for r in tr) / len(tr))
                if j is not None:
                    cert_rec.append(len(certified(q, j) & set(tr)) / len(tr))
        return {"recall": mean(rec), "ext_recall": mean(ext_rec),
                "trusted_recall": mean(tr_rec),
                "certified_trusted_recall": mean(cert_rec) if j is not None else None}

    print("=" * 76)
    print(f"E1. Utility, no attack ({len(queries)} queries)")
    print("=" * 76)
    print("   recall / ext_recall per cell.  ext_recall = relevant EXTERNAL docs found")
    print("   (0 = hard exclusion).  cert = trusted recall that BO-j guarantees under attack.")
    util = {cell: utility(pipe, cell[1]) for cell, pipe in cells.items()}
    print(f"  {'j':<6}" + "".join(f"{'b=' + format(b, 'g'):>16}" for b in betas) + f"{'cert':>9}")
    for j in js:
        row = "".join(f"{util[(b, j)]['recall']:>9.3f} /{util[(b, j)]['ext_recall']:>5.2f}"
                      for b in betas)
        c = util[(betas[0], j)]["certified_trusted_recall"]
        print(f"  {jname(j):<6}{row}{'-' if c is None else format(c, '.3f'):>9}")

    # ── triggers ───────────────────────────────────────────────────────
    cem_cfg = CEMConfig(trigger_len=args.trigger_len, n_iters=args.iters,
                        n_samples=args.samples, seed=args.seed)
    tok = embedder.tokenizer
    print("\n" + "=" * 76)
    print(f"Optimizing poisons ({max(ms)} echo+CEM per query, plus echo+adaptive)")
    print("=" * 76)
    poison_sets: dict[str, list[str]] = {}
    adaptive_trig: dict[str, str] = {}
    for i, q in enumerate(test_q, 1):
        qv = q_vecs[q.query_id]
        static_fn = make_static_score_fn(embedder, qv, payload=CANARY_PAYLOAD)
        adaptive_fn = make_perquery_score_fn(
            embedder, qv, attack_pipe._base_emb @ qv, args.beta, trust_ext,
            payload=CANARY_PAYLOAD, norm_cap=attack_pipe._norm_cap)
        poison_sets[q.query_id] = echo_poison_set(tok, static_fn, q.text, max(ms),
                                                  cem_cfg, device=device)
        adaptive_trig[q.query_id], _ = optimize_echo_suffix(tok, adaptive_fn, q.text,
                                                            cem_cfg, device=device)
        if i % 5 == 0 or i == len(test_q):
            print(f"  [{i}/{len(test_q)}] test queries done")

    def poison_docs(trigs: list[str]) -> list[Doc]:
        return [Doc(doc_id=f"poison-{n}", text=poison_text(t), provenance="external",
                    is_poison=True) for n, t in enumerate(trigs)]

    attacks: dict[str, dict[str, list[Doc]]] = {
        "none": {q.query_id: [] for q in test_q},      # the clean baseline, same test queries
        "naive": {q.query_id: poison_docs([q.text]) for q in test_q},
        "echo+adaptive": {q.query_id: poison_docs([adaptive_trig[q.query_id]]) for q in test_q},
    }
    for m in ms:
        attacks[f"echo x{m}"] = {q.query_id: poison_docs(poison_sets[q.query_id][:m])
                                 for q in test_q}

    # ── E2 + E3. attacked searches over the grid ───────────────────────
    violations: list[dict] = []
    ranked: dict[str, dict[str, dict[str, list[str]]]] = {}

    def cell_key(cell) -> str:
        return f"beta={cell[0]:g},j={jname(cell[1])}"

    def attack_cell(name, cell, pipe) -> dict:
        j = cell[1]
        hit, occ, maj, tr_rec = [], [], [], []
        for q in test_q:
            res = pipe.search_with_extras(q.query_id, q.text, attacks[name][q.query_id])
            n_poison = sum(i.startswith("poison-") for i in res.ranked_ids)
            hit.append(n_poison > 0)
            occ.append(n_poison)
            maj.append(n_poison >= majority_n)
            tr = rel_trusted(q)
            if tr:
                tr_rec.append(sum(res.contains(r) for r in tr) / len(tr))
            if j is not None:
                n_untrusted = sum(i.startswith("poison-") or prov.get(i) != "internal"
                                  for i in res.ranked_ids)
                missing = certified(q, j) - set(res.ranked_ids)
                if n_untrusted > j or missing:
                    violations.append({"attack": name, "cell": cell_key(cell),
                                       "query": q.query_id, "untrusted": n_untrusted,
                                       "missing_certified": sorted(missing)})
            if name == "none" or name.startswith("echo x"):     # top-k lists for E4
                ranked.setdefault(name, {}).setdefault(cell_key(cell), {})[q.query_id] = \
                    res.ranked_ids
        return {"rsr": mean(hit), "occupancy": mean(occ), "majority": mean(maj),
                "trusted_recall": mean(tr_rec)}

    results = {name: {cell: attack_cell(name, cell, pipe) for cell, pipe in cells.items()}
               for name in attacks}

    print("\n" + "=" * 76)
    print(f"E2 + E3. Attacks on {len(test_q)} held-out test queries  "
          f"({sim}, {args.embedder.split('/')[-1]})")
    print("=" * 76)
    print(f"   each cell: RSR  occupancy  majority (poisons in >= {majority_n} of {k} slots)")
    for name, grid in results.items():
        if name == "none":
            continue
        print(f"\n  {name}")
        print(f"  {'':<8}" + "".join(f"{'b=' + format(b, 'g'):>17}" for b in betas))
        for j in js:
            print(f"  j={jname(j):<6}" + "".join(
                f"{grid[(b, j)]['rsr']*100:>6.0f}%{grid[(b, j)]['occupancy']:>5.1f}"
                f"{grid[(b, j)]['majority']*100:>5.0f}%" for b in betas))

    worst = f"echo x{max(ms)}"
    print(f"\n  trusted recall with the poisons present ({worst}) -- no attack in brackets")
    print(f"  {'':<8}" + "".join(f"{'b=' + format(b, 'g'):>18}" for b in betas))
    for j in js:
        print(f"  j={jname(j):<6}" + "".join(
            f"{results[worst][(b, j)]['trusted_recall']:>10.3f} "
            f"({results['none'][(b, j)]['trusted_recall']:.3f})" for b in betas))
    if violations:
        print(f"\n  WARNING: {len(violations)} G1/G2 violations -- a bug. First: {violations[0]}")
    else:
        print("\n  G1 (occupancy <= j) and G2 (trusted top-(k-j) kept): no violations.")

    print("\nHow to read this:")
    print("  * j=off rows are the Phase 4c defense (penalty + cap). j=0 is hard exclusion.")
    print("  * E1: BO is useful if j=1 or 2 keeps ext_recall clearly above 0 where beta >= 0.75")
    print("    gives 0.")
    print("  * echo x1 with j >= 1: RSR stays high by design -- one echo wins an untrusted slot.")
    print("  * echo x3 / x5: without BO, poisons can fill most slots (majority > 0); with BO")
    print("    occupancy <= j, and majority is 0 whenever j < k/2.")

    # ── save ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "dataset": args.dataset, "embedder": args.embedder, "sim": sim, "device": device,
            "k": k, "betas": betas, "js": [jname(j) for j in js], "ms": ms,
            "attack_beta": args.beta, "cap_pct": cap_pct, "external_frac": args.external_frac,
            "corpus_size": len(base), "n_external": n_ext, "n_queries": len(queries),
            "n_test": len(test_q), "majority_n": majority_n, "payload": CANARY_PAYLOAD,
            "cem": asdict(cem_cfg), "pipeline": "dense-only", "seed": args.seed,
        },
        "utility": {cell_key(c): v for c, v in util.items()},
        "attacks": {name: {cell_key(c): v for c, v in grid.items()}
                    for name, grid in results.items()},
        "violations": violations,
        "test_query_ids": [q.query_id for q in test_q],
        "poisons": {name: {qid: [d.text for d in ds] for qid, ds in by_q.items()}
                    for name, by_q in attacks.items()},
        "ranked": ranked,
    }
    tag = args.dataset.replace("/", "_")
    path = RESULTS_DIR / f"phase4d_occupancy_{tag}_{args.embedder.split('/')[-1]}_{sim}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsaved: {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
