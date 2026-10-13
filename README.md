# rbench — Retrieval-Barrier benchmark

Research code for **indirect prompt injection at the retrieval stage**. It reproduces the
black-box trigger attack of arXiv:2601.07072 (*Overcoming the Retrieval Barrier*) and
studies **provenance-aware defenses** (provenance weighting, an untrusted-only norm cap,
bounded occupancy) with per-query certificates, evaluated against a **defense-aware
adaptive attacker**.

Authors: Sadman Bin Tareq, Rageeb Hasan Shafee.

## Status

- [x] **Phase 1 — RAG testbed**: corpus + embedder + retrieval pipeline
      (dense-only, or full hybrid dense+BM25+reranker behind a flag) + smoke test.
- [x] **Phase 2 — CEM trigger attack**: black-box, **token-level** Cross-Entropy
      Method trigger optimizer (faithful to the authors' USENIX'26 method: samples
      raw token IDs from the embedder's tokenizer; independently implemented) +
      attack runner, on the toy corpus **or real BEIR datasets**
      (SciFact/NFCorpus/FiQA, fetched directly from the official BEIR host).
      Reports Retrieval Success Rate (RSR@k) for none/naive/cem variants across
      dense-only vs hybrid+rerank, plus honest recall@k as a corpus-health check.
      Run: `python -m rbench.attack.run_attack --dataset scifact --device auto`
- [x] **Phase 3 — provenance-weighted defense**: penalizes low-trust (external)
      docs in the final ranking. Evaluated honestly — a fraction of the honest
      corpus is legitimately external — sweeping penalty strength beta to show the
      security (RSR down) vs. utility (honest recall down) trade-off against the
      static attacker.
      Run: `python -m rbench.defense.run_defense --dataset scifact --device auto`
- [x] **Phase 4 — adaptive attacker**: defense-aware CEM (post-defense
      objective) + T2 query-distribution training with held-out evaluation.
      SciFact result: **both static and adaptive CEM hit 0% RSR** at beta=0.5.
      Universal triggers (T2) fail even without defense. *(Phase 4c later showed the
      query echo beats beta=0.5 on Contriever/TAS-B — CEM was the weaker attacker.)*
      Run: `python -m rbench.attack.run_adaptive --dataset scifact --device auto`
- [x] **Phase 4b — stronger-attacker ablation**: greedy coordinate search (GCS)
      vs. CEM through the defended pipeline. GCS finds higher-scoring triggers
      (0.74 vs 0.51 avg, 100% vs 73% no-defense RSR) but **both hit 0% RSR**
      against the defense. The defense is attack-agnostic.
      Run: `python -m rbench.attack.run_phase4b --dataset scifact --device auto`
- [x] **Phase 4c — norm inflation + certificate**: dot-product retrievers
      (Contriever-ms, TAS-B) vs. a cosine control, an untrusted-only norm cap, and a
      per-query certificate that bounds RSR for ANY attacker.
      Run 2 (2026-10-13, 300 SciFact queries, echo+CEM attacker): the **echo+CEM
      suffix** is the strongest attacker — it beats the bare echo on 89–90/90 queries
      and reaches **38–52% RSR at β=0.5**. The norm cap is only a **partial fix**: it
      cuts echo+static (52→40% on Contriever-dot) but a defense-aware echo+adaptive
      stays under the cap (48→48%). **0% RSR for every attacker needs β≥0.75**, where
      external recall is 0 (hard exclusion) and overall recall drops 0.745→0.483.
      Negative result for the norm-cap idea; motivates bounded occupancy. See ROADMAP.md.
      Run: `python -m rbench.attack.run_norm --dataset scifact --max-queries 300 --device auto`
- [~] **Phase 4d — bounded occupancy (BO-j) + LLM obedience test**: at most j
      untrusted docs in the top k. Guarantees (dense-only, any poisons): at most j
      poisons get in (G1), and the trusted top-(k-j) is always kept (G2). It does NOT
      keep one echo poison out, so E4 asks a small LLM whether fewer poisons means
      less obedience (harmless CANARY-42 payload). Code + tests done; SciFact run pending.
      Run: `python -m rbench.attack.run_occupancy --dataset scifact --max-queries 300 --device auto`
      then `python -m rbench.eval.run_isr --results results/phase4d_occupancy_scifact_contriever-msmarco_dot.json --device auto`
- [ ] Phase 5 — learned trust, transferability, writeup.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`faiss-cpu` is optional — the dense index falls back to a numpy brute-force search if it
is not installed, so Phase 1 runs either way.

## Run the Phase 1 smoke test

```bash
python -m rbench.eval.sanity                    # dense-only (the base paper's setting)
python -m rbench.eval.sanity --hybrid           # dense + BM25 + cross-encoder reranker
python -m rbench.eval.sanity --hybrid --device auto   # auto = GPU if present, else CPU
```

It checks that honest retrieval works (recall/MRR) and that the **retrieval barrier is
intact before any attack** — the untriggered poison doc must not reach top_k. Phase 2 is
what breaks that barrier on purpose.

## Tests

```bash
python -m pytest tests -p no:cacheprovider
```

Fake embedders and a fake LLM — no model downloads, runs on CPU in seconds.

## Running in the cloud (Colab GPU + local VS Code)

All experiments run on a Colab GPU runtime with **zero local disk cost** — see
[`colab_bootstrap.ipynb`](colab_bootstrap.ipynb). It clones this repo onto the Colab VM,
installs deps there, and has one cell per phase. Run its Drive cell first: result JSONs
are git-ignored and the Colab disk is wiped when the session ends. It can also expose the
runtime to local VS Code over Remote-SSH (cloudflared). `--device auto` means the same
code runs unchanged on the laptop (CPU) and in the cloud (GPU).

## Layout

```
README.md              # this file: status + how to run
ROADMAP.md             # the living plan: every result table, finding, and next step
colab_bootstrap.ipynb  # Colab GPU runner, one cell per phase (keep at the repo root)
requirements.txt
rbench/
  config.py            # single source of truth for models, top_k, pipeline flags
  retriever/
    corpus.py          # Doc/Query model (Doc.provenance ready for the Phase 3 defense)
    embedder.py        # sentence-transformers wrapper (black-box text->vector)
    pipeline.py        # dense-only | hybrid(dense+bm25) | +reranker
  attack/
    cem.py             # Cross-Entropy Method trigger optimizer (Phase 2)
    gcs.py             # Greedy Coordinate Search optimizer (Phase 4b)
    adaptive.py        # defense-aware score functions (Phase 4)
    run_attack.py      # Phase 2 runner
    run_adaptive.py    # Phase 4 runner (adaptive + T2)
    run_phase4b.py     # Phase 4b runner (CEM vs GCS ablation)
    run_norm.py        # Phase 4c runner (dot product vs. cosine, norm cap, certificate)
    echo.py            # query echo + CEM suffix attacker (Phase 4c)
    multi.py           # m distinct echo+CEM poisons per query (Phase 4d)
    payload.py         # harmless CANARY-42 payload + obedience checks (Phase 4d)
    run_occupancy.py   # Phase 4d runner (bounded occupancy, E1-E3)
  defense/
    provenance.py      # provenance assignment for honest evaluation
    certify.py         # certificates: RSR ceiling (4c), trusted top-(k-j) kept (4d)
    run_defense.py     # Phase 3 runner (beta sweep)
  llm/
    prompt.py          # plain RAG prompt (Phase 4d E4 only)
    generate.py        # HF instruct-model wrapper: batched, greedy
  eval/
    metrics.py         # recall@k, MRR
    sanity.py          # Phase 1 smoke test
    run_isr.py         # Phase 4d E4 runner: LLM obedience vs. poison count
tests/                 # fake-embedder / fake-LLM tests, no downloads
data/toy_corpus.json   # synthetic lab corpus (all content fabricated; inert poison doc)
results/               # experiment outputs (JSON per phase); git-ignored, saved to Drive
slides/                # Beamer progress deck (compile on Overleaf/Colab)
```
