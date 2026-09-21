# rbench — Retrieval-Barrier benchmark

A small, CPU-portable testbed for studying **indirect prompt injection at the retrieval
stage** (arXiv:2601.07072 — *Overcoming the Retrieval Barrier*), extended toward a
**provenance-weighted defense** evaluated against a **defense-aware adaptive attacker**.

Course project — Sadman Bin Tareq (2105040), Rageeb Hasan Shafee (2105175).

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
      Universal triggers (T2) fail even without defense.
      Run: `python -m rbench.attack.run_adaptive --dataset scifact --device auto`
- [x] **Phase 4b — stronger-attacker ablation**: greedy coordinate search (GCS)
      vs. CEM through the defended pipeline. GCS finds higher-scoring triggers
      (0.74 vs 0.51 avg, 100% vs 73% no-defense RSR) but **both hit 0% RSR**
      against the defense. The defense is attack-agnostic.
      Run: `python -m rbench.attack.run_phase4b --dataset scifact --device auto`
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

## Running in the cloud (Colab GPU + local VS Code)

Heavier phases (CEM optimization) are GPU-friendly and can run on a Colab runtime with
**zero local disk cost** — see [`colab_bootstrap.ipynb`](colab_bootstrap.ipynb). It clones
this repo onto the Colab VM, installs deps there, runs the smoke test on GPU, and can
expose the runtime to local VS Code over Remote-SSH (cloudflared). `--device auto` means
the same code runs unchanged on the laptop (CPU) and in the cloud (GPU).

It checks that honest retrieval works (recall/MRR) and that the **retrieval barrier is
intact before any attack** — the untriggered poison doc must not reach top_k. Phase 2 is
what breaks that barrier on purpose.

## Layout

```
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
  defense/
    provenance.py      # provenance assignment for honest evaluation
    run_defense.py     # Phase 3 runner (beta sweep)
  eval/
    metrics.py         # recall@k, MRR
    sanity.py          # Phase 1 smoke test
data/toy_corpus.json   # synthetic lab corpus (all content fabricated; inert poison doc)
results/               # experiment outputs (JSON per phase)
```

## Ethics / scope

Defensive security research on a published paper, for a course. Synthetic corpora only;
the bundled poison document carries an inert placeholder payload. No weaponizable payloads
are included in this repo.
