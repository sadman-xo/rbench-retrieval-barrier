# rbench — Roadmap & Open Ideas

Living notes for the retrieval-barrier IPI attack/defense project (arXiv:2601.07072).
Captures decisions and ideas from the 2026-09-12 planning chat. Nothing here is dropped —
"someday/maybe" items are kept in their own section rather than deleted.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Where we are (done)

- [x] **Phase 1 — Testbed.** RAG retriever, switchable dense-only vs. hybrid(dense+BM25)+cross-encoder rerank; incremental poison insertion for BEIR-scale corpora; `--device auto` (CPU↔T4).
- [x] **Phase 2 — Attack.** Black-box, token-level CEM trigger optimizer (faithful to the USENIX'26 method, independently implemented). SciFact (5k docs, 50 q): none=0%, cem=**100% dense / 72% hybrid+rerank**, naive=100%/100%. Honest recall@5=0.71.
- [x] **Phase 3 — Defense.** Provenance-weighted retrieval: `score' = score − β·(1−trust)·spread`. SciFact β-sweep (confirmed real): β=0 → RSR 100%/100%; **β=0.5 (sweet spot) → 8% naive / 6% cem**, recall 0.71→0.50; β≥0.75 → 0%. Recall floor ~0.50 is structural (30% of relevant docs are legitimately external).
- [x] **Progress deck** — `slides/rbench_progress.tex` (Beamer, Phases 1–3). Compile on Colab/Overleaf (no LaTeX locally).

Key finding so far: **source-based (provenance) beats content-based (reranker)** — the reranker only raises the attacker's budget (100→72) and misses the naive attack; provenance collapses naive AND cem together (attack-agnostic).

---

## What I (Sadman) want to do — ideas raised this chat

### Idea A — Try optimizers other than CEM
Replace/augment CEM with a stronger black-box optimizer for the trigger search.
- Motivation: a defense is only convincing if it beats the *strongest* attacker, not just CEM.
- Candidates discussed (see "Optimizer options" below): greedy coordinate search, GCG-on-surrogate, genetic algorithm, simulated annealing, NES/SPSA.
- Decision: **keep CEM as the baseline; add a stronger attacker as an ablation** so we can claim "the defense holds even against a better optimizer than CEM."

### Idea B — Learn the trust score instead of hand-labeling it
Train a model to predict whether a source is trustworthy, and use that to drive the defense (per-doc trust, and/or the β operating point).
- The genuinely new part is **learned `trust(d)`**, not "set β from trust" (β already multiplies (1−trust) in the formula).
- **Critical constraint:** the trust model must use **non-forgeable, source-side features** (domain reputation, cryptographic signatures/provenance, ingestion channel, registration age), **NOT document content.** A content-based trust model reintroduces the exact attack surface provenance was meant to remove — the attacker just optimizes the poison to "look trustworthy."
- Data problem: BEIR has no provenance, so this needs realistic simulated source metadata (or a corpus that actually has provenance). Current `assign_provenance` synthetically relabels 30% of honest docs as external.
- This merges naturally with the adaptive attacker: a learned trust model becomes a new target → **trust-spoofing** is the interesting experiment.

---

## What I (Claude) recommend doing — proposed to-dos

### Phase 4 — Defense-aware adaptive attacker (the actual contribution)
- [ ] New `rbench/attack/adaptive.py`: reuse the optimizer loop but change the **objective to the post-defense score** (`score − β·(1−trust)·spread`), so the attacker knows it's penalized and tries to push similarity high enough to survive.
- [ ] Optimize over a **query distribution (T2)**: train one trigger on a train split, evaluate RSR on **held-out** queries.
- [ ] Report the three-column story: **RSR: no defense → defense (static attacker) → defense (adaptive attacker).**
- [ ] Both outcomes are publishable: defense holds (robust) OR defense partially breaks (found the real limit — likely trust-spoofing).

### Phase 4b — Stronger-attacker ablation (ties in Idea A)
- [ ] Add **greedy coordinate search** as a second attacker (cheap, black-box, no gradient plumbing — often beats CEM in success at higher query cost).
- [ ] Optional: **GCG-on-surrogate** (gradients on a downloaded copy of the embedder, transfer to target) — connects to Phase 5 transferability.
- [ ] Claim to establish: provenance defense is **attack-agnostic** — a better optimizer makes a better trigger, but it's still an external doc, so the defense shouldn't care.

### Phase 5 — Learned trust / data-driven provenance (Idea B, done right)
- [ ] Trust predictor `t̂(d) ∈ [0,1]` from **source-side, non-forgeable features only** (no content).
- [ ] Keep **β as the global operating knob**; use calibrated `t̂(d)` (and optionally classifier confidence) in the penalty.
- [ ] Construct realistic simulated source metadata for BEIR (or find a provenance-bearing corpus).

### Phase 6 — Learned defense vs. adaptive attacker (the payoff)
- [ ] Run the **same adaptive attacker** against the **learned** defense, now including **trust-spoofing** (attacker optimizes to fool the trust model).
- [ ] Final table: RSR: no defense → static trust → learned trust → learned trust vs. spoofing-aware attacker.
- [ ] Hits **both surviving research gaps**: provenance-aware retrieval + query-distribution adaptive attacker.

### Cross-cutting / smaller to-dos
- [ ] **Sensitivity sweep over `external_frac`** (currently fixed at 0.3) so the recall floor and RSR aren't seen as hand-picked. Reviewer-defense.
- [ ] **Save result JSONs** for the SciFact Phase 2 & Phase 3 runs into `results/` (currently only toy/mini committed) so every deck number is backed by a file.
- [ ] **Token-level CEM re-eval on SciFact** — confirm/replace the Phase 2 table numbers with the token-level attacker's numbers.
- [ ] **Transferability** across embedders (bge-small, gte-small) — do triggers/defense transfer?
- [ ] **Embedding-model fingerprinting** (arXiv:2607.01276) — how the attacker would identify the target's embedder to begin with.
- [ ] **Writeup** positioning vs. **Semantic Chameleon** (arXiv:2603.18034) and **CRCP** (arXiv:2606.11265) for any hybrid/rerank claim.
- [ ] **Compile the progress deck** to PDF (Colab `texlive` cell or Overleaf).

---

## Optimizer options (reference for Idea A)

Threat model: black-box (no gradients on the *target*), discrete token space, each eval = one embedder forward pass. "Black-box" ≠ "no access" — the attacker can still query/run the embedder (or a downloaded copy) and compute the score themselves.

| Optimizer | Access needed | Notes |
|---|---|---|
| **CEM** (current) | query-only | Parallel (score whole batch), robust, few knobs. Solid baseline, not the ceiling. |
| **Greedy coordinate search** | query-only | Swap one token at a time, keep if better. Simple, often beats CEM in success; costs more queries. |
| **Genetic algorithm** | query-only | CEM + crossover; better at escaping local optima. |
| **Simulated annealing** | query-only | Greedy but sometimes accepts worse moves to avoid getting stuck. |
| **NES / SPSA (zeroth-order)** | query-only | Estimate gradient from finite differences; more query-hungry. |
| **GCG / HotFlip** | gradients (surrogate) | Strongest text-trigger method. Needs gradients → run on a downloaded copy of the embedder, transfer to target. Ties to Phase 5. |
| Bayesian optimization | query-only | Poor fit — chokes on high-dim discrete (10 pos × 30k vocab). |
| CMA-ES | query-only | Poor fit — built for continuous spaces, awkward on discrete tokens. |

---

## Concepts settled this chat (reference)

- **A RAG system has two separate models.** (1) The **embedder/retriever** turns text → **vectors** and decides which docs get retrieved. (2) The **LLM** reads the top-k and produces **text**. The attack targets model (1) only. Our project is **retrieval-only** — no LLM stage.
- **"Black-box" means no gradients/weights/internals — NOT no access.** The attacker can still query the embedder (or download the same open-source one) and observe outputs.
- **The embedder outputs vectors, not a relevance score.** The attacker computes the score themselves = dot product of query vector and (trigger+payload) vector (`encode(...) @ target`). The chat/LLM endpoint hides retrieval and only shows text; the *embeddings* endpoint returns raw vectors.
- **How CEM climbs:** it hill-climbs a *probability table* over tokens, not a single guess. Each round: sample candidates → score → keep the elite (top ~20%) → nudge the table toward them (with smoothing to avoid premature collapse). Expected score rises each round because the losers are discarded.
- **The reranker is not a defense** — it raises the mountain (attacker needs more budget: 100→72), it doesn't block the climb.
- **Provenance/trust comes from ingestion, not inference.** You know a doc's source because your system recorded it when the doc entered the index (source URL, upload channel, signature) — not because you read the text at retrieval time. Poison necessarily enters through an untrusted channel (that's what "indirect" means). Content is attacker-controlled; the ingestion channel is defender-controlled — that's the security argument. It fails only if metadata is forgeable (claimed author) or a trusted channel is compromised (→ trust-spoofing).

---

## Prior-work anchors
- Source paper: Chang, Bao, Luo, Yu — *Overcoming the Retrieval Barrier: IPI in the Wild for LLM Systems*, USENIX Security 2026 (arXiv:2601.07072). Attack only, no defense.
- Two surviving research gaps (from the gap audit): **provenance-aware retrieval** + **query-distribution (T2) defenses vs. adaptive attacker**.
- Differentiate from: Semantic Chameleon (arXiv:2603.18034), CRCP (arXiv:2606.11265).
- Embedding-model fingerprinting: arXiv:2607.01276.
