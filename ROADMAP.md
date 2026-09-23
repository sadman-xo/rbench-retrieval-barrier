# rbench — Roadmap & Open Ideas

Living notes for the retrieval-barrier IPI attack/defense project (arXiv:2601.07072).
Captures decisions and ideas from the 2026-09-12 planning chat. Nothing here is dropped —
"someday/maybe" items are kept in their own section rather than deleted.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Where we are (done)

- [x] **Phase 1 — Testbed.** RAG retriever, switchable dense-only vs. hybrid(dense+BM25)+cross-encoder rerank; incremental poison insertion for BEIR-scale corpora; `--device auto` (CPU↔T4).
- [x] **Phase 2 — Attack.** Black-box, token-level CEM trigger optimizer (faithful to the USENIX'26 method, independently implemented). SciFact (5k docs, 50 q): none=0%, cem=**100% dense / 72% hybrid+rerank**, naive=100%/100%. Honest recall@5=0.71.
- [x] **Phase 3 — Defense.** Provenance-weighted retrieval: `score' = score − β·(1−trust)·spread`. SciFact β-sweep (confirmed real): β=0 → RSR 100%/100%; **β=0.5 (sweet spot) → 8% naive / 6% cem** *(2026-09-24: this "sweet spot" is almost hard exclusion — ext_recall ≈ 0 at β = 0.5 on three other embedders; see Phase 4c)*, recall 0.71→0.50; β≥0.75 → 0%. Recall floor ~0.50 is structural (30% of relevant docs are legitimately external).
- [x] **Progress deck** — `slides/rbench_progress.tex` (Beamer, Phases 1–3). Compile on Colab/Overleaf (no LaTeX locally).

- [x] **Phase 4 — Adaptive attacker (the contribution).** Defense-aware CEM + T2 query-distribution training. SciFact results (β=0.5, 35 train / 15 held-out test):
  - Per-query: no defense → static CEM 73.3% RSR; defense → static 0%, **adaptive 0%**. Knowing the defense gives the attacker zero advantage.
  - T2 universal trigger: 0% RSR even without defense — a single trigger cannot generalize across queries.
  - **Theoretical confirmation:** the provenance penalty is monotonic in raw similarity (dense-only), so the adaptive objective is equivalent to the static one. The defense is structurally robust, not security-through-obscurity.

Key findings:
1. **Source-based (provenance) beats content-based (reranker)** — the reranker only raises the attacker's budget (100→72) and misses the naive attack; provenance collapses naive AND cem together (attack-agnostic).
2. **Defense holds at 0% RSR against the adaptive attacker** — the attacker cannot exploit knowledge of the defense because the penalty is monotonic in the score it already maximizes. *Caveat (2026-09-24, Phase 4c): the 0% is against CEM, which is WEAKER than the naive query echo on Contriever/TAS-B; the echo gets 20–47% at β = 0.5. On SciFact, 0% RSR for every attacker needs β ≥ 0.75, where no external evidence is retrieved (hard exclusion).*
3. **Universal triggers fail** — per-query optimization is required, which demands the attacker know the exact query.

---

## What I want to do 

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

## proposed to-dos

### Phase 4 — Defense-aware adaptive attacker (the actual contribution)
- [x] New `rbench/attack/adaptive.py`: reuse the optimizer loop but change the **objective to the post-defense score** (`score − β·(1−trust)·spread`), so the attacker knows it's penalized and tries to push similarity high enough to survive.
- [x] Optimize over a **query distribution (T2)**: train one trigger on a train split, evaluate RSR on **held-out** queries.
- [x] Report the three-column story: **RSR: no defense → defense (static attacker) → defense (adaptive attacker).**
- [x] **Outcome: defense holds.** Adaptive attacker achieves 0% RSR — same as static. Provenance defense is structurally robust (monotonicity argument confirmed empirically). The attacker's next move is trust-spoofing (Phase 6).

### Phase 4b — Stronger-attacker ablation (ties in Idea A)
- [x] Add **greedy coordinate search** as a second attacker (cheap, black-box, no gradient plumbing — often beats CEM in success at higher query cost).
- [ ] Optional: **GCG-on-surrogate** (gradients on a downloaded copy of the embedder, transfer to target) — connects to Phase 5 transferability.
- [x] **Claim established:** provenance defense is **attack-agnostic**. SciFact results: GCS finds higher-scoring triggers than CEM (0.74 vs 0.51 avg raw score, 100% vs 73% no-defense RSR), uses 12x more queries, but **both hit 0% RSR against the defense**. A better optimizer does not help — the poison is external, and the defense penalizes origin, not content.

### Phase 4c — Norm inflation vs. the defense (dot-product retrievers)
**Hypothesis.** The Phase 4 robustness needs *bounded* similarity. rbench uses unit-norm embeddings (cosine ≤ 1). Dot-product retrievers (Contriever, TAS-B, DPR, ANCE) score `|e_d|·|q|·cos θ`, so an attacker can grow `|e_d|`. For β < 1 the defended poison score `(1−β)·s_p + β·m` then grows without limit → the soft penalty should break. Only β ≥ 1 stays safe, and β ≥ 1 is hard exclusion of all external docs.

**Prior work — cite, do not claim.** Zhong et al., EMNLP 2023 (arXiv:2310.19156) already showed that white-box HotFlip passages against Contriever have large ℓ2 norms (their Fig. 4) and that clipping ALL passage norms stops the attack at a small recall cost (their Table 5, α sweep). **Our delta:** (1) the interaction with the provenance penalty; (2) a black-box token-level CEM attacker (the source paper's threat model), not white-box; (3) the cap applies to UNTRUSTED docs only, fitted on trusted norms, so trusted docs keep their scores; (4) a per-query certificate.

**Certificate (new, `rbench/defense/certify.py`).** If the poison's raw score is bounded by `B` (B = |q| for cosine, `cap·|q|` with the norm cap, ∞ for raw dot product), then its defended score is at most `m + (1−β)·(B−m)`. If k trusted docs score above that, NO poison text can reach top-k — for any optimizer, white-box included. This answers the "maybe a stronger attacker exists" objection (Tramèr et al.) with a proof, not more attacks. `1 − certified fraction` = the RSR ceiling for any attacker.

- [x] `norm_cap_pct` config + untrusted-norm cap in the pipeline; `with_config()` to evaluate a (β, cap) grid on one index.
- [x] Cap-aware adaptive attacker (`norm_cap=` in `adaptive.py`). Under the cap, extra norm is worthless, so this objective is genuinely different from the static one.
- [x] `rbench/attack/run_norm.py`: norm stats, utility + certified fraction per cell, per-query + T2 attacks on the full grid.
- [x] `tests/test_norm_defense.py` (fake embedders, no downloads): cap logic, the hypothesis in miniature, certificate soundness against the real pipeline code, end-to-end toy run. 10/10 pass.
- [x] **Run on SciFact (2026-09-24, T4)**: 5000 docs (1475 external), 50 queries (35 train / 15 test), k=5, CEM 12 tokens × 40 iters × 96 samples, cap = p99 of trusted norms. No certificate violations in any arm.

**Results — RSR@5 on 15 held-out queries at β = 0 / 0.25 / 0.5 / 0.75 / 1.0 (cap OFF; cap ON gave the SAME attack numbers in every arm)**

| Arm | naive | static CEM | adaptive CEM | T2 (both) | ceiling, cap off → on (β=0.5) |
|---|---|---|---|---|---|
| Contriever-ms, dot | 100 / 93 / **40** / 0 / 0 | 60 / 7 / 0 / 0 / 0 | 47 / 7 / 0 / 0 / 0 | 0 everywhere | 100% → 93% |
| Contriever-ms, cosine (control) | 100 / 100 / **47** / 0 / 0 | 47 / 0 / 0 / 0 / 0 | = static (identical run) | 0 everywhere | 53% → 53% |
| TAS-B, dot | 100 / 93 / **20** / 0 / 0 | 47 / 0 / 0 / 0 / 0 | 47 / 0 / 0 / 0 / 0 | 0 everywhere | 100% → 93% |

**Utility — recall / ext_recall (recall of relevant EXTERNAL docs), 50 queries**

| β | Contriever dot | Contriever cosine | TAS-B dot |
|---|---|---|---|
| 0 | 0.68 / 0.69 | 0.74 / 0.68 | 0.63 / 0.60 |
| 0.25 | 0.56 / 0.19 | 0.64 / 0.14 | 0.57 / 0.27 |
| 0.5 | 0.52 / **0.06** | 0.61 / **0.00** | 0.49 / **0.00** |
| ≥ 0.75 | 0.50 / 0.00 | 0.61 / 0.00 | 0.49 / 0.00 |

Certified share (50 queries): raw dot, cap off → 0% for every β < 1, 100% at β = 1 (as the theory says). Cap on → 2% at β = 0.5, **100% at β = 0.75**. Cosine → 42% at β = 0.5, 100% at β ≥ 0.75.

**Findings**
1. **Hypothesis not confirmed for this attacker.** Black-box CEM inflates norms only a little: static poisons sit at 0.97× the cap on average (max 1.02×; 20% above p99 on Contriever, 0% on TAS-B). Zhong's white-box passages reached ~1.3×. Dot product did not raise RSR over cosine, and the cap changed no RSR number. The door stays open in theory: under raw dot product NOTHING is certified for β < 1, so a stronger attacker could still use norm. The cap closes it provably from β = 0.75 (dot, cap on: 100% certified) instead of only β = 1.
2. **The naive query echo is the strongest attacker, and it beats β = 0.5** (40% / 47% / 20%). CEM triggers reach cosine ≈ 0.5 with the query; the echo reaches ≈ 0.9. Our CEM budget (3,840 scored candidates per trigger) is ~40× below the paper's (30 × 5000 = 150,000). So CEM "0%" numbers overstate the defense. On MiniLM (Phase 3) naive was 8% at β = 0.5 — the β = 0.5 result does not carry over to other embedders.
3. **There is no soft regime on SciFact.** RSR reaches 0 for every attacker only where ext_recall is ≈ 0 (β ≥ 0.75). Where external evidence still gets through (β = 0.25, ext_recall 0.14–0.27), the naive echo gets 93–100%. The Phase 3 "sweet spot" β = 0.5 has ext_recall 0.06 / 0.00 / 0.00 — it is (almost) hard exclusion. This independently reproduces the trade-off in arXiv:2608.21230 (a weight big enough to stop the poison excludes all untrusted evidence), now with optimized attackers, three embedders, and a certificate.
4. **The certificate is close to tight.** Cosine arm, β = 0.5: ceiling 53%, naive reaches 47%.
5. **Rank-invariance confirmed.** Under cosine the adaptive run is identical to the static run (same RSR, same cosine 0.520, same T2 trigger): CEM uses only ranks, and the adaptive objective is a monotone transform of the raw score.
6. **T2 universal triggers: 0% on all three models, even with no defense.**
7. **Sample size.** 15 test queries: 0/15 only shows RSR < 20% at 95% confidence (Wilson); 6/15 = 40% has a CI of about 20–64%.

**Next (in order)**
- [ ] **Query-echo + CEM suffix attacker** — start from the query text (cosine ≈ 0.9) and let CEM add tokens. This is the right test of the norm hypothesis (high angle AND extra norm) and the strongest cheap attacker we have. Compare dot vs. cosine vs. cap on.
- [ ] **More queries** — per-query attacks on all 300 SciFact test queries (cheap on the H100), so the CIs shrink.
- [ ] **Cap percentile sweep** (p50 / p75 / p90 / p99) — utility + certificate only, no attacks. A p99 cap still lets a poison carry ~12% more norm than a typical doc; a tighter cap should move dot-product certification toward the cosine arm.
- [ ] **Bounded occupancy (the unbuilt fix in 2608.21230)** — give untrusted docs at most j of the k slots. Security becomes "the poison takes at most j slots; trusted evidence always keeps k − j", which can admit external evidence AND be certified. This is the candidate contribution now that finding 3 rules out the soft penalty.
- [ ] Follow-up: compare against Zhong's GLOBAL clip (all docs) at the same cap — the utility gap is the value of conditioning the cap on provenance.

### Phase 5 — Learned trust / data-driven provenance (Idea B, done right)
- [ ] Trust predictor `t̂(d) ∈ [0,1]` from **source-side, non-forgeable features only** (no content).
- [ ] Keep **β as the global operating knob**; use calibrated `t̂(d)` (and optionally classifier confidence) in the penalty.
- [ ] Construct realistic simulated source metadata for BEIR (or find a provenance-bearing corpus).

### Phase 6 — Learned defense vs. adaptive attacker (the payoff)
- [ ] Run the **same adaptive attacker** against the **learned** defense, now including **trust-spoofing** (attacker optimizes to fool the trust model).
- [ ] Final table: RSR: no defense → static trust → learned trust → learned trust vs. spoofing-aware attacker.
- [ ] Hits **both surviving research gaps**: provenance-aware retrieval + query-distribution adaptive attacker.

### Phase 7 — Attacker-knowledge track (transferability & unknown embedder)
Two SEPARATE contributions — do not fuse them. Both attack the paper's "trigger is not
transferable across architectures" limitation, but from opposite directions:
one generalizes across models, the other sidesteps the need to know the model at all.

**Phase 7a — Ensemble-surrogate transferability (the direct rebuttal).**
- Goal: produce ONE trigger that works on an *unseen* embedder without re-optimizing — the real "transferable trigger" the paper says doesn't exist.
- Method: optimize the trigger against an **ensemble of surrogate embedders** at once; a trigger forced to satisfy many architectures tends to transfer to a held-out one (standard transferable-adversarial-example / GCG recipe).
- rbench changes:
  - [ ] `EnsembleEmbedder` wrapper that encodes with N models (e.g. `bge-small`, `gte-small`, `e5-small`).
  - [ ] CEM `score_fn` = aggregate of **per-model normalized** scores — use `min` (robustness: trigger must satisfy the weakest model) or mean; z-score/rank-normalize per model so different score scales don't let one model dominate.
  - [ ] `--surrogates A,B,C --target D` split: optimize on {A,B,C}, evaluate RSR on **held-out** D (e.g. `all-MiniLM`) never seen in optimization.
- Metric: RSR on held-out D — **ensemble-trained trigger vs. single-model-trained trigger** (train-on-A→test-on-D). The gap = the transferability the method buys. Report as a transfer matrix (extends the paper's single-dataset Figure 5 into a *method*, not just a measurement).
- Claim if it works: "cross-architecture transfer, the paper's stated limitation, is achievable via ensemble-surrogate optimization."

**Phase 7b — Case-3 retrieval-outcome-only attacker (the new threat model).**
- Goal: attack a target whose embedder you **cannot query at all** — optimize using only the target pipeline's **retrieval outcomes** (rank / retrieved-or-not). This is NOT transfer; it's re-optimizing directly on the unknown target via weak feedback. Genuinely untested (confirmed: the paper requires score access under budget `B`).
- rbench changes:
  - [ ] CEM `score_fn` swaps the embedder dot-product for a **rank-based reward** from `pipeline.search` (already returns ranks): e.g. `reward = (k - rank + 1)` if in top-k else 0, or a smoother graded rank signal.
  - [ ] Handle the **sparse/flat-reward cold start**: warm-start from a Query+ trigger so early candidates already land in top-k and produce a usable gradient of reward; otherwise all candidates score 0 and CEM has no elites to select.
  - [ ] Log **query budget** (# pipeline queries) vs. RSR to quantify the "cost of blindness."
- Metric: RSR and query-count for case-3 vs. case-1 (full score access). The delta = what score-access is worth to the attacker.
- Honest either way: works but is far more query-hungry (shows score access is the real enabler) OR barely works (shows the retrieval-outcome signal is too weak) — both are results.
- Caveat to state: assumes the attacker can *observe* retrieval outcomes; in reality they often see only the LLM's text answer, which hides retrieval — so this is an upper bound on the case-3 attacker.

**7a vs 7b in one line:** 7a = *one trigger, many models, no re-opt*; 7b = *re-optimize per target, no model knowledge, weak feedback.*

### Cross-cutting / smaller to-dos
- [ ] **Sensitivity sweep over `external_frac`** (currently fixed at 0.3) so the recall floor and RSR aren't seen as hand-picked. Reviewer-defense.
- [ ] **Save result JSONs** for the SciFact Phase 2 & Phase 3 runs into `results/` (currently only toy/mini committed) so every deck number is backed by a file.
- [ ] **Token-level CEM re-eval on SciFact** — confirm/replace the Phase 2 table numbers with the token-level attacker's numbers.
- [ ] **Transferability** across embedders (bge-small, gte-small) — now split into its own **Phase 7a** (ensemble-surrogate → held-out model); also check whether the *defense* transfers.
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

## Untested attack dimensions (VERIFIED against the full paper, 2026-09-13)

Checked against the actual arXiv:2601.07072 PDF (35pp), not just the deck. Split into
what the paper already covers (do NOT claim as gaps) vs. what genuinely survives.

**Already tested by the paper — NOT gaps:**
- Query budget / cost — central to the method (budget `B`, cost `O(log|V|·log 1/ε)`, ~$0.21/query).
- Trigger fragment length — Figure 2.
- Cross-model transferability — Figure 5 (on FiQA). Deeper/model-agnostic transfer is future work, but an experiment exists → soften Phase 5 framing.
- Adaptive attacker vs. their 3 simple defenses — they already break paraphrasing/perplexity/masking. Our Phase 4 novelty must be adaptive-vs-**provenance** at the query-distribution level.
- Trigger position — position-agnostic (Figure 6); position defenses excluded by design.

**Threat model (verified):** black-box QUERY access to the embedder returning embeddings/scores (cases 1 open-source copy + 2 proprietary API); single injected item; no corpus/param access. Defenses tested = exactly THREE (paraphrasing, perplexity filtering, token masking); NO rerank/hybrid, NO provenance; fine-tuning defenses (SecAlign/DataSentinel/StruQ) scoped out. Retrieval is DENSE-only.

**Confirmed untested — real gaps (ranked):**
1. **Document chunking at ingestion** — poison is one atomic item; no passage-splitting anywhere. Strongest, most realistic. Feasible in rbench.
2. **Multiple poison docs / corpus saturation** — explicitly restricted: *"inject only a single malicious item."* Untested by design.
3. **Payload `Dadv` construction & length + retrieval-vs-obedience tension** — explicitly scoped out (*"do not study the construction process or the downstream effect of Dadv"*). They test trigger length, not payload length.
4. **Ingestion preprocessing/normalization robustness** (lowercasing, Unicode norm, stopword/punct stripping) — token masking is adjacent but not the same.
5. **Temporal durability** as the corpus grows.
6. **Retrieval-outcome-only weak attacker (case 3)** — no score access, only "was it retrieved." Never considered; harder attacker.

**Our confirmed-novel space:** provenance defense (untested), hybrid+rerank robustness (their own Research Gap 1, dense-only in their eval), and adaptive-vs-provenance @ T2.

## Prior-work anchors
- Source paper: Chang, Bao, Luo, Yu — *Overcoming the Retrieval Barrier: IPI in the Wild for LLM Systems*, USENIX Security 2026 (arXiv:2601.07072). Attack only, no defense.
- Two surviving research gaps (from the gap audit): **provenance-aware retrieval** + **query-distribution (T2) defenses vs. adaptive attacker**.
- Differentiate from: Semantic Chameleon (arXiv:2603.18034), CRCP (arXiv:2606.11265).
- Embedding-model fingerprinting: arXiv:2607.01276.
