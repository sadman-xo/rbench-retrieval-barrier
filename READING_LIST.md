# Reading list — retrieval-stage IPI, corpus poisoning, provenance defenses

Compiled 2026-09-22 for the rbench thesis (see `ROADMAP.md`).
Read Tier 0 first. Tier 1 decides your novelty claim. Tier 2–4 support the method chapters.

**Prerequisites live in [`FOUNDATIONS.md`](FOUNDATIONS.md).** That file holds the
machinery (DPR, RAG, CEM, HotFlip, adaptive-attack methodology, Biba integrity).
This file holds the problem. Read `FOUNDATIONS.md` first if the machinery is new to you.

Status legend: `[ ]` to read · `[x]` read

---

## Tier 0 — The spine (read these five first)

These five give you the whole problem statement. Read them in this order.

- [ ] **Greshake et al., "Not what you've signed up for: Compromising Real-World
      LLM-Integrated Applications with Indirect Prompt Injection"** — arXiv:2302.12173,
      AISec 2023.
      Why: this paper defines indirect prompt injection. Every IPI paper cites it.
      Take from it: the threat-model vocabulary (injection channel, payload, delivery).

- [ ] **Zhong, Huang, Wettig, Chen, "Poisoning Retrieval Corpora by Injecting
      Adversarial Passages"** — arXiv:2310.19156, EMNLP 2023.
      Why: this is the direct ancestor of your source paper. It optimizes discrete
      tokens with HotFlip to maximize similarity to a query set. It then transfers to
      unseen queries.
      Take from it: your T2 query-distribution setting comes from here. Their result
      (transfer works with ~500 passages) is the contrast for your 0% T2 result with
      one passage.

- [ ] **Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to RAG"** —
      arXiv:2402.07867, USENIX Security 2025.
      Why: it separates the retrieval condition from the generation condition.
      Take from it: the standard RSR/ASR metric split. It justifies your
      retrieval-only scope.

- [ ] **Xiang, Wu et al., "Certifiably Robust RAG against Retrieval Corruption"** —
      arXiv:2405.15556, ICML 2024.
      Why: an isolate-then-aggregate defense with a formal certificate.
      Take from it: the certificate template. Your roadmap wants a certificate for the
      provenance defense. Copy their proof structure, not their mechanism.

- [x] **Chang, Bao, Luo, Yu, "Overcoming the Retrieval Barrier"** — arXiv:2601.07072,
      USENIX Security 2026. Your source paper. Already read and audited.

---

## Tier 1 — Closest competitors to the provenance defense (cite and differentiate)

Read all six. Your contribution lives in the gaps between them.

- [ ] **Koganti, Garrido-Lestache Belinchon, "Source-Aware Reranking for RAG:
      A Reliability Prior Approach"** — arXiv:2607.22584 (Jun 2026). **VERIFIED.**
      This is the nearest neighbour to your Phase 3. They use a multiplicative prior:
      `score = sim(q,d) * lambda(s)`. You use an additive penalty:
      `score - beta*(1-trust)*spread`.
      **Your delta:** they use 120 documents, one health corpus, and no optimized
      attacker. You use BEIR-scale SciFact, a token-level CEM attacker, a GCS attacker,
      and a defense-aware adaptive attacker. State this difference explicitly.

- [ ] **"Utility Under Attack"** — arXiv:2608.21230 (Aug 2026).
      Already flagged in your notes. It finds the same security/utility trade-off with
      additive provenance weighting on agent memory. Its attacker does not optimize.
      Its proposed fix (bounded occupancy) is unbuilt.
      **Your delta:** optimized and adaptive attackers, plus building the occupancy cap.

- [ ] **Yin, Qi, Cheng, "ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever
      RAG from Corpus Poisoning"** — arXiv:2603.22934 (Mar 2026). **VERIFIED.**
      A training-free retriever-stage defense. It perturbs parameters and reranks on
      instability signals. It claims evaluation against adaptive evasive attacks.
      **Use it as the content-based baseline.** Your central claim is "source-based
      beats content-based". ProGRank is the strongest content-based opponent.
      Consider implementing it as a comparison row.

- [ ] **"TrustRAG: Blockchain-Enhanced RAG via Committee-Based Credibility Scoring"** —
      arXiv:2608.20097 (Aug 2026).
      Credibility scoring with heavy trust infrastructure.
      Use it to argue that your trust signal is cheap and your defense is one line.

- [ ] **"PRA-RAG: Provably Robust Aggregation in RAG against Retrieval Corruption"** —
      arXiv:2607.00012 (Jul 2026).
      A second certificate template, newer than RobustRAG. Read it with RobustRAG.

- [ ] **"Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow
      Control"** — arXiv:2605.26754 (May 2026).
      Information-flow control is the systems-level version of provenance.
      Use it to place your ranking-level defense inside a larger architecture.

---

## Tier 2 — Attack side (supports Phase 4b and Phase 7)

- [ ] **Zou, Wang, Carlini, Nasr, Kolter, Fredrikson, "Universal and Transferable
      Adversarial Attacks on Aligned Language Models" (GCG)** — arXiv:2307.15043.
      The standard token optimizer. Read it before you write Phase 7a.
      Take from it: the multi-model ensemble recipe for transferable triggers.

- [ ] **"Universal Adversarial Triggers Are Not Universal"** — arXiv:2404.16020.
      **Read this early.** It refutes broad trigger-transfer claims.
      It supports your Phase 4 finding that T2 universal triggers reach 0%.
      This turns a negative result into a citable, expected result.

- [ ] **"Corpus Poisoning via Approximate Greedy Gradient Descent" (AGGD)** —
      arXiv:2406.05087.
      A stronger optimizer than HotFlip for the same task. It is the natural third
      attacker for your Phase 4b ablation (CEM vs GCS vs AGGD).

- [ ] **"Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval"** —
      arXiv:2501.04802, ECIR 2025.
      A reproduction study. Use it to set honest baselines. It also defends your own
      reimplementation choice (see the `authors-artifact` license note).

- [ ] **"Joint-GCG: Unified Gradient-Based Poisoning Attacks on RAG"** —
      arXiv:2506.06151.
      It attacks retriever and generator together. Cite it to justify your
      retrieval-only scope, and to name the obvious extension.

- [ ] **"Micro-Collaborative Poisoning: A Distributed Attack on RAG Systems"** —
      arXiv:2609.21573 (Sep 2026).
      Multiple poison documents. This is your listed gap #2. Check if it closes it.

- [ ] **Semantic Chameleon** — arXiv:2603.18034. Corpus-dependent poisoning and defense.
- [ ] **CRCP, "When Poison Fails After Retrieval"** — arXiv:2606.11265.
      Chunking and reranking pipelines. Both are already flagged in your notes.
      Read them together. They own the chunking angle.

---

## Tier 3 — Adversarial information retrieval (the field your work really sits in)

An IR reviewer will ask why you do not cite these. Read PRADA and TrustRank at minimum.

- [ ] **Wu et al., "PRADA: Practical Black-box Adversarial Attacks against Neural
      Ranking Models"** — arXiv:2204.01321, TOIS 2023.
      Black-box, decision-based, surrogate-driven ranking attack.
      Take from it: the "word substitution ranking attack" task definition, and the
      web-spam framing. Your case-3 attacker (Phase 7b) is their decision-based setting.

- [ ] **"Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models"** —
      arXiv:2304.14867. The query-distribution version. This is your T2 setting in IR form.

- [ ] **"Multi-granular Adversarial Attacks against Black-box Neural Ranking Models"** —
      arXiv:2404.01574.

- [ ] **"Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view
      Contrastive Learning Method"** — arXiv:2308.09861.

- [ ] **Gyöngyi, Garcia-Molina, Pedersen, "Combating Web Spam with TrustRank"** —
      VLDB 2004.
      Why: this is the classical precedent for your whole defense. It propagates trust
      from a seed set instead of scoring content.
      **Framing gift:** "provenance-weighted retrieval is TrustRank for RAG."
      Also read **"Topical TrustRank"** for the trust-plus-relevance combination.

---

## Tier 4 — Benchmarks, agent context, and IR foundations

Skim these. Cite them for setup and motivation.

- [ ] **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR** —
      arXiv:2104.08663. Your datasets (SciFact, NFCorpus, FiQA) come from here.
- [ ] **"In Defense of Cross-Encoders for Zero-Shot Retrieval"** — arXiv:2212.06121.
      It justifies your reranker stage.
- [ ] **BIPIA, "Benchmarking and Defending Against Indirect Prompt Injection"** —
      arXiv:2312.14197, KDD 2025. The standard IPI benchmark.
- [ ] **"AutoDojo: Adaptive Black-Box Attacks Reveal the Limits of IPI Defenses"** —
      arXiv:2606.15057 (Jun 2026).
      **Methodologically important.** It makes the same argument as your Phase 4:
      a defense is only proven by a defense-aware attacker. Cite it for your method.
- [ ] **"Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense"** —
      arXiv:2606.30783 (Jun 2026).
      Your beta sweep is exactly this trade-off curve. Use its vocabulary.
- [ ] **"Indirect Prompt Injection in the Wild: An Empirical Study"** —
      arXiv:2604.27202 (Apr 2026). Prevalence data for your motivation chapter.
- [ ] **"LivePI: More Realistic Benchmarking of Agents Against IPI"** — arXiv:2605.17986.
- [ ] **Beurer-Kellner et al., "Design Patterns for Securing LLM Agents against Prompt
      Injections"** — arXiv:2506.08837.
- [ ] **Debenedetti et al., "Defeating Prompt Injections by Design" (CaMeL)** —
      arXiv:2503.18813.
      CaMeL tracks data provenance and enforces capability policies.
      Use it to show that provenance already works at the agent layer. Your work moves
      the same idea down into the ranking function.

---

## How to read this efficiently

1. Read Tier 0. Write one paragraph per paper: threat model, defense, metric.
2. Read Tier 1. Build a table with these columns: paper, trust signal, penalty form,
   corpus size, attacker strength, adaptive attacker yes/no. Your row is the last row.
3. Read the two framing papers early: arXiv:2404.16020 (universal triggers fail) and
   TrustRank. They turn two of your results into expected, well-grounded findings.
4. Read Tier 2 and Tier 3 while you build Phases 5–7.
5. Skim Tier 4 last, for the introduction and the setup chapter.

## Note on identifiers

All arXiv identifiers here came from a live search on 2026-09-22.
Two were opened and confirmed in full: arXiv:2607.22584 and arXiv:2603.22934.
Confirm the rest when you download them. Titles are more reliable than numbers.
