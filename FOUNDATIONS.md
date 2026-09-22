# Foundations — the prerequisite papers

Compiled 2026-09-22 for the rbench thesis.
`READING_LIST.md` holds the **problem** (who attacked what, who defended it).
This file holds the **machinery** (why your code works and why your claim is valid).

Read this file first if you have not read it. Each paper here is a thing you already
use in `rbench/`, or a rule your main claim depends on.

Status legend: `[ ]` to read · `[x]` read

---

## Read these six before anything else

If you read nothing else this week, read these. They carry your whole method chapter.

1. Tramèr et al., **On Adaptive Attacks to Adversarial Example Defenses** (F3)
2. de Boer et al., **A Tutorial on the Cross-Entropy Method** (F5)
3. Wallace et al., **Universal Adversarial Triggers** (F4)
4. Karpukhin et al., **Dense Passage Retrieval** (F1)
5. Lewis et al., **Retrieval-Augmented Generation** (F2)
6. Biba, **Integrity Considerations for Secure Computer Systems** (F6)

---

## F1 — Retrieval machinery (what your pipeline is made of)

Every item below is a component in `rbench/retriever/`.

- [ ] **Karpukhin et al., "Dense Passage Retrieval for Open-Domain QA"** —
      arXiv:2004.04906, EMNLP 2020.
      **Read this.** It defines the dual-encoder with dot-product scoring. Your whole
      attack surface is this design. Everything after it is a variation.

- [ ] **Reimers & Gurevych, "Sentence-BERT"** — arXiv:1908.10084, EMNLP 2019.
      This is the `sentence-transformers` library you import. Cite it for the
      `Embedder` wrapper.

- [ ] **Izacard et al., "Unsupervised Dense Information Retrieval with Contrastive
      Learning" (Contriever)** — arXiv:2112.09118.
      **Important for one open hypothesis.** Contriever uses unnormalized dot product.
      Your pipeline sets `normalize_embeddings=True`, so similarity is bounded.
      Your notes raise the question of whether an attacker can inflate the embedding
      norm on an unnormalized retriever. Read this before you test that.

- [ ] **Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"** —
      FnTIR 3(4), 2009.
      The basis of your BM25 branch in the hybrid mode. Skim sections 1–3 only.

- [ ] **Nogueira & Cho, "Passage Re-ranking with BERT"** — arXiv:1901.04085.
      The cross-encoder reranker you run in hybrid mode. Short paper.

- [ ] **Thakur et al., "BEIR"** — arXiv:2104.08663, NeurIPS 2021 Datasets.
      Your SciFact, NFCorpus and FiQA datasets come from here. Cite it for setup.

- [ ] **Li et al., "Towards General Text Embeddings with Multi-stage Contrastive
      Learning" (GTE)** — arXiv:2308.03281.
      The model family of `gte-modernbert-base`, the source paper's default embedder.

- [ ] **Xiao et al., "C-Pack: Packed Resources for General Chinese Embeddings" (BGE)** —
      arXiv:2309.07597.
      The `bge-small` model in your Phase 7a transfer plan. Skim.

- [ ] **Muennighoff et al., "MTEB: Massive Text Embedding Benchmark"** —
      arXiv:2210.07316. Optional. Use it to justify which embedders you picked.

---

## F2 — RAG machinery (what the system is)

- [ ] **Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
      Tasks"** — arXiv:2005.11401, NeurIPS 2020.
      **Read this.** It names the architecture you attack. Your introduction needs it.

- [ ] **Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training"** —
      arXiv:2002.08909. Optional. The earlier end-to-end version.

- [ ] **Gao et al., "Retrieval-Augmented Generation for LLMs: A Survey"** —
      arXiv:2312.10997. Skim only. Use it for the pipeline-stage vocabulary
      (ingestion, chunking, indexing, retrieval, reranking, generation).

---

## F3 — Adversarial-robustness methodology (your claim depends on this)

This group is the most important one in the file. Your central claim is
"the defense holds against a defense-aware adaptive attacker". These papers set the
rules for making that claim. A reviewer will check your work against them.

- [ ] **Tramèr, Carlini, Brendel, Madry, "On Adaptive Attacks to Adversarial Example
      Defenses"** — arXiv:2002.08347, NeurIPS 2020. **VERIFIED.**
      **Read this first.** They break 13 published defenses that all claimed adaptive
      evaluation. The paper is a methodology guide: how to build an attack that truly
      adapts to the defense.
      **Use it as a checklist for Phase 4.** Your monotonicity argument is strong, but
      you must show you tried the attacks this paper lists. Otherwise your 0% result
      reads as a weak attacker, not a strong defense.

- [ ] **Carlini et al., "On Evaluating Adversarial Robustness"** — arXiv:1902.06705.
      The broader evaluation checklist. Read it with the paper above.
      Take from it: report attack budget, report failure cases, and never claim
      robustness from a single attack.

- [ ] **Athalye, Carlini, Wagner, "Obfuscated Gradients Give a False Sense of
      Security"** — arXiv:1802.00420, ICML 2018.
      **This is the source of your framing.** Your Phase 4 wording is "structurally
      robust, not security-through-obscurity". This paper defines that distinction.
      Cite it on that sentence.

- [ ] **Biggio & Roli, "Wild Patterns: Ten Years After the Rise of Adversarial Machine
      Learning"** — arXiv:1712.03141.
      The history and the threat-model taxonomy. Use it for your background chapter.

---

## F4 — Discrete text attacks (where the word "trigger" comes from)

- [ ] **Wallace, Feng, Kandpal, Gardner, Singh, "Universal Adversarial Triggers for
      Attacking and Analyzing NLP"** — arXiv:1908.07125, EMNLP 2019.
      **Read this.** It introduces the input-agnostic trigger, optimized over a
      distribution of inputs. Your T2 setting is this idea moved to retrieval.
      Your 0% T2 result is a direct negative answer to their premise. That is a result
      worth stating plainly.

- [ ] **Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text
      Classification"** — arXiv:1712.06751, ACL 2018.
      The gradient-guided token flip. It is the ancestor of GCG, AGGD, and of your GCS.

- [ ] **Shin et al., "AutoPrompt"** — arXiv:2010.15980, EMNLP 2020.
      The bridge from HotFlip to GCG. Short. Read it if you build Phase 7a.

---

## F5 — The optimizers you actually implemented

- [ ] **de Boer, Kroese, Mannor, Rubinstein, "A Tutorial on the Cross-Entropy
      Method"** — Annals of Operations Research 134(1):19–67, 2005. **VERIFIED.**
      PDF: https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf
      **You must cite this.** It is the basis of `rbench/attack/cem.py`.
      Take from it: the elite fraction, the smoothing parameter, and the convergence
      argument. Your defaults (`elite_frac=0.2`, `smoothing=0.55`) come from this
      family of methods. The tutorial tells you why they work and when they collapse.
      Sections 2 and 4 cover combinatorial optimization. Read those.

- [ ] **Rubinstein, "The Cross-Entropy Method for Combinatorial and Continuous
      Optimization"** — Methodology and Computing in Applied Probability, 1999.
      Optional. The original paper. Cite the tutorial instead.

Note on GCS: greedy coordinate search has no single canonical paper. Cite HotFlip and
AutoPrompt as its discrete-text ancestors, and describe your variant in full.

---

## F6 — Poisoning and trust foundations

- [ ] **Biba, "Integrity Considerations for Secure Computer Systems"** —
      MITRE Technical Report MTR-3153, 1977.
      **Read this.** It is the formal precedent for your entire defense.
      Biba integrity says a high-integrity process must not read low-integrity data.
      Your provenance penalty is a **soft, continuous Biba policy applied to ranking**.
      That sentence is a strong framing for your thesis. It also names your failure
      mode: the policy fails when integrity labels are forgeable, which is exactly
      your Phase 6 trust-spoofing threat.

- [ ] **Denning, "A Lattice Model of Secure Information Flow"** — CACM 19(5), 1976.
      Optional. Read it if you extend toward Cordon-MAS style information-flow control.

- [ ] **Biggio, Nelson, Laskov, "Poisoning Attacks against Support Vector Machines"** —
      arXiv:1206.6389, ICML 2012.
      The first data-poisoning paper. Background chapter.

- [ ] **Carlini et al., "Poisoning Web-Scale Training Datasets is Practical"** —
      arXiv:2302.10149, IEEE S&P 2024.
      **Use this for motivation.** It shows that injecting content into a real corpus
      is cheap and practical. It answers the reviewer question "is this threat real?".

- [ ] **Cohen, Rosenfeld, Kolter, "Certified Adversarial Robustness via Randomized
      Smoothing"** — arXiv:1902.02633, ICML 2019.
      The certification machinery. Read it before you attempt the certificate that
      your roadmap wants. Read it with RobustRAG and PRA-RAG from `READING_LIST.md`.

- [ ] **Kamvar, Schlosser, Garcia-Molina, "The EigenTrust Algorithm for Reputation
      Management in P2P Networks"** — WWW 2003.
      Optional. Reputation as a computed, non-content signal. It pairs with TrustRank.

- [ ] **Perez & Ribeiro, "Ignore Previous Prompt: Attack Techniques for Language
      Models"** — arXiv:2211.09527, NeurIPS 2022 workshop.
      The origin of prompt injection, before the "indirect" variant. One page of value.

---

## How these map to your chapters

| Chapter | Papers to cite |
|---|---|
| Introduction / motivation | F2 Lewis, F6 Carlini web-scale poisoning, F6 Perez |
| Background: retrieval | F1 all |
| Background: adversarial ML | F3 Biggio & Roli, F4 Wallace, F4 Ebrahimi |
| Method: attack | F5 CEM tutorial, F4 HotFlip, F4 AutoPrompt |
| Method: defense | F6 Biba, TrustRank (in `READING_LIST.md`) |
| Evaluation protocol | F3 Tramèr, F3 Carlini, F3 Athalye |
| Limitations / future work | F1 Contriever (norm inflation), F6 Cohen (certificate) |

## Note on identifiers

Two entries were confirmed by a live search on 2026-09-22: Tramèr et al.
(arXiv:2002.08347) and the Cross-Entropy Method tutorial.
The rest are well-known papers taken from memory. Confirm each number when you
download it. Search by title, not by number.
