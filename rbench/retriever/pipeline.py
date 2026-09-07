"""The retrieval pipeline: dense-only, or full hybrid (dense + BM25) + reranker.

The `hybrid` / `rerank` flags on RetrievalConfig switch between:
  * dense-only          — the base paper's setting (embedding similarity, top_k)
  * hybrid              — dense + BM25 fused into a candidate pool
  * hybrid + rerank     — candidate pool re-scored by a cross-encoder  <-- gap #1

Being able to run the SAME attack through all three is what lets this project say
something the base paper (dense-only) did not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RetrievalConfig
from .corpus import Doc
from .embedder import Embedder


@dataclass
class RetrievalResult:
    query_id: str
    ranked_ids: list[str]          # doc ids, best first (length <= top_k)
    scores: list[float]            # aligned with ranked_ids

    def rank_of(self, doc_id: str) -> int | None:
        """1-based rank of doc_id within the returned list, or None if not returned."""
        try:
            return self.ranked_ids.index(doc_id) + 1
        except ValueError:
            return None

    def contains(self, doc_id: str) -> bool:
        return doc_id in self.ranked_ids


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class RetrievalPipeline:
    """Indexes a corpus once; answers many queries.

    Rebuild the index whenever the corpus changes (e.g. after a poison doc is
    inserted or its trigger is re-optimized).
    """

    def __init__(self, embedder: Embedder, cfg: RetrievalConfig | None = None) -> None:
        self.embedder = embedder
        self.cfg = cfg or RetrievalConfig()
        self.docs: list[Doc] = []
        self._ids: list[str] = []
        self._emb: np.ndarray | None = None       # (n, dim) doc embeddings
        self._faiss = None
        self._bm25 = None
        self._reranker = None
        # --- cached "base" corpus for incremental poison insertion (attack phases) ---
        self._base_docs: list[Doc] = []
        self._base_ids: list[str] = []
        self._base_emb: np.ndarray | None = None
        self._base_tokens: list[list[str]] | None = None

    # ---- indexing -------------------------------------------------------
    def index(self, docs: list[Doc]) -> "RetrievalPipeline":
        self.docs = list(docs)
        self._ids = [d.doc_id for d in self.docs]
        texts = [d.text for d in self.docs]

        self._emb = self.embedder.encode(texts)
        self._build_dense(self._emb)
        if self.cfg.hybrid:
            self._build_bm25(texts)
        return self

    def _build_dense(self, emb: np.ndarray) -> None:
        try:
            import faiss  # optional

            index = faiss.IndexFlatIP(emb.shape[1])  # inner product; unit vecs => cosine
            index.add(emb)
            self._faiss = index
        except Exception:
            self._faiss = None  # numpy brute-force fallback in _dense_scores

    def _build_bm25(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([t.lower().split() for t in texts])

    # ---- scoring --------------------------------------------------------
    def _dense_scores(self, q_vec: np.ndarray) -> np.ndarray:
        assert self._emb is not None, "call index() first"
        return self._emb @ q_vec  # cosine for unit vectors

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(query.lower().split()), dtype=np.float32)

    # ---- search ---------------------------------------------------------
    def search(self, query_id: str, query_text: str) -> RetrievalResult:
        q_vec = self.embedder.encode_one(query_text)
        dense = self._dense_scores(q_vec)

        if not self.cfg.hybrid:
            order = np.argsort(-dense)[: self.cfg.top_k]
            return RetrievalResult(query_id, [self._ids[i] for i in order],
                                   [float(dense[i]) for i in order])

        # hybrid: fuse dense + bm25 into a candidate pool
        bm25 = self._bm25_scores(query_text)
        fused = (1 - self.cfg.bm25_weight) * _minmax(dense) + self.cfg.bm25_weight * _minmax(bm25)
        pool = np.argsort(-fused)[: self.cfg.candidate_k]

        if not self.cfg.rerank:
            top = pool[: self.cfg.top_k]
            return RetrievalResult(query_id, [self._ids[i] for i in top],
                                   [float(fused[i]) for i in top])

        # cross-encoder rerank over the candidate pool
        from sentence_transformers import CrossEncoder

        if not hasattr(self, "_reranker") or self._reranker is None:
            self._reranker = CrossEncoder(self.cfg.reranker_name, device=self.cfg.device)
        pairs = [(query_text, self.docs[i].text) for i in pool]
        ce = np.asarray(self._reranker.predict(pairs), dtype=np.float32)
        rerank_order = pool[np.argsort(-ce)][: self.cfg.top_k]
        ce_by_pool = {int(p): float(s) for p, s in zip(pool, ce)}
        return RetrievalResult(query_id, [self._ids[i] for i in rerank_order],
                               [ce_by_pool[int(i)] for i in rerank_order])

    # ---- incremental base + poison (attack phases) ----------------------
    # Embed the honest corpus ONCE, then evaluate many poison documents by only
    # embedding the single poison doc and stacking it on. Essential for large
    # (BEIR-scale) corpora, where re-indexing thousands of docs per trial is
    # prohibitive. Uses numpy for dense scores (exact; identical to FAISS IP).
    def index_base(self, docs: list[Doc]) -> "RetrievalPipeline":
        self._base_docs = list(docs)
        self._base_ids = [d.doc_id for d in self._base_docs]
        self._base_emb = self.embedder.encode([d.text for d in self._base_docs])
        self._base_tokens = ([d.text.lower().split() for d in self._base_docs]
                             if self.cfg.hybrid else None)
        return self

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.reranker_name, device=self.cfg.device)
        return self._reranker

    def _apply_defense(self, cand: np.ndarray, scores: np.ndarray, docs) -> np.ndarray:
        """Provenance-weighted penalty on candidate scores (Phase 3 defense).

        A doc of trust t loses provenance_penalty * (1 - t) * (score spread) from its
        rank score. Scaling by the candidate score spread makes one penalty constant
        behave sensibly whether scores are cosines, fused [0,1], or reranker logits.
        Returns adjusted scores aligned with `cand`. No-op if the defense is off.
        """
        if not self.cfg.provenance_defense or len(cand) == 0:
            return scores
        spread = float(scores.max() - scores.min()) or 1.0
        tw = self.cfg.trust_weights
        penalty = np.array(
            [self.cfg.provenance_penalty * (1.0 - tw.get(docs[i].provenance, 0.0)) * spread
             for i in cand], dtype=np.float32)
        return scores - penalty

    def _rank_view(self, query_id, query_text, emb, ids, docs, tokens) -> RetrievalResult:
        q_vec = self.embedder.encode_one(query_text)
        dense = emb @ q_vec

        if not self.cfg.hybrid:
            cand = np.arange(len(ids))
            scores = dense.astype(np.float32)
        else:
            from rank_bm25 import BM25Okapi

            bm = np.asarray(BM25Okapi(tokens).get_scores(query_text.lower().split()), dtype=np.float32)
            fused = (1 - self.cfg.bm25_weight) * _minmax(dense) + self.cfg.bm25_weight * _minmax(bm)
            cand = np.argsort(-fused)[: self.cfg.candidate_k]
            if not self.cfg.rerank:
                scores = fused[cand].astype(np.float32)
            else:
                pairs = [(query_text, docs[i].text) for i in cand]
                scores = np.asarray(self._get_reranker().predict(pairs), dtype=np.float32)

        adj = self._apply_defense(cand, scores, docs)
        order = np.argsort(-adj)[: self.cfg.top_k]
        sel = cand[order]
        return RetrievalResult(query_id, [ids[i] for i in sel], [float(adj[o]) for o in order])

    def search_base(self, query_id: str, query_text: str) -> RetrievalResult:
        """Honest retrieval over the base corpus only (no poison)."""
        return self._rank_view(query_id, query_text, self._base_emb,
                               self._base_ids, self._base_docs, self._base_tokens)

    def search_with_extra(self, query_id: str, query_text: str, extra: Doc) -> RetrievalResult:
        """Retrieval over base corpus + one extra (poison) doc, reusing cached base embeddings."""
        extra_emb = self.embedder.encode_one(extra.text)[None, :]
        emb = np.vstack([self._base_emb, extra_emb])
        ids = self._base_ids + [extra.doc_id]
        docs = self._base_docs + [extra]
        tokens = (self._base_tokens + [extra.text.lower().split()]) if self.cfg.hybrid else None
        return self._rank_view(query_id, query_text, emb, ids, docs, tokens)
