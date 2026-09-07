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
