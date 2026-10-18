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
        self._bm25 = None
        self._bm25_tokens: list[list[str]] | None = None
        self._reranker = None
        # --- cached "base" corpus for incremental poison insertion (attack phases) ---
        self._base_docs: list[Doc] = []
        self._base_ids: list[str] = []
        self._base_emb: np.ndarray | None = None       # after the norm cap (if any)
        self._base_emb_raw: np.ndarray | None = None   # as the embedder returned them
        self._base_tokens: list[list[str]] | None = None
        self._base_bm25 = None                          # BM25 over the base corpus (hybrid only)
        self._norm_cap: float | None = None            # fitted untrusted-norm cap (Phase 4c)

    # ---- indexing -------------------------------------------------------
    def index(self, docs: list[Doc]) -> "RetrievalPipeline":
        self.docs = list(docs)
        self._ids = [d.doc_id for d in self.docs]
        texts = [d.text for d in self.docs]

        self._emb = self.embedder.encode(texts)
        if self.cfg.hybrid:
            self._build_bm25(texts)
        return self

    def _build_bm25(self, texts: list[str]) -> None:
        self._bm25_tokens = [t.lower().split() for t in texts]
        self._bm25 = self._bm25_from(self._bm25_tokens)

    @staticmethod
    def _bm25_from(tokens: list[list[str]] | None):
        """Build a BM25 index from pre-tokenized docs, once. None -> None (dense-only)."""
        if tokens is None:
            return None
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokens)

    # ---- search ---------------------------------------------------------
    def search(self, query_id: str, query_text: str,
               q_vec: np.ndarray | None = None) -> RetrievalResult:
        """Retrieval over the index() corpus, through the shared ranking core.

        Same ranking path as the attack phases (search_base / _rank_view). With the
        default config (no provenance defense, no occupancy cap) this is plain
        dense-only, or hybrid+rerank, retrieval. Pass a pre-encoded q_vec to skip
        re-encoding the query.
        """
        assert self._emb is not None, "call index() first"
        return self._rank_view(query_id, query_text, self._emb, self._ids, self.docs,
                               self._bm25_tokens, q_vec=q_vec, bm25=self._bm25)

    # ---- incremental base + poison (attack phases) ----------------------
    # Embed the honest corpus ONCE, then evaluate many poison documents by only
    # embedding the single poison doc and stacking it on. Essential for large
    # (BEIR-scale) corpora, where re-indexing thousands of docs per trial is
    # prohibitive. Uses numpy for dense scores (exact inner product).
    def index_base(self, docs: list[Doc]) -> "RetrievalPipeline":
        self._base_docs = list(docs)
        self._base_ids = [d.doc_id for d in self._base_docs]
        self._base_emb_raw = self.embedder.encode([d.text for d in self._base_docs])
        self._base_tokens = ([d.text.lower().split() for d in self._base_docs]
                             if self.cfg.hybrid else None)
        self._base_bm25 = self._bm25_from(self._base_tokens)
        self._fit_norm_cap()
        return self

    def with_config(self, cfg: RetrievalConfig) -> "RetrievalPipeline":
        """The same indexed base corpus under a different config, with no re-embedding.

        Lets one run evaluate a whole (beta, norm cap) grid for the cost of one index.
        """
        other = RetrievalPipeline(self.embedder, cfg)
        other._base_docs, other._base_ids = self._base_docs, self._base_ids
        other._base_emb_raw = self._base_emb_raw
        other._base_tokens = self._base_tokens
        if cfg.hybrid and other._base_tokens is None:
            other._base_tokens = [d.text.lower().split() for d in other._base_docs]
        # same base corpus, so reuse the prebuilt BM25; only build when this view
        # turns hybrid on over a base that was indexed dense-only.
        other._base_bm25 = (other._bm25_from(other._base_tokens)
                            if cfg.hybrid and self._base_bm25 is None else self._base_bm25)
        other._reranker = self._reranker
        other._fit_norm_cap()
        return other

    # ---- untrusted-norm cap (Phase 4c) ----------------------------------
    # Dot-product similarity is |e_d| |q| cos(theta): an attacker who cannot turn the
    # angle further can still win by growing |e_d|. Clipping untrusted norms to what
    # trusted docs look like bounds that, and leaves every trusted doc's score as is.
    def _trust(self, doc: Doc) -> float:
        return self.cfg.trust_weights.get(doc.provenance, 0.0)

    def _fit_norm_cap(self) -> None:
        """Fit the cap on trusted-doc norms, then build the capped base embeddings."""
        self._norm_cap = None
        if self.cfg.norm_cap_pct is not None and len(self._base_docs):
            norms = np.linalg.norm(self._base_emb_raw, axis=1)
            trusted = np.array([self._trust(d) >= 1.0 for d in self._base_docs])
            ref = norms[trusted] if trusted.any() else norms
            self._norm_cap = float(np.percentile(ref, self.cfg.norm_cap_pct))
        self._base_emb = self._cap_norms(self._base_emb_raw, self._base_docs)

    def _cap_norms(self, emb: np.ndarray, docs) -> np.ndarray:
        """Clip the embedding norm of every untrusted doc (trust < 1) to the cap."""
        if self._norm_cap is None:
            return emb
        norms = np.linalg.norm(emb, axis=1)
        untrusted = np.array([self._trust(d) < 1.0 for d in docs])
        scale = np.where(untrusted & (norms > self._norm_cap),
                         self._norm_cap / np.maximum(norms, 1e-12), 1.0)
        return (emb * scale[:, None]).astype(np.float32)

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

    def _rank_view(self, query_id, query_text, emb, ids, docs, tokens,
                   q_vec=None, bm25=None) -> RetrievalResult:
        q_vec = self.embedder.encode_one(query_text) if q_vec is None else q_vec
        dense = emb @ q_vec

        if not self.cfg.hybrid:
            cand = np.arange(len(ids))
            scores = dense.astype(np.float32)
        else:
            if bm25 is None:
                bm25 = self._bm25_from(tokens)
            bm = np.asarray(bm25.get_scores(query_text.lower().split()), dtype=np.float32)
            fused = (1 - self.cfg.bm25_weight) * _minmax(dense) + self.cfg.bm25_weight * _minmax(bm)
            cand = np.argsort(-fused)[: self.cfg.candidate_k]
            if not self.cfg.rerank:
                scores = fused[cand].astype(np.float32)
            else:
                pairs = [(query_text, docs[i].text) for i in cand]
                scores = np.asarray(self._get_reranker().predict(pairs), dtype=np.float32)

        adj = self._apply_defense(cand, scores, docs)
        order = self._fill_top_k(np.argsort(-adj), cand, docs)
        sel = cand[order]
        return RetrievalResult(query_id, [ids[i] for i in sel], [float(adj[o]) for o in order])

    # ---- bounded occupancy (Phase 4d) -----------------------------------
    # The soft penalty has no middle ground on SciFact (Phase 4c): either the poison
    # gets in, or no external evidence does. A slot quota bounds the damage instead:
    # at most j untrusted docs in the top k, whatever their scores. Trusted scores do
    # not depend on the poison, so the trusted top-(k - j) is always kept
    # (rbench/defense/certify.py: certified_trusted_ids).
    def _fill_top_k(self, order: np.ndarray, cand: np.ndarray, docs) -> np.ndarray:
        """Best-first top_k over `order`, with at most cfg.max_untrusted untrusted docs."""
        j = self.cfg.max_untrusted
        if j is None:
            return order[: self.cfg.top_k]
        keep, n_untrusted = [], 0
        for o in order:
            if self._trust(docs[cand[o]]) < 1.0:
                if n_untrusted >= j:
                    continue
                n_untrusted += 1
            keep.append(o)
            if len(keep) == self.cfg.top_k:
                break
        return np.asarray(keep, dtype=int)

    def search_base(self, query_id: str, query_text: str,
                    q_vec: np.ndarray | None = None) -> RetrievalResult:
        """Honest retrieval over the base corpus only (no poison)."""
        return self._rank_view(query_id, query_text, self._base_emb,
                               self._base_ids, self._base_docs, self._base_tokens,
                               q_vec=q_vec, bm25=self._base_bm25)

    def search_with_extra(self, query_id: str, query_text: str, extra: Doc,
                          q_vec: np.ndarray | None = None) -> RetrievalResult:
        """Retrieval over base corpus + one extra (poison) doc, reusing cached base embeddings."""
        return self.search_with_extras(query_id, query_text, [extra], q_vec=q_vec)

    def search_with_extras(self, query_id: str, query_text: str,
                           extras: list[Doc],
                           q_vec: np.ndarray | None = None) -> RetrievalResult:
        """Retrieval over base corpus + several extra (poison) docs (Phase 4d saturation)."""
        if not extras:
            return self.search_base(query_id, query_text, q_vec=q_vec)
        extra_emb = self._cap_norms(self.embedder.encode([d.text for d in extras]), extras)
        emb = np.vstack([self._base_emb, extra_emb])
        ids = self._base_ids + [d.doc_id for d in extras]
        docs = self._base_docs + list(extras)
        # hybrid adds the poison tokens, so the prebuilt base BM25 cannot be reused;
        # _rank_view rebuilds from these tokens. dense-only ignores BM25 (tokens=None).
        tokens = ((self._base_tokens + [d.text.lower().split() for d in extras])
                  if self.cfg.hybrid else None)
        return self._rank_view(query_id, query_text, emb, ids, docs, tokens, q_vec=q_vec)
