"""Phase 4 — defense-aware adaptive attacker score functions.

The static attacker (Phase 2) maximizes raw embedding similarity. It does not
know the defense exists.  The adaptive attacker optimizes the POST-DEFENSE score
directly: it knows β, knows its own trust class (external → trust 0), and
accounts for the provenance penalty in its objective.

Two modes
---------
per-query   One trigger per query.  Directly comparable to Phases 2 and 3.
T2          One trigger trained on a DISTRIBUTION of queries, evaluated on
            HELD-OUT queries the attacker has never seen.  The realistic threat
            model: the attacker does not know the exact query the user will ask.

Analytical insight (dense-only, per-query)
------------------------------------------
The provenance penalty is  β · (1 − trust) · spread.  For an external poison
(trust 0) whose raw score exceeds the honest maximum:

    post_defense = raw · (1 − β) + β · min_score

This is monotonically increasing in raw similarity.  The adaptive attacker's
ranking of candidates is the SAME as the static attacker's.  The defense is
robust not because the attacker is ignorant — it is robust because the
embedding space has a capacity limit.  We implement the full computation
anyway (a) to validate this analysis empirically and (b) because hybrid+rerank
scores are not pure cosine and the monotonicity argument may not hold.

Norm cap (Phase 4c)
-------------------
Against a dot-product retriever with the untrusted-norm cap, the defended pipeline
clips the poison's embedding norm before scoring.  Pass `norm_cap` so the adaptive
objective scores the CLIPPED embedding: extra norm is then worth nothing and the
attacker must spend its whole budget on the angle.  Here the adaptive objective is
genuinely different from the static one (which keeps rewarding norm).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..retriever.embedder import Embedder

ScoreFn = Callable[[list[str]], np.ndarray]


def clip_norms(embs: np.ndarray, norm_cap: float | None) -> np.ndarray:
    """Clip each row's L2 norm to `norm_cap` (the defended pipeline does the same
    to untrusted docs). No-op when norm_cap is None."""
    if norm_cap is None:
        return embs
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs * np.minimum(1.0, norm_cap / np.maximum(norms, 1e-12))


def make_perquery_score_fn(
    embedder: Embedder,
    q_vec: np.ndarray,
    base_scores: np.ndarray,
    beta: float,
    trust_external: float = 0.0,
    payload: str = "",
    norm_cap: float | None = None,
) -> ScoreFn:
    """CEM score function: post-defense score of the poison doc for one query.

    Parameters
    ----------
    embedder      : the embedding model (query access only)
    q_vec         : (dim,) pre-encoded query vector
    base_scores   : (n_docs,) cosine similarities of the base corpus to q_vec
    beta          : defense penalty strength
    trust_external: trust assigned to external docs (0.0 by default)
    payload       : the poison payload text appended after the trigger
    norm_cap      : the defense's untrusted-norm cap, if on (base_scores must then
                    come from the capped base embeddings)
    """
    b_max = float(base_scores.max())
    b_min = float(base_scores.min())
    coeff = beta * (1.0 - trust_external)

    def score_fn(cands: list[str]) -> np.ndarray:
        embs = clip_norms(embedder.encode([f"{t}. {payload}" for t in cands]), norm_cap)
        raw = embs @ q_vec
        spread = np.maximum(np.maximum(b_max, raw) - np.minimum(b_min, raw), 1e-12)
        return raw - coeff * spread

    return score_fn


def make_t2_score_fn(
    embedder: Embedder,
    q_vecs: np.ndarray,
    base_scores_list: list[np.ndarray],
    beta: float,
    trust_external: float = 0.0,
    payload: str = "",
    norm_cap: float | None = None,
) -> ScoreFn:
    """CEM score function: average post-defense score across multiple queries.

    Parameters
    ----------
    q_vecs           : (n_queries, dim) pre-encoded query vectors
    base_scores_list : list of (n_docs,) base-corpus scores, one per query
    """
    b_maxes = np.array([s.max() for s in base_scores_list], dtype=np.float32)
    b_mins = np.array([s.min() for s in base_scores_list], dtype=np.float32)
    coeff = beta * (1.0 - trust_external)
    n_q = len(base_scores_list)

    def score_fn(cands: list[str]) -> np.ndarray:
        embs = clip_norms(embedder.encode([f"{t}. {payload}" for t in cands]), norm_cap)
        raw = embs @ q_vecs.T                          # (n_cands, n_queries)
        spread = np.maximum(
            np.maximum(b_maxes[None, :], raw) - np.minimum(b_mins[None, :], raw),
            1e-12,
        )
        post = raw - coeff * spread
        return post.mean(axis=1)                        # average across queries

    return score_fn


def make_static_score_fn(
    embedder: Embedder,
    q_vec: np.ndarray,
    payload: str = "",
) -> ScoreFn:
    """Baseline CEM score function: raw similarity, defense-unaware."""
    def score_fn(cands: list[str]) -> np.ndarray:
        embs = embedder.encode([f"{t}. {payload}" for t in cands])
        return embs @ q_vec

    return score_fn


def make_static_t2_score_fn(
    embedder: Embedder,
    q_vecs: np.ndarray,
    payload: str = "",
) -> ScoreFn:
    """Baseline T2 score function: average raw similarity across queries."""
    def score_fn(cands: list[str]) -> np.ndarray:
        embs = embedder.encode([f"{t}. {payload}" for t in cands])
        return (embs @ q_vecs.T).mean(axis=1)

    return score_fn
