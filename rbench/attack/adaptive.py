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
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..retriever.embedder import Embedder

ScoreFn = Callable[[list[str]], np.ndarray]


def make_perquery_score_fn(
    embedder: Embedder,
    q_vec: np.ndarray,
    base_scores: np.ndarray,
    beta: float,
    trust_external: float = 0.0,
    payload: str = "",
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
    """
    b_max = float(base_scores.max())
    b_min = float(base_scores.min())
    coeff = beta * (1.0 - trust_external)

    def score_fn(cands: list[str]) -> np.ndarray:
        embs = embedder.encode([f"{t}. {payload}" for t in cands])
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
        embs = embedder.encode([f"{t}. {payload}" for t in cands])
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
