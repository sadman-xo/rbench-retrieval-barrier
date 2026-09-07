"""Retrieval metrics shared across all phases.

Honest-retrieval quality (Phase 1): recall@k, MRR over ground-truth relevant docs.
Attack metrics (Phase 2+) reuse RetrievalResult.rank_of / .contains directly:
Retrieval Success Rate = fraction of queries where the poison doc is in top_k.
"""
from __future__ import annotations

from ..retriever.pipeline import RetrievalResult


def recall_at_k(result: RetrievalResult, relevant_ids: tuple[str, ...]) -> float:
    if not relevant_ids:
        return float("nan")
    hits = sum(1 for r in relevant_ids if result.contains(r))
    return hits / len(relevant_ids)


def mrr(result: RetrievalResult, relevant_ids: tuple[str, ...]) -> float:
    """Reciprocal rank of the first relevant doc; 0 if none retrieved."""
    ranks = [result.rank_of(r) for r in relevant_ids]
    ranks = [r for r in ranks if r is not None]
    return 1.0 / min(ranks) if ranks else 0.0
