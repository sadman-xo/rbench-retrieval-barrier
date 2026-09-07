"""Central configuration for the retrieval testbed.

Everything that later phases (attack, defense, adaptive attacker) need to agree
on lives here, so all experiments report comparable numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass
class RetrievalConfig:
    # --- embedding model (CPU-runnable open-source defaults) ---
    # Primary embedder for the smoke test; transferability probes swap this out.
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternates used later for the transferability gap (Phase 5).
    transfer_embedders: tuple[str, ...] = (
        "BAAI/bge-small-en-v1.5",
        "thenlper/gte-small",
    )

    # --- retrieval pipeline ---
    top_k: int = 5                 # docs returned to the LLM; success = poisoned doc in top_k
    candidate_k: int = 50          # first-stage pool size before reranking (hybrid path)
    hybrid: bool = False           # False = dense-only; True = dense + BM25 fused, then reranked
    rerank: bool = True            # apply cross-encoder reranker on the hybrid path
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_weight: float = 0.5       # fusion weight for sparse scores (dense weight = 1 - this)

    # --- provenance-weighted defense (Phase 3) ---
    provenance_defense: bool = False       # penalize low-trust docs in the final ranking
    # trust in [0,1] per provenance class; 1 = fully trusted, 0 = untrusted
    trust_weights: dict = field(default_factory=lambda: {
        "internal": 1.0, "unknown": 0.5, "external": 0.0})
    # penalty strength beta: a zero-trust doc loses beta * (score spread) from its rank score,
    # scaled to each pipeline's own score range so one beta works across dense/hybrid/rerank
    provenance_penalty: float = 0.5

    # --- runtime ---
    device: str = "cpu"
    normalize_embeddings: bool = True   # cosine similarity via dot product on unit vectors
    seed: int = 20260907

    def describe(self) -> str:
        path = "hybrid(dense+bm25)" + ("+rerank" if self.rerank else "") if self.hybrid else "dense-only"
        return f"{self.embedder_name} | {path} | top_k={self.top_k}"


DEFAULT = RetrievalConfig()


def resolve_device(name: str) -> str:
    """Turn 'auto' into 'cuda' when a GPU is present, else 'cpu'. Pass-through otherwise.

    Lets the same code run on the CPU laptop and on a Colab/Lightning GPU with no edit:
    `--device auto` picks the GPU when there is one.
    """
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
