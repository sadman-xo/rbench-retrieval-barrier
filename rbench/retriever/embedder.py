"""Thin wrapper over a sentence-transformers embedding model.

Kept deliberately minimal: the attack phase treats this as a black box that maps
text -> vector, which matches the paper's threat model (query-only access to an
embedding model, no gradients, no internals).
"""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        # Imported lazily so `import rbench` works before heavy deps are installed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def tokenizer(self):
        """The underlying HF tokenizer (for token-level CEM). Falls back across the
        SentenceTransformer API variants that expose it differently."""
        tok = getattr(self.model, "tokenizer", None)
        if tok is not None:
            return tok
        return self.model._first_module().tokenizer

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (n, dim) float32 array. Rows are unit vectors if normalize=True."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
