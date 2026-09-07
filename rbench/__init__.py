import os as _os

# This machine has TensorFlow + Keras 3 installed globally. `transformers` will try to
# import its TF backend and crash on Keras 3 (tf_keras missing). We only use the PyTorch
# path, so disable the TF/Flax backends before anything imports transformers. Must run
# before the first `import sentence_transformers` / `import transformers`.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""rbench — Retrieval-Barrier benchmark.

A small, CPU-portable testbed for studying indirect prompt injection at the
retrieval stage (arXiv:2601.07072), extended toward provenance-weighted defense
and a defense-aware adaptive attacker.

Phase 1 (this stage): the RAG testbed — corpus, embedder, and a retrieval
pipeline that can run dense-only or the full hybrid (dense + BM25) + reranker path.
"""

__version__ = "0.1.0"
