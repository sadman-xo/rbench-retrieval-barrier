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

Research code for indirect prompt injection at the retrieval stage
(arXiv:2601.07072): attack reproduction, provenance-aware defenses with
per-query certificates, and a defense-aware adaptive attacker.
"""

__version__ = "0.1.0"
