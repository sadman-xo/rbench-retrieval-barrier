"""Greedy Coordinate Search (GCS) trigger optimizer -- black-box.

An alternative to CEM that optimizes one token position at a time.  At each
position it tries a random subset of the vocabulary and keeps the replacement
that gives the highest score.  More query-hungry than CEM but often finds
better triggers because it searches each position exhaustively within its
candidate set.

The comparison with CEM tells us whether the defense is attack-agnostic:
if a different, stronger optimizer also fails against the provenance defense,
the defense's robustness is not an artifact of CEM's search limitations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cem import CEMResult


@dataclass
class GCSConfig:
    trigger_len: int = 12
    n_passes: int = 5
    candidates_per_pos: int = 500
    seed: int = 20260907


class QueryCounter:
    """Wraps a score_fn to count the total number of candidates scored."""

    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, cands: list[str]) -> np.ndarray:
        self.count += len(cands)
        return self.fn(cands)


def optimize_gcs(tokenizer, score_fn, cfg: GCSConfig | None = None,
                 device: str = "cpu") -> CEMResult:
    """Token-level greedy coordinate search.

    Same interface as optimize_tokens (CEM): takes a tokenizer and a score_fn,
    returns a CEMResult so the callers can swap optimizers without changes.

    Algorithm:
      1. Start from a random token sequence.
      2. For each position (in random order): try candidates_per_pos random
         tokens, keep the one that gives the highest score.
      3. Repeat for n_passes.  Stop early if no position improves in a pass.
    """
    cfg = cfg or GCSConfig()
    V = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    rng = np.random.default_rng(cfg.seed)

    current = rng.integers(0, V, size=cfg.trigger_len)
    current_text = tokenizer.decode(current.tolist(), skip_special_tokens=True).strip()
    current_score = float(score_fn([current_text])[0])

    best_trigger = current_text
    best_score = current_score
    history: list[float] = [best_score]

    for _ in range(cfg.n_passes):
        improved = False
        positions = rng.permutation(cfg.trigger_len)

        for pos in positions:
            n_cand = min(cfg.candidates_per_pos, V)
            cand_tokens = rng.choice(V, size=n_cand, replace=False)

            batch = np.tile(current, (n_cand, 1))
            batch[:, pos] = cand_tokens
            batch_texts = [t.strip() for t in
                           tokenizer.batch_decode(batch.tolist(),
                                                  skip_special_tokens=True)]

            scores = np.asarray(score_fn(batch_texts), dtype=np.float64)
            top = int(np.argmax(scores))

            if scores[top] > current_score:
                current[pos] = cand_tokens[top]
                current_score = float(scores[top])
                current_text = batch_texts[top]
                improved = True

                if current_score > best_score:
                    best_score = current_score
                    best_trigger = current_text

        history.append(best_score)
        if not improved:
            break

    return CEMResult(best_trigger=best_trigger, best_score=best_score,
                     history=history)
