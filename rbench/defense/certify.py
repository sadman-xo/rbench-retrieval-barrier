"""Attack-independent certificate for the provenance defense (dense-only path).

The Phase 4 result (adaptive attacker = 0% RSR) is empirical: it says the attackers
we built failed. A certificate says NO poison text can succeed, whatever the
optimizer. It follows from the defense's own arithmetic, for one query q:

  * the poison p is untrusted (trust 0), so its adjusted score is
        a_p = s_p - beta * spread,      spread = max(all scores) - min(all scores)
  * let m = the lowest honest score. If s_p >= m, then spread >= s_p - m, so
        a_p <= (1 - beta) * s_p + beta * m
    and if s_p < m, then a_p <= s_p < m. Trusted docs are never penalized.
  * s_p = e_p . q <= |e_p| |q| <= B, where B = (largest poison norm allowed) * |q|:
    |q| for unit-norm (cosine) embeddings, cap * |q| under the untrusted-norm cap,
    and INFINITY for a raw dot-product retriever.

So  a_p <= bound = m + max(0, 1 - beta) * max(B - m, 0),  and if at least k TRUSTED
docs score strictly above `bound`, the poison cannot enter the top k. Two corollaries:

  * beta >= 1: bound = m, so the poison always loses -- but every honest external doc
    loses too (hard exclusion). This is the only secure setting when B = infinity.
  * beta < 1 with B = infinity: bound = infinity, nothing is certified. The soft
    defense is secure ONLY because similarity is bounded (cosine, or the norm cap).

The certificate is conservative: it ignores honest external docs, which can only
push the poison further down.
"""
from __future__ import annotations

import numpy as np


def poison_score_bound(q_vec: np.ndarray, max_poison_norm: float) -> float:
    """B: the largest raw score any poison embedding can reach for this query."""
    if not np.isfinite(max_poison_norm):
        return float("inf")
    return float(max_poison_norm * np.linalg.norm(q_vec))


def certify_query(base_scores: np.ndarray, trusted: np.ndarray, q_vec: np.ndarray,
                  k: int, beta: float, max_poison_norm: float) -> tuple[bool, float]:
    """Return (certified, margin) for one query.

    base_scores     : (n,) raw honest scores e_d . q, AFTER any norm cap
    trusted         : (n,) bool, True for docs with trust 1 (never penalized)
    max_poison_norm : 1.0 for unit embeddings, the cap for capped dot product,
                      float('inf') for an uncapped dot-product retriever
    margin          : (k-th best trusted score) - bound. Certified iff margin > 0.
    """
    trusted_scores = base_scores[trusted]
    if len(trusted_scores) < k:
        return False, float("-inf")
    tau_k = float(np.partition(trusted_scores, -k)[-k])
    m = float(base_scores.min())
    b = poison_score_bound(q_vec, max_poison_norm)
    if beta >= 1.0:
        bound = m
    elif not np.isfinite(b):
        bound = float("inf")
    else:
        bound = (1.0 - beta) * max(b, m) + beta * m
    margin = tau_k - bound
    return margin > 0, margin
