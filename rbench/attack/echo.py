"""Query-echo + CEM suffix attacker (Phase 4c follow-up).

Phase 4c showed that pasting the query text into the poison (the naive "echo")
beats token-level CEM on Contriever and TAS-B: the echo starts at cosine ~0.9 with
the query, while CEM triggers from a uniform start reach ~0.5. This attacker keeps
the echo as a FIXED prefix and lets CEM choose only the suffix tokens, so the search
starts from the strongest point we know. Under a dot product the suffix can also add
norm, so this is the attacker that tests the norm hypothesis properly.

The attacker keeps the plain echo when no suffix beats it on the attacker's own
objective, so it is never weaker than naive on that objective.
"""
from __future__ import annotations

import numpy as np

from .cem import CEMConfig, optimize_tokens


def optimize_echo_suffix(tokenizer, score_fn, prefix: str, cfg: CEMConfig | None = None,
                         device: str = "cpu") -> tuple[str, bool]:
    """Return (trigger, suffix_won).

    score_fn : the same trigger-string -> score callable the other attackers use
               (static or adaptive); the prefix is prepended to every candidate here.
    trigger  : f"{prefix} {best suffix}" if the suffix helps, else the bare prefix.
    """
    res = optimize_tokens(tokenizer, lambda cands: score_fn([f"{prefix} {c}" for c in cands]),
                          cfg, device=device)
    echo_score = float(np.asarray(score_fn([prefix]))[0])
    if res.best_score > echo_score:
        return f"{prefix} {res.best_trigger}", True
    return prefix, False
