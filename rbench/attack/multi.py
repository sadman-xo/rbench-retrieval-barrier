"""m distinct echo + CEM poisons per query (Phase 4d saturation).

The source paper injects ONE item. An attacker who can plant one document through an
untrusted channel can usually plant several. Each poison here is the query echo plus
its own CEM suffix (seed + i), so the texts differ: identical copies would be trivial
to deduplicate at ingestion, and deduplication is not the defense under test.

The first poison is exactly the Phase 4c `echo+static` trigger (the bare echo if no
suffix beats it), so m = 1 reproduces the single-poison attacker. Later poisons always
carry their suffix, even when it scores a little below the bare echo.

A new seed alone does not give a new poison: when the objective has one clear optimum,
CEM converges to the same suffix from every seed. So poison i gets a suffix of
trigger_len + i tokens -- texts of different lengths cannot be equal -- and a repeat
(possible after decoding) is re-drawn with the next seed.
"""
from __future__ import annotations

from dataclasses import replace

from .cem import CEMConfig, optimize_tokens
from .echo import optimize_echo_suffix


def echo_poison_set(tokenizer, score_fn, prefix: str, m: int, cfg: CEMConfig | None = None,
                    device: str = "cpu") -> list[str]:
    """Return m distinct triggers, all starting with `prefix` (the query text).

    score_fn : trigger-string -> score, as for the other attackers (static or adaptive)
    """
    cfg = cfg or CEMConfig()
    first, _ = optimize_echo_suffix(tokenizer, score_fn, prefix, cfg, device=device)
    out = [first]
    seed = cfg.seed
    for _ in range(3 * m):
        if len(out) >= m:
            break
        seed += 1
        res = optimize_tokens(tokenizer, lambda cands: score_fn([f"{prefix} {c}" for c in cands]),
                              replace(cfg, seed=seed, trigger_len=cfg.trigger_len + len(out)),
                              device=device)
        trig = f"{prefix} {res.best_trigger}"
        if trig not in out:
            out.append(trig)
    if len(out) < m:
        raise RuntimeError(f"only {len(out)} distinct poisons for {prefix!r} (wanted {m})")
    return out
