"""Cross-Entropy Method (CEM) trigger optimizer — black-box.

The attacker searches for a short trigger phrase that, when prepended to the
payload, makes the poisoned document look highly relevant to a target. It needs
NO model internals: it only calls a user-supplied `score_fn` that maps candidate
trigger strings to scores (in the attack, that score is embedding similarity to
the target query/queries). This matches arXiv:2601.07072's threat model.

Algorithm (the "guess -> score -> keep winners -> refine" loop from the explainer):
  * maintain a per-position categorical distribution P over a fixed vocabulary
  * sample N candidate triggers from P
  * score them, take the top `elite_frac` as elites
  * move P toward the elites' token frequencies (with smoothing)
  * repeat; return the best trigger ever seen
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Vector = np.ndarray


@dataclass
class CEMConfig:
    # Defaults aligned to the authors' released config (USENIX'26 #1226) where cheap:
    # adv_length=10, elite_frac=0.2, alpha(=smoothing)=0.55. n_samples/n_iters are kept
    # smaller than their 5000/30 for tractability; raise them for a closer match.
    trigger_len: int = 10          # trigger length in tokens
    n_samples: int = 96            # candidates sampled per iteration
    n_iters: int = 40              # optimization rounds
    elite_frac: float = 0.2        # fraction of candidates kept as elites
    smoothing: float = 0.55        # alpha: weight on elite frequencies vs. previous P
    seed: int = 20260907


@dataclass
class CEMResult:
    best_trigger: str
    best_score: float
    history: list[float] = field(default_factory=list)   # best score per iteration


def optimize(vocab: list[str], score_fn, cfg: CEMConfig | None = None) -> CEMResult:
    """Optimize a trigger over `vocab`.

    score_fn: Callable[[list[str]], np.ndarray] — given a batch of trigger strings,
              return a 1-D array of scores (higher = better). The caller decides what
              "better" means (e.g. cosine similarity of trigger+payload to the query).
    """
    cfg = cfg or CEMConfig()
    rng = np.random.default_rng(cfg.seed)
    V = len(vocab)
    if V == 0:
        raise ValueError("empty vocabulary")

    P = np.full((cfg.trigger_len, V), 1.0 / V)         # uniform start
    n_elite = max(2, int(round(cfg.n_samples * cfg.elite_frac)))

    best_trigger, best_score = "", -np.inf
    history: list[float] = []

    for _ in range(cfg.n_iters):
        # --- sample candidates: (n_samples, trigger_len) token indices ---
        idx = np.empty((cfg.n_samples, cfg.trigger_len), dtype=np.int64)
        for pos in range(cfg.trigger_len):
            idx[:, pos] = rng.choice(V, size=cfg.n_samples, p=P[pos])
        triggers = [" ".join(vocab[i] for i in row) for row in idx]

        # --- score ---
        scores = np.asarray(score_fn(triggers), dtype=np.float64)

        # --- track global best ---
        top = int(np.argmax(scores))
        if scores[top] > best_score:
            best_score, best_trigger = float(scores[top]), triggers[top]
        history.append(float(scores.max()))

        # --- update distribution toward elites ---
        elite = idx[np.argsort(-scores)[:n_elite]]
        newP = np.empty_like(P)
        for pos in range(cfg.trigger_len):
            counts = np.bincount(elite[:, pos], minlength=V).astype(np.float64)
            counts /= counts.sum()
            row = cfg.smoothing * counts + (1.0 - cfg.smoothing) * P[pos]
            newP[pos] = row / row.sum()
        P = newP

    return CEMResult(best_trigger=best_trigger, best_score=best_score, history=history)


def optimize_tokens(tokenizer, score_fn, cfg: CEMConfig | None = None,
                    device: str = "cpu") -> CEMResult:
    """Token-level CEM — faithful to the authors' method (arXiv:2601.07072 / USENIX'26 #1226).

    Instead of a curated word vocabulary, the trigger is a sequence of raw token IDs
    from the embedding model's own tokenizer, decoded to text before scoring. This is
    the paper's actual search space (any token string, not just readable words).
    Independently implemented; no code copied from the authors' artifact.

    tokenizer: the embedder's HF tokenizer (has vocab_size + batch_decode).
    score_fn : Callable[[list[str]], np.ndarray] — trigger strings -> scores (higher better).
    """
    import torch

    cfg = cfg or CEMConfig()
    V = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    probs = torch.full((cfg.trigger_len, V), 1.0 / V, device=device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    n_elite = max(1, min(int(np.ceil(cfg.elite_frac * cfg.n_samples)), cfg.n_samples - 1))

    best_trigger, best_score = "", -np.inf
    history: list[float] = []

    for _ in range(cfg.n_iters):
        # sample (n_samples, trigger_len) token ids, one categorical per position
        samples = torch.stack(
            [torch.multinomial(probs[pos], cfg.n_samples, replacement=True, generator=gen)
             for pos in range(cfg.trigger_len)], dim=1)
        triggers = [t.strip() for t in
                    tokenizer.batch_decode(samples.tolist(), skip_special_tokens=True)]

        scores = np.asarray(score_fn(triggers), dtype=np.float64)
        top = int(np.argmax(scores))
        if scores[top] > best_score:
            best_score, best_trigger = float(scores[top]), triggers[top]
        history.append(float(scores.max()))

        elite_idx = np.argpartition(scores, -n_elite)[-n_elite:]
        elite = samples[torch.as_tensor(elite_idx, device=device)]        # (n_elite, trigger_len)
        counts = torch.zeros((cfg.trigger_len, V), device=device)
        counts.scatter_add_(1, elite.T, torch.ones_like(elite.T, dtype=counts.dtype))
        new_probs = counts / counts.sum(dim=1, keepdim=True)
        probs = (1.0 - cfg.smoothing) * probs + cfg.smoothing * new_probs

        if torch.all(probs.max(dim=1).values > 0.995):   # distribution collapsed
            break

    return CEMResult(best_trigger=best_trigger, best_score=best_score, history=history)
