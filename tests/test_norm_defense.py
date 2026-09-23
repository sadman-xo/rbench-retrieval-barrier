"""Phase 4c checks with fake embedders -- no model download, runs on CPU in seconds.

  * the untrusted-norm cap clips only untrusted docs, fitted on trusted norms
  * the hypothesis in miniature: a huge-norm poison beats the penalty alone under a
    dot product, and the cap stops it
  * certificate soundness: whenever certify_query says "certified", no poison vector
    (worst case included) reaches top-k in the REAL pipeline code
  * the echo + CEM suffix attacker is never worse than the bare echo on its objective
  * run_norm.main() runs end to end on the toy corpus
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np
import pytest

from rbench.attack.adaptive import clip_norms
from rbench.config import RetrievalConfig
from rbench.defense.certify import certify_query
from rbench.retriever import Doc, RetrievalPipeline

K = 5


class TableEmbedder:
    """text -> a fixed vector from a lookup table (the doc/query/poison text IS the key)."""

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self.table = table

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        return np.stack([self.table[t] for t in texts]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def make_docs(n: int, internal: np.ndarray) -> list[Doc]:
    return [Doc(doc_id=f"d{i}", text=f"d{i}",
                provenance="internal" if internal[i] else "external") for i in range(n)]


def pipe_for(table, docs, **cfg_kw) -> RetrievalPipeline:
    cfg = RetrievalConfig(top_k=K, **cfg_kw)
    return RetrievalPipeline(TableEmbedder(table), cfg).index_base(docs)


def poison_hits(pipe, table, vec) -> bool:
    table["p"] = vec.astype(np.float32)
    poison = Doc(doc_id="p", text="p", provenance="external", is_poison=True)
    return pipe.search_with_extra("q", "q", poison).contains("p")


# ── the cap ──────────────────────────────────────────────────────────────
def test_cap_clips_only_untrusted_docs():
    rng = np.random.default_rng(0)
    n, dim = 100, 8
    dirs = rng.standard_normal((n, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = rng.uniform(1.0, 2.0, n)
    internal = np.arange(n) % 2 == 0
    norms[1] = 5.0                       # an untrusted doc with a huge norm
    norms[3] = 1.1                       # an untrusted doc under the cap
    emb = dirs * norms[:, None]
    docs = make_docs(n, internal)
    table = {d.text: emb[i] for i, d in enumerate(docs)}
    table["q"] = dirs[0]

    pipe = pipe_for(table, docs, norm_cap_pct=99.0)
    cap = float(np.percentile(norms[internal], 99.0))
    assert pipe._norm_cap == pytest.approx(cap)
    got = np.linalg.norm(pipe._base_emb, axis=1)
    assert got[1] == pytest.approx(cap, rel=1e-5)                    # clipped
    assert got[3] == pytest.approx(1.1, rel=1e-5)                    # below cap: kept
    assert np.allclose(got[internal], norms[internal], rtol=1e-5)    # trusted: never touched

    off = pipe.with_config(RetrievalConfig(top_k=K))                 # same index, cap off
    assert off._norm_cap is None
    assert np.linalg.norm(off._base_emb[1]) == pytest.approx(5.0, rel=1e-5)


def test_adaptive_clip_matches_pipeline_clip():
    embs = np.array([[3.0, 4.0], [0.3, 0.4]])
    out = clip_norms(embs, 2.0)
    assert np.linalg.norm(out, axis=1) == pytest.approx([2.0, 0.5])
    assert clip_norms(embs, None) is embs


# ── the hypothesis in miniature ──────────────────────────────────────────
def test_norm_inflation_beats_penalty_alone_and_cap_stops_it():
    rng = np.random.default_rng(1)
    dim, n = 8, 60
    q = np.eye(dim)[0]
    emb = rng.standard_normal((n, dim)) * 0.5
    emb[:10] = q + rng.standard_normal((10, dim)) * 0.05   # 10 trusted, on-topic docs
    internal = np.zeros(n, dtype=bool)
    internal[:10] = True
    internal[30:45] = True
    docs = make_docs(n, internal)
    table = {d.text: emb[i] for i, d in enumerate(docs)}
    table["q"] = q
    big = q * 1e3                                           # norm inflation along q

    penalty_only = pipe_for(table, docs, provenance_defense=True, provenance_penalty=0.5)
    assert poison_hits(penalty_only, table, big)            # the soft defense breaks

    with_cap = penalty_only.with_config(RetrievalConfig(
        top_k=K, provenance_defense=True, provenance_penalty=0.5, norm_cap_pct=99.0))
    assert not poison_hits(with_cap, table, big)            # the cap restores it
    ok, margin = certify_query(with_cap._base_emb @ q, internal, q, K, 0.5,
                               with_cap._norm_cap * (1 + 1e-5))
    assert ok and margin > 0

    hard = penalty_only.with_config(RetrievalConfig(
        top_k=K, provenance_defense=True, provenance_penalty=1.0))
    assert not poison_hits(hard, table, big)                # beta = 1: hard exclusion holds


# ── certificate soundness against the real pipeline ──────────────────────
@pytest.mark.parametrize("seed", range(4))
def test_certificate_is_sound(seed):
    rng = np.random.default_rng(seed)
    certified_trials = 0
    for _ in range(60):
        n, dim = 40, 6
        normalize = bool(rng.random() < 0.3)
        emb = rng.standard_normal((n, dim)) * rng.lognormal(0.0, 0.4, (n, 1))
        q = rng.standard_normal(dim)
        # pull some docs toward q so certificates are sometimes non-trivial
        emb[: rng.integers(3, 12)] += q * rng.uniform(0.5, 2.0)
        if normalize:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            q /= np.linalg.norm(q)
        internal = rng.random(n) < 0.6
        internal[:K] = True
        docs = make_docs(n, internal)
        table = {d.text: emb[i] for i, d in enumerate(docs)}
        table["q"] = q
        beta = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]))
        cap_on = bool(rng.random() < 0.5)
        pipe = pipe_for(table, docs, provenance_defense=beta > 0, provenance_penalty=beta,
                        norm_cap_pct=99.0 if cap_on else None)

        lim = 1.0 if normalize else float("inf")
        if cap_on:
            lim = min(lim, pipe._norm_cap)
        ok, _ = certify_query(pipe._base_emb @ q, internal, q, K, beta, lim * (1 + 1e-5))
        if not ok:
            continue
        certified_trials += 1

        qdir = q / np.linalg.norm(q)
        worst = qdir * (1.0 if normalize else 1e4)          # along q, as large as allowed
        cands = [worst]
        for _ in range(20):
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            cands.append(v if normalize else v * rng.uniform(0.1, 1e4))
        for v in cands:
            assert not poison_hits(pipe, table, v), (beta, cap_on, normalize)
    assert certified_trials >= 5, "test is vacuous: almost nothing was certified"


def test_nothing_certified_for_soft_penalty_on_raw_dot_product():
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((30, 5))
    q = rng.standard_normal(5)
    internal = np.ones(30, dtype=bool)
    for beta in (0.0, 0.25, 0.5, 0.75):
        ok, margin = certify_query(emb @ q, internal, q, K, beta, float("inf"))
        assert not ok and margin == float("-inf")


# ── end to end: run_norm.main() on the toy corpus with a fake model ──────
class FakeTokenizer:
    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.vocab_size = len(words)

    def batch_decode(self, rows, skip_special_tokens: bool = True) -> list[str]:
        return [" ".join(self.words[i] for i in row) for row in rows]


class BagOfWordsEmbedder:
    """Sum of fixed per-word vectors / sqrt(#words). Unnormalized: repeating an
    on-topic word grows the norm -- a toy stand-in for norm inflation."""

    def __init__(self, model_name: str = "fake", device: str = "cpu",
                 normalize: bool = False, dim: int = 16) -> None:
        self.normalize, self.dim = normalize, dim
        self.tokenizer = FakeTokenizer(
            ["refund", "policy", "password", "reset", "shipping", "account", "invoice",
             "delivery", "days", "support", "email", "order", "the", "a", "how", "what"])

    def _word(self, w: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little")
        return np.random.default_rng(seed).standard_normal(self.dim)

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            ws = re.findall(r"[a-z0-9]+", t.lower()) or ["empty"]
            v = np.sum([self._word(w) for w in ws], axis=0) / np.sqrt(len(ws))
            out.append(v / np.linalg.norm(v) if self.normalize else v)
        return np.asarray(out, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ── the echo + CEM suffix attacker ───────────────────────────────────────
def test_echo_suffix_used_only_when_it_helps():
    pytest.importorskip("torch")
    from rbench.attack.cem import CEMConfig
    from rbench.attack.echo import optimize_echo_suffix

    tok = FakeTokenizer(["good", "bad", "filler"])
    cfg = CEMConfig(trigger_len=3, n_iters=5, n_samples=16, seed=0)

    def rewards_good(cands):                 # a suffix of "good" tokens helps
        return np.array([c.split().count("good") for c in cands], dtype=float)

    trig, won = optimize_echo_suffix(tok, rewards_good, "what is x", cfg)
    assert won and trig.startswith("what is x ") and "good" in trig

    def punishes_length(cands):              # every suffix hurts: keep the bare echo
        return np.array([-len(c.split()) for c in cands], dtype=float)

    trig, won = optimize_echo_suffix(tok, punishes_length, "what is x", cfg)
    assert not won and trig == "what is x"


@pytest.mark.parametrize("normalize", [False, True])
def test_run_norm_end_to_end_on_toy(monkeypatch, tmp_path, normalize):
    pytest.importorskip("torch")
    from rbench.attack import run_norm

    monkeypatch.setattr(run_norm, "Embedder", BagOfWordsEmbedder)
    monkeypatch.setattr(run_norm, "RESULTS_DIR", tmp_path)
    argv = ["run_norm", "--dataset", "toy", "--device", "cpu", "--iters", "3",
            "--samples", "8", "--trigger-len", "4", "--embedder", "fake/bow"]
    if normalize:
        argv.append("--normalize")
    monkeypatch.setattr(sys, "argv", argv)
    run_norm.main()

    out = json.loads((tmp_path / f"phase4c_norm_toy_bow_"
                      f"{'cosine' if normalize else 'dot'}.json").read_text())
    assert out["certificate_violations"] == []
    assert set(out["rsr"]) == {"naive", "static", "adaptive", "echo+static", "echo+adaptive",
                               "T2 static", "T2 adaptive"}
    assert set(out["echo_suffix_won"]) == {"echo+static", "echo+adaptive"}
    assert len(out["utility"]) == 10                   # 5 betas x cap off/on
    assert out["norms"]["cap"] > 0
