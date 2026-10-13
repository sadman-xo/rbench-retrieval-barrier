"""Phase 4d checks with fake embedders -- bounded occupancy (BO-j), no downloads.

  * j = None leaves the ranking exactly as before; j >= k is the same as off
  * search_with_extra is search_with_extras with one doc
  * G1: at most j untrusted docs (so at most min(m, j) poisons) in the top k
  * G2: the trusted top-(k - j) is always kept, for any poisons (certified_trusted_ids)
  * "at most", not "exactly": weak external docs get no reserved slots
  * j = 0 is hard exclusion
"""
from __future__ import annotations

import numpy as np
import pytest

from rbench.config import RetrievalConfig
from rbench.defense.certify import certified_trusted_ids
from rbench.retriever import Doc

from test_norm_defense import K, make_docs, pipe_for


def scenario(rng, normalize: bool, n: int = 40, dim: int = 6):
    emb = rng.standard_normal((n, dim)) * rng.lognormal(0.0, 0.4, (n, 1))
    q = rng.standard_normal(dim)
    emb[: rng.integers(3, 12)] += q * rng.uniform(0.5, 2.0)    # some on-topic docs
    if normalize:
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        q /= np.linalg.norm(q)
    internal = rng.random(n) < 0.6
    internal[:K] = True                                         # enough trusted docs
    docs = make_docs(n, internal)
    table = {d.text: emb[i] for i, d in enumerate(docs)}
    table["q"] = q
    return q, internal, docs, table


def make_poisons(rng, table, q, m: int, normalize: bool) -> list[Doc]:
    """m distinct poisons near the query direction -- strong enough to want every slot."""
    qdir = q / np.linalg.norm(q)
    out = []
    for i in range(m):
        v = qdir + rng.standard_normal(len(q)) * 0.05
        v = v / np.linalg.norm(v) if normalize else v * rng.uniform(1.0, 1e4)
        table[f"p{i}"] = v.astype(np.float32)
        out.append(Doc(doc_id=f"p{i}", text=f"p{i}", provenance="external", is_poison=True))
    return out


# ── j = None is the old behaviour ────────────────────────────────────────
def test_off_matches_plain_ranking_and_j_at_least_k_is_off():
    rng = np.random.default_rng(0)
    for beta in (0.0, 0.5):
        q, internal, docs, table = scenario(rng, normalize=False)
        poisons = make_poisons(rng, table, q, 3, normalize=False)
        kw = dict(provenance_defense=beta > 0, provenance_penalty=beta)
        off = pipe_for(table, docs, **kw)
        wide = off.with_config(RetrievalConfig(top_k=K, max_untrusted=K, **kw))
        for p in (off, wide):
            assert p.search_with_extras("q", "q", poisons).ranked_ids == \
                off.search_with_extras("q", "q", poisons).ranked_ids
        if beta == 0:                                  # no defense: plain top-k by e_d . q
            expect = [docs[i].doc_id for i in np.argsort(-(off._base_emb @ q))[:K]]
            assert off.search_base("q", "q").ranked_ids == expect


def test_single_extra_is_extras_of_one():
    rng = np.random.default_rng(1)
    q, internal, docs, table = scenario(rng, normalize=False)
    (p,) = make_poisons(rng, table, q, 1, normalize=False)
    for j in (None, 0, 1, 2):
        pipe = pipe_for(table, docs, provenance_defense=True, provenance_penalty=0.25,
                        max_untrusted=j)
        one = pipe.search_with_extra("q", "q", p)
        many = pipe.search_with_extras("q", "q", [p])
        assert one.ranked_ids == many.ranked_ids and one.scores == many.scores
    assert pipe.search_with_extras("q", "q", []).ranked_ids == \
        pipe.search_base("q", "q").ranked_ids


# ── G1 and G2 against many poisons, on the real pipeline code ────────────
@pytest.mark.parametrize("seed", range(4))
def test_occupancy_and_trusted_retention_hold(seed):
    rng = np.random.default_rng(seed)
    saturated = flooded_without_bo = 0
    for _ in range(60):
        normalize = bool(rng.random() < 0.3)
        q, internal, docs, table = scenario(rng, normalize)
        m = int(rng.choice([1, 2, 3, 5, 8]))
        j = int(rng.choice([0, 1, 2, 3]))
        beta = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
        cap = 99.0 if rng.random() < 0.5 else None
        poisons = make_poisons(rng, table, q, m, normalize)
        kw = dict(provenance_defense=beta > 0, provenance_penalty=beta, norm_cap_pct=cap)
        no_bo = pipe_for(table, docs, **kw)
        bo = no_bo.with_config(RetrievalConfig(top_k=K, max_untrusted=j, **kw))

        res = bo.search_with_extras("q", "q", poisons)
        by_id = {d.doc_id: d for d in docs + poisons}
        untrusted = [i for i in res.ranked_ids if by_id[i].provenance != "internal"]
        n_poison = sum(by_id[i].is_poison for i in res.ranked_ids)
        assert len(res.ranked_ids) == K
        assert len(untrusted) <= j                                         # G1
        assert n_poison <= min(m, j)
        kept = certified_trusted_ids(bo._base_emb @ q, internal, K, j)
        assert {docs[i].doc_id for i in kept} <= set(res.ranked_ids)       # G2

        saturated += j > 0 and n_poison == min(m, j)
        flooded_without_bo += sum(by_id[i].is_poison for i in
                                  no_bo.search_with_extras("q", "q", poisons).ranked_ids) > j
    # not vacuous: the poisons do fill the slots BO allows, and without BO they take more
    assert saturated >= 10 and flooded_without_bo >= 10


# ── "at most j", and j = 0 ───────────────────────────────────────────────
def test_weak_external_docs_get_no_reserved_slots():
    q = np.eye(4)[0]
    emb = np.vstack([q * s for s in (5.0, 4.0, 3.0, 2.0, 1.5, 1.2)]          # trusted, on topic
                    + [np.eye(4)[1] * 3.0, np.eye(4)[2] * 3.0])              # external, off topic
    internal = np.array([True] * 6 + [False] * 2)
    docs = make_docs(len(emb), internal)
    table = {d.text: emb[i] for i, d in enumerate(docs)}
    table["q"] = q
    res = pipe_for(table, docs, max_untrusted=2).search_base("q", "q")
    assert res.ranked_ids == ["d0", "d1", "d2", "d3", "d4"]


def test_j_zero_is_hard_exclusion():
    rng = np.random.default_rng(3)
    q, internal, docs, table = scenario(rng, normalize=False)
    poisons = make_poisons(rng, table, q, 5, normalize=False)
    pipe = pipe_for(table, docs, max_untrusted=0)
    trusted_ids = {d.doc_id for d in docs if d.provenance == "internal"}
    for res in (pipe.search_base("q", "q"), pipe.search_with_extras("q", "q", poisons)):
        assert set(res.ranked_ids) <= trusted_ids


def test_certified_trusted_ids_edges():
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.95])
    trusted = np.array([True, True, False, True, False])
    assert list(certified_trusted_ids(scores, trusted, k=3, j=1)) == [0, 3]
    assert len(certified_trusted_ids(scores, trusted, k=3, j=3)) == 0
    assert list(certified_trusted_ids(scores, trusted, k=5, j=0)) == [0, 3, 1]   # all trusted
