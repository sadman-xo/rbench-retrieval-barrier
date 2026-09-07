"""Provenance assignment for a realistic defense evaluation.

A trivial defense ("block everything external") only looks good because the
benchmark labels *all* honest docs 'internal' and only the poison 'external'. Real
RAG systems legitimately ingest external content (web pages, emails, uploaded
files). So we relabel a fraction of the HONEST corpus as 'external' — legitimate
untrusted-origin content — which the provenance defense cannot distinguish from the
poison by origin alone. That forces the real security/utility trade-off: penalizing
external docs demotes the poison AND some genuinely relevant external docs.
"""
from __future__ import annotations

import numpy as np

from ..retriever.corpus import Doc


def assign_provenance(docs: list[Doc], external_frac: float = 0.3,
                      seed: int = 20260907) -> list[Doc]:
    """Return a copy of `docs` with a random `external_frac` relabeled 'external',
    the rest 'internal'. Poison docs (is_poison) are never touched here — the attack
    injects those as 'external' itself."""
    rng = np.random.default_rng(seed)
    out: list[Doc] = []
    for d in docs:
        if d.is_poison:
            out.append(d)
            continue
        prov = "external" if rng.random() < external_frac else "internal"
        out.append(Doc(doc_id=d.doc_id, text=d.text, provenance=prov,
                       is_poison=False, meta=d.meta))
    return out
