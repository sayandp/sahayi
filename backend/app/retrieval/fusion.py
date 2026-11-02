"""Reciprocal Rank Fusion (Cormack et al., 2009) with k=60."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
