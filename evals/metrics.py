"""Retrieval metrics over (doc_id, page) gold sets: hit@k, MRR@k, recall@k."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Page = tuple[str, int]


def first_relevant_rank(retrieved: Sequence[Page], gold: Iterable[Page]) -> int | None:
    """1-based rank of the first retrieved page that is in gold, else None."""
    gold_set = set(gold)
    for rank, page in enumerate(retrieved, start=1):
        if page in gold_set:
            return rank
    return None


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def recall_at_k(retrieved: Sequence[Page], gold: Iterable[Page], k: int) -> float:
    gold_set = set(gold)
    return len(gold_set & set(retrieved[:k])) / len(gold_set) if gold_set else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]
