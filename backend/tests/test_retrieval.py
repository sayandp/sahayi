"""Offline tests for Phase 2 retrieval pieces (no models are loaded)."""

from __future__ import annotations

import pytest

from backend.app.retrieval.bm25 import BM25Index, TokenizerConfig, ml_stem, tokenize
from backend.app.retrieval.fusion import rrf
from evals.build_gold import norm, resolve
from evals.metrics import first_relevant_rank, hit_at_k, mrr_at_k, percentile, recall_at_k


def test_tokenize_normalises_legacy_chillu_and_case() -> None:
    legacy = "വിധവ പെന്‍ഷന്‍"  # legacy chillu spelling (consonant+virama+ZWJ)
    assert tokenize(legacy, TokenizerConfig(stem=False)) == ["വിധവ", "പെൻഷൻ"]
    assert tokenize("PM-KISAN Scheme", TokenizerConfig(stem=False)) == ["pm", "kisan", "scheme"]


def test_tokenize_drops_joiners_inside_tokens() -> None:
    with_zwnj = "ലാപ്‌ടോപ്പ്"
    assert tokenize(with_zwnj, TokenizerConfig(stem=False)) == ["ലാപ്ടോപ്പ്"]


def test_ml_stem_strips_case_endings() -> None:
    assert ml_stem("പെൻഷന്റെ") == "പെൻഷ"
    assert ml_stem("പദ്ധതിയുടെ") == "പദ്ധതി"
    assert ml_stem("വീട്") == "വീട്"  # too short to strip
    toks = tokenize("കർഷകരുടെ", TokenizerConfig(stem=True))
    assert toks[0] == "കർഷകരുടെ" and len(toks) == 2  # exact token kept + stem


def test_bm25_cross_spelling_match() -> None:
    index = BM25Index(
        ["a", "b", "c"],
        ["വിധവ പെൻഷൻ അർഹത 50 വയസ്സ്", "കർഷക പെൻഷൻ", "PM Kisan instalment Rs 2000"],
    )
    assert index.search("വിധവ പെന്‍ഷന്‍", k=2)[0][0] == "a"
    assert index.search("kisan instalment")[0][0] == "c"
    assert index.search("") == []


def test_rrf_orders_by_reciprocal_rank() -> None:
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["d"] == pytest.approx(1 / 63)


def test_metrics() -> None:
    retrieved = [("d1", 1), ("d2", 3), ("d3", 1)]
    assert first_relevant_rank(retrieved, [("d2", 3)]) == 2
    assert first_relevant_rank(retrieved, [("x", 1)]) is None
    ranks = [1, 2, None, 6]
    assert hit_at_k(ranks, 5) == 0.5
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 0.5 + 1 / 6) / 4)
    assert recall_at_k(retrieved, [("d1", 1), ("d9", 9)], 3) == 0.5
    assert percentile([5, 1, 3], 0.5) == 3


def test_gold_resolution_by_evidence() -> None:
    ml = norm("പ്രതിവർഷം 5 ലക്ഷം രൂപ ചികിത്സക്കായി")
    pages = [
        {"doc_id": "kasp--a", "page": 1, "scheme_id": "kasp", "norm": ml},
        {"doc_id": "kasp--a", "page": 2, "scheme_id": "kasp", "norm": norm("other")},
        {"doc_id": "x--b", "page": 1, "scheme_id": "x", "norm": ml},
    ]
    q = {"schemes": ["kasp"], "evidence": ["പ്രതിവർഷം 5  ലക്ഷം", "missing text"]}
    gold, missing = resolve(q, pages)
    assert gold == [{"doc_id": "kasp--a", "page": 1}]  # other scheme's page is ignored
    assert missing == ["missing text"]


def test_weighted_rrf() -> None:
    fused = dict(rrf([["a"], ["b"]], k=60, weights=[0.5, 1.0]))
    assert fused["b"] > fused["a"]
    assert fused["a"] == pytest.approx(0.5 / 61)


def test_rerank_uses_score_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.retrieval import rerank as rr

    calls = []

    class FakeBackend:
        def score(self, query, texts):  # noqa: ANN001, ANN201
            calls.append(len(texts))
            return [float(len(t)) for t in texts]

    monkeypatch.setattr(rr, "load_backend", lambda spec: FakeBackend())
    monkeypatch.setattr(rr, "score_cache", lambda tag: rr.ScoreCache(tag, tmp_path))
    spec = rr.RerankSpec(model="m/x")
    cands = [("a", "short"), ("b", "a much longer text")]
    assert [c for c, _ in rr.rerank("q", cands, spec)] == ["b", "a"]
    assert [c for c, _ in rr.rerank("q", cands, spec)] == ["b", "a"]
    assert calls == [2]  # second call fully served from the on-disk cache
    rr.rerank("q", [*cands, ("c", "new")], spec)
    assert calls == [2, 1]
    # a different max_length is a different cache key space
    rr.rerank("q", cands, rr.RerankSpec(model="m/x", max_length=128))
    assert calls == [2, 1, 2]


def test_api_reranker_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.retrieval import rerank as rr

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(rr.RerankUnavailable, match="COHERE_API_KEY"):
        rr._ApiReranker(rr.RerankSpec(backend="api", provider="cohere"))


def test_abstain_threshold_maximises_f1() -> None:
    from evals.abstain import best_threshold

    # answerable scores high, no-answer low, one overlap
    scores = [
        (0.9, False),
        (0.8, False),
        (0.7, False),
        (0.3, False),
        (0.2, True),
        (0.1, True),
        (0.35, True),
    ]
    best = best_threshold(scores)
    assert 0.2 < best["threshold"] <= 0.35 or 0.35 < best["threshold"] < 0.7
    assert best["f1"] >= 0.8
