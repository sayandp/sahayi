"""Offline tests for Phase 2b: query rewrite plumbing, contextual index text, LLM fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app import query
from backend.app.llm import client
from backend.app.retrieval.search import index_text


def test_index_text_adds_scheme_context_only_when_enabled() -> None:
    chunk = {"scheme": "Widow Pension (Kerala)", "department": "LSGD, Kerala",
             "section": "eligibility", "text": "1. വരുമാനം ഒരു ലക്ഷം"}  # fmt: skip
    assert index_text(chunk, False) == "1. വരുമാനം ഒരു ലക്ഷം"
    assert (
        index_text(chunk, True)
        == "Widow Pension (Kerala) — LSGD, Kerala [eligibility]\n1. വരുമാനം ഒരു ലക്ഷം"
    )


def test_rewrite_variants_dedup_and_order() -> None:
    rw = query.Rewrite("pension kittumo?", "manglish", "പെൻഷൻ കിട്ടുമോ?", "Can I get a pension?")
    assert rw.variants() == ["pension kittumo?", "പെൻഷൻ കിട്ടുമോ?", "Can I get a pension?"]
    same = query.Rewrite("Can I get a pension?", "en", None, "Can I get a pension?")
    assert same.variants() == ["Can I get a pension?"]


def test_clean_rejects_non_malayalam_ml_query_and_null_hints() -> None:
    lang, ml, en, hints = query._clean(
        {"lang": "weird", "ml_query": "pension kittumo", "en_query": "Pension?",
         "profile_hints": {"age": 62, "district": None}}, "pension kittumo")  # fmt: skip
    assert lang == "en" and ml is None and en == "Pension?" and hints == {"age": 62}


def test_rewrite_falls_back_to_no_rewrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(query, "CACHE_DIR", tmp_path)

    def boom(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        raise client.LLMError("no provider")

    monkeypatch.setattr(query, "chat_json", boom)
    rw = query.rewrite("വിധവ പെൻഷൻ", use_cache=True)
    assert rw.provider == "none" and rw.lang == "ml" and rw.variants() == ["വിധവ പെൻഷൻ"]
    assert not list(tmp_path.iterdir())  # failures are not cached


def test_rewrite_cache_stores_no_profile_hints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(query, "CACHE_DIR", tmp_path)
    result = client.LLMResult("", "groq", "m", 900.0)
    result.data = {"lang": "manglish", "ml_query": "എനിക്ക് പെൻഷൻ", "en_query": "my pension",
                   "profile_hints": {"age": 70, "district": "Kollam"}}  # fmt: skip
    monkeypatch.setattr(query, "chat_json", lambda *a, **kw: result)
    first = query.rewrite("ente pension")
    assert first.profile_hints == {"age": 70, "district": "Kollam"} and not first.cached
    (cached_file,) = tmp_path.iterdir()
    assert "profile_hints" not in json.loads(cached_file.read_text(encoding="utf-8"))
    second = query.rewrite("ente pension")
    assert second.cached and second.ml_query == "എനിക്ക് പെൻഷൻ"


def test_chat_json_without_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.config import Settings

    monkeypatch.setattr(
        client, "get_settings", lambda: Settings(groq_api_key="", gemini_api_key="")
    )
    with pytest.raises(client.LLMError, match="no LLM provider"):
        client.chat_json([{"role": "user", "content": "hi"}])


class _FakeDense:
    def __init__(self, lists: dict[str, list[tuple[str, float]]]) -> None:
        self.lists = lists

    def search(self, vec, k: int = 20):  # noqa: ANN001, ANN201
        return self.lists[vec][:k]

    def close(self) -> None:
        pass


def _fake_retriever(monkeypatch: pytest.MonkeyPatch, lang: str):  # noqa: ANN202
    from backend.app.retrieval import search as srch

    chunks = {c: {"chunk_id": c, "text": f"text {c}", "scheme": "S", "department": "D",
                  "doc_id": c, "page": 1} for c in "abcdef"}  # fmt: skip
    monkeypatch.setattr(srch, "load_chunks", lambda strategy: chunks)
    monkeypatch.setattr(srch, "embed_queries", lambda qs, model: list(qs))
    calls: list[tuple[str, list[str]]] = []

    def fake_rerank(q, cands, spec, info=None):  # noqa: ANN001, ANN202
        calls.append((q, [c for c, _ in cands]))
        return sorted(((c, float(ord(c))) for c, _ in cands), key=lambda kv: -kv[1])

    monkeypatch.setattr(srch, "rerank", fake_rerank)
    rw = query.Rewrite("orig", lang, "മലയാളം", "english", provider="groq")
    cfg = srch.RetrievalConfig(mode="dense", rewrite=True, overlap_rewrite=True, rerank=True,
                               rerank_pool=3, k_final=5)  # fmt: skip
    r = srch.Retriever.__new__(srch.Retriever)
    r.config, r.chunks, r.bm25, r.last_trace = cfg, chunks, None, srch.SearchTrace()
    r.dense = _FakeDense({"orig": [("a", 0.9), ("b", 0.8), ("c", 0.7)],
                          "മലയാളം": [("d", 0.95), ("a", 0.6)],
                          "english": [("e", 0.9), ("b", 0.5)]})  # fmt: skip
    monkeypatch.setattr(r, "_rewrite", lambda q, trace: rw)
    return r, calls


def test_overlap_reuses_speculative_scores_for_malayalam(monkeypatch: pytest.MonkeyPatch) -> None:
    r, calls = _fake_retriever(monkeypatch, "ml")
    hits = r.search("orig")
    assert calls[0] == ("orig", ["a", "b", "c"])  # speculative, before the rewrite
    assert all(q == "orig" for q, _ in calls)
    assert set(calls[1][1]).isdisjoint({"a", "b", "c"}) or calls[1][1] == []  # only new ids
    assert r.last_trace.reranked and hits[0].chunk_id in "abcde"


def test_overlap_rescores_manglish_with_malayalam_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    r, calls = _fake_retriever(monkeypatch, "manglish")
    r.search("orig")
    assert calls[-1][0] == "മലയാളം" and len(calls[-1][1]) == 3  # full pool, new query
