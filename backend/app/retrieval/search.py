"""Compose query rewrite / BM25 / dense / RRF / rerank into one retriever, by config."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from backend.app.query import Rewrite, query_kind
from backend.app.retrieval.bm25 import BM25Index, TokenizerConfig
from backend.app.retrieval.dense import DenseIndex, embed_queries, model_slug, onnx_query_encoder
from backend.app.retrieval.fusion import RRF_K, rrf
from backend.app.retrieval.rerank import RerankSpec, RerankUnavailable, rerank

ROOT = Path(__file__).resolve().parents[3]
# rewrite calls run here; shared so a stuck call never blocks a request's exit
_REWRITE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rewrite")
INDEX_DIR = ROOT / "data" / "index"
PROCESSED_DIR = ROOT / "data" / "processed"

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "default"
    strategy: str = "section_aware"
    mode: Mode = "hybrid"
    embed_model: str = "BAAI/bge-m3"
    embed_onnx_file: str | None = None  # query encoder under data/cache/onnx (deploy image)
    k_retrieve: int = 20  # per retriever and query variant, before fusion / rerank
    k_final: int = 10
    rrf_k: int = RRF_K
    bm25_weight: float = 1.0  # weight of BM25 lists in RRF (dense lists weigh 1.0)
    bm25_stem: bool = True
    # contextual chunk headers: index (and rerank) "scheme - department [section]" + text;
    # the displayed / cited text stays the raw chunk
    index_context: bool = False
    # query rewrite (Groq/Gemini): retrieve with original + Malayalam + English variants
    rewrite: bool = False
    rewrite_model: str = "qwen/qwen3.8-27b"
    rewrite_weight: float = 1.0  # RRF weight of rewritten variants (original query = 1.0)
    # overlap mode: give up on the rewrite this long after the search started and use the
    # original query alone (Groq 429 retries can otherwise hold a request for a minute)
    rewrite_deadline_s: float | None = None
    # query kinds (query.query_kind: "ml", "en") that get no rewrite call; retrieval then
    # uses the original query alone
    skip_rewrite_kinds: tuple[str, ...] = ()
    # run the rewrite call in a thread while the original query is retrieved and its pool
    # speculatively reranked; after the rewrite only new candidates are reranked
    overlap_rewrite: bool = False
    # which query the cross-encoder sees: "clearest" = the Malayalam-script rewrite for
    # Manglish questions, else the original; "original" = always the user's own words
    # (lets the speculative rerank in overlap mode count for Manglish too)
    rerank_with: str = "clearest"
    # reranking
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_backend: str = "st"  # st | onnx | api
    rerank_onnx_file: str | None = None
    rerank_max_length: int = 384
    rerank_pool: int = 20  # candidates sent to the cross-encoder
    rerank_provider: str | None = None  # api backend: cohere | jina
    # api backend: deadline per call, and the local spec used on error / timeout
    rerank_timeout_s: float | None = None
    rerank_fallback_onnx_file: str | None = None  # e.g. "hub:onnx/model_int8.onnx"
    # selective rerank: skip when the best dense list's top-1 minus top-2 cosine >= margin
    rerank_skip_margin: float | None = None
    # abstain when the top rerank score is below this (set by evals/abstain.py)
    abstain_threshold: float | None = None

    @property
    def bm25_path(self) -> Path:
        stem = "" if self.bm25_stem else ".nostem"
        ctx = ".ctx" if self.index_context else ""
        return INDEX_DIR / f"bm25.{self.strategy}{stem}{ctx}.pkl"

    @property
    def collection(self) -> str:
        ctx = "_ctx" if self.index_context else ""
        return f"chunks_{self.strategy}_{model_slug(self.embed_model)}{ctx}"

    @property
    def rerank_spec(self) -> RerankSpec:
        fallback = None
        if self.rerank_fallback_onnx_file:
            fallback = RerankSpec(self.rerank_model, "onnx", self.rerank_fallback_onnx_file,
                                  self.rerank_max_length)  # fmt: skip
        return RerankSpec(self.rerank_model, self.rerank_backend, self.rerank_onnx_file,
                          self.rerank_max_length, self.rerank_provider,
                          self.rerank_timeout_s, fallback)  # fmt: skip

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk_id: str
    score: float
    chunk: dict = field(repr=False)


@dataclass
class SearchTrace:
    """Per-query timings (ms) and decisions, for latency breakdowns and logs."""

    rewrite_ms: float = 0.0
    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    variants: list[str] = field(default_factory=list)
    rewrite_lang: str | None = None
    rewrite_provider: str | None = None
    rewrite_cached: bool = False
    reranked: bool = False
    rerank_skipped: bool = False
    dense_margin: float | None = None
    top_rerank_score: float | None = None
    spec_rerank_ms: float = 0.0  # speculative rerank, overlapped with the rewrite call
    rewrite_wait_ms: float = 0.0  # time spent blocked on the rewrite after local work
    rerank_backend_used: str | None = None  # api | onnx | st (onnx when api fell back)
    rerank_fallback: str | None = None  # why the hosted reranker was not used
    rewrite_tokens_in: int = 0
    rewrite_tokens_out: int = 0
    rewrite_model: str | None = None
    rewrite_timeout: bool = False  # the rewrite missed rewrite_deadline_s
    rewrite_skipped: bool = False  # skip_rewrite_kinds applied


def index_text(chunk: dict, context: bool) -> str:
    """Text used for BM25 / embeddings / reranking (not for display or citation)."""
    if not context:
        return chunk["text"]
    section = f" [{chunk['section']}]" if chunk.get("section") else ""
    return f"{chunk['scheme']} — {chunk['department']}{section}\n{chunk['text']}"


def load_chunks(strategy: str) -> dict[str, dict]:
    path = PROCESSED_DIR / f"chunks.{strategy}.jsonl"
    with path.open(encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in map(json.loads, f)}


class Retriever:
    """Loads the indexes for a config once; ``search`` is then cheap per query."""

    def __init__(self, config: RetrievalConfig, qdrant_path: Path | None = None) -> None:
        self.config = config
        self.chunks = load_chunks(config.strategy)
        self.bm25 = BM25Index.load(config.bm25_path) if config.mode != "dense" else None
        self.dense = (
            DenseIndex(qdrant_path or INDEX_DIR / "qdrant", config.collection)
            if config.mode != "bm25"
            else None
        )
        self.last_trace = SearchTrace()
        self.last_rewrite = None  # query.Rewrite of the last search (lang, profile hints)
        # overlap mode: called with the speculative ranking (original query only, reranked)
        # while the rewrite call is still in flight, so callers can start work early
        self.on_speculative = None

    def close(self) -> None:
        if self.dense:
            self.dense.close()

    def _rewrite(self, query: str, trace: SearchTrace):  # noqa: ANN202 - query.Rewrite
        from backend.app.query import rewrite

        rw = rewrite(query, model=self.config.rewrite_model)
        self._apply_rewrite(rw, trace)
        return rw

    @staticmethod
    def _apply_rewrite(rw, trace: SearchTrace) -> None:  # noqa: ANN001 - query.Rewrite
        trace.rewrite_ms = rw.latency_ms
        trace.rewrite_lang, trace.rewrite_provider, trace.rewrite_cached = (
            rw.lang, rw.provider, rw.cached
        )  # fmt: skip
        trace.rewrite_tokens_in, trace.rewrite_tokens_out = rw.tokens_in, rw.tokens_out
        trace.rewrite_model = rw.model

    def _lists(
        self, variants: list[str], trace: SearchTrace, first: int = 0
    ) -> tuple[list[list[str]], list[float], list[list[tuple[str, float]]]]:
        """Ranked id lists (BM25 and dense per variant) with their RRF weights.

        ``first`` is the index of variants[0] in the full variant list: index 0 is the
        user's own query (weight 1.0), later ones are rewrites (rewrite_weight).
        """
        c = self.config
        lists: list[list[str]] = []
        weights: list[float] = []
        dense_lists: list[list[tuple[str, float]]] = []
        vw = [1.0 if first + i == 0 else c.rewrite_weight for i in range(len(variants))]
        if self.bm25:
            t0 = time.perf_counter()
            for q, w in zip(variants, vw, strict=True):
                lists.append([cid for cid, _ in self.bm25.search(q, c.k_retrieve)])
                weights.append(c.bm25_weight * w)
            trace.bm25_ms += (time.perf_counter() - t0) * 1000
        if self.dense:
            t0 = time.perf_counter()
            if c.embed_onnx_file:
                vectors = onnx_query_encoder(c.embed_onnx_file).encode(variants)
            else:
                vectors = embed_queries(variants, c.embed_model)
            for vec in vectors:
                dense_lists.append(self.dense.search(vec, c.k_retrieve))
            trace.dense_ms += (time.perf_counter() - t0) * 1000
            for hits, w in zip(dense_lists, vw, strict=True):
                lists.append([cid for cid, _ in hits])
                weights.append(w)
        return lists, weights, dense_lists

    def _fuse(
        self,
        lists: list[list[str]],
        weights: list[float],
        dense_lists: list[list[tuple[str, float]]],
        trace: SearchTrace,
    ) -> list[tuple[str, float]]:
        if dense_lists:
            best = max(dense_lists, key=lambda h: h[0][1] if h else -1.0)
            trace.dense_margin = best[0][1] - best[1][1] if len(best) > 1 else None
        if len(lists) == 1:
            return dense_lists[0] if dense_lists else [(cid, 0.0) for cid in lists[0]]
        return rrf(lists, self.config.rrf_k, weights)

    def candidates(self, query: str, trace: SearchTrace | None = None) -> list[tuple[str, float]]:
        trace = trace or SearchTrace()
        if self.config.rewrite:
            self.last_rewrite = self._rewrite(query, trace)
            variants = self.last_rewrite.variants()
        else:
            variants = [query]
        trace.variants = variants
        return self._fuse(*self._lists(variants, trace), trace)

    def _pool(self, ranked: list[tuple[str, float]]) -> list[tuple[str, str]]:
        c = self.config
        return [
            (cid, index_text(self.chunks[cid], c.index_context))
            for cid, _ in ranked[: c.rerank_pool]
        ]

    def _rerank(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        trace: SearchTrace,
        scored: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank the pool, scoring only ids missing from ``scored`` (same query)."""
        c = self.config
        scored = dict(scored or {})
        pool = self._pool(ranked)
        t0 = time.perf_counter()
        try:
            missing = [(cid, text) for cid, text in pool if cid not in scored]
            info: dict = {}
            scored.update(rerank(query, missing, c.rerank_spec, info=info))
            self._note_backend(trace, info)
        except RerankUnavailable:
            return ranked
        finally:
            trace.rerank_ms += (time.perf_counter() - t0) * 1000
        reranked = sorted(((cid, scored[cid]) for cid, _ in pool), key=lambda kv: -kv[1])
        trace.reranked = True
        trace.top_rerank_score = reranked[0][1] if reranked else None
        return reranked + [(cid, float("-inf")) for cid, _ in ranked[c.rerank_pool :]]

    @staticmethod
    def _note_backend(trace: SearchTrace, info: dict) -> None:
        if info.get("backend"):
            trace.rerank_backend_used = info["backend"]
        if info.get("fallback"):
            trace.rerank_fallback = info["fallback"]

    def _skip(self, trace: SearchTrace) -> bool:
        c = self.config
        return (
            c.rerank_skip_margin is not None
            and trace.dense_margin is not None
            and trace.dense_margin >= c.rerank_skip_margin
        )

    def search(self, query: str) -> list[Hit]:
        c = self.config
        trace = SearchTrace()
        t_start = time.perf_counter()
        kind = query_kind(query) if c.rewrite and c.skip_rewrite_kinds else None
        if kind in c.skip_rewrite_kinds:
            trace.rewrite_lang, trace.rewrite_skipped, trace.variants = kind, True, [query]
            self.last_rewrite = Rewrite(query, kind, provider="skipped")
            ranked = self._fuse(*self._lists([query], trace), trace)
            if c.rerank:
                ranked = self._rerank(query, ranked, trace)
        elif c.rewrite and c.overlap_rewrite:
            ranked = self._search_overlap(query, trace)
        else:
            ranked = self.candidates(query, trace)
            skip = self._skip(trace)
            if c.rerank and not skip:
                # rerank with the clearest variant: the Malayalam-script rewrite of a
                # Manglish question beats the romanised original for a cross-encoder
                manglish = (c.rerank_with == "clearest" and trace.rewrite_lang == "manglish"
                            and len(trace.variants) > 1)  # fmt: skip
                ranked = self._rerank(trace.variants[1] if manglish else query, ranked, trace)
            trace.rerank_skipped = c.rerank and skip
        trace.total_ms = (time.perf_counter() - t_start) * 1000
        self.last_trace = trace
        return [Hit(cid, score, self.chunks[cid]) for cid, score in ranked[: c.k_final]]

    def _search_overlap(self, query: str, trace: SearchTrace) -> list[tuple[str, float]]:
        from concurrent.futures import TimeoutError as FutureTimeout

        from backend.app.query import script_lang

        c = self.config
        t_start = time.perf_counter()
        # a throwaway trace: a late rewrite must never write into this query's trace
        future = _REWRITE_POOL.submit(self._rewrite, query, SearchTrace())
        lists, weights, dense_lists = self._lists([query], trace, first=0)
        spec: dict[str, float] = {}
        if c.rerank:
            base = self._fuse(lists, weights, dense_lists, trace)
            t0 = time.perf_counter()
            try:
                info: dict = {}
                spec = dict(rerank(query, self._pool(base), c.rerank_spec, info=info))
                self._note_backend(trace, info)
            except RerankUnavailable:
                spec = {}
            trace.spec_rerank_ms = (time.perf_counter() - t0) * 1000
        if getattr(self, "on_speculative", None) is not None:
            ranked0 = (sorted(spec.items(), key=lambda kv: -kv[1]) if spec
                       else self._fuse(lists, weights, dense_lists, trace))  # fmt: skip
            self.on_speculative([cid for cid, _ in ranked0[: c.k_final]])
        t_wait = time.perf_counter()
        left = None
        if c.rewrite_deadline_s is not None:
            left = max(0.0, c.rewrite_deadline_s - (t_wait - t_start))
        try:
            rw = future.result(timeout=left)
            self._apply_rewrite(rw, trace)
        except FutureTimeout:  # provider slow / rate-limited: answer without variants
            rw = Rewrite(query, script_lang(query), error="rewrite deadline")
            trace.rewrite_timeout = True
        trace.rewrite_wait_ms = (time.perf_counter() - t_wait) * 1000
        self.last_rewrite = rw
        variants = rw.variants()
        trace.variants = variants
        if len(variants) > 1:
            more = self._lists(variants[1:], trace, first=1)
            lists, weights = lists + more[0], weights + more[1]
            dense_lists = dense_lists + more[2]
        ranked = self._fuse(lists, weights, dense_lists, trace)
        skip = self._skip(trace)
        trace.rerank_skipped = c.rerank and skip
        if not c.rerank or skip:
            return ranked
        manglish = c.rerank_with == "clearest" and rw.lang == "manglish" and len(variants) > 1
        rq = variants[1] if manglish else query
        return self._rerank(rq, ranked, trace, scored=None if manglish else spec)


CONFIGS_YAML = ROOT / "evals" / "configs.yaml"


def load_named_config(name: str, path: Path = CONFIGS_YAML) -> RetrievalConfig:
    """A named retrieval config from evals/configs.yaml (the API uses ``default``)."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in raw:
        raise KeyError(f"unknown retrieval config {name!r}; choose from {sorted(raw)}")
    return RetrievalConfig(name=name, **raw[name])


def tokenizer_config(config: RetrievalConfig) -> TokenizerConfig:
    return TokenizerConfig(stem=config.bm25_stem)
