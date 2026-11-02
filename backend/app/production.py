"""The Answerer the API serves: fallback chain, card filter, degrade mode, answer cache.

Evals build Answerer directly with all of these off, so their numbers measure one generator.
One key per provider; rate limits are handled by moving down the chain, not by rotating keys.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from pathlib import Path

from backend.app.answer import Answerer, load_retriever
from backend.app.cache import AnswerCache
from backend.app.config import ROOT, get_settings
from backend.app.llm.client import AllGeneratorsBusy, FallbackStream

CACHE_DIR = ROOT / "data" / "cache" / "api_answers"
DEMO_USAGE = ROOT / "data" / "run" / "demo_usage.json"


def generator_chain() -> list[tuple[str, str]]:
    s = get_settings()
    chain = [("groq", s.answer_model), *(("groq", m) for m in s.answer_fallback_models)]
    if s.gemini_api_key:
        chain.append(("gemini", s.answer_model_gemini))
    return chain


def corpus_version(strategy: str) -> str:
    path = ROOT / "data" / "processed" / f"chunks.{strategy}.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class NoGenerator:
    """Retrieval-only mode: behaves like a chain whose generators are all busy, with no calls."""

    model, text, tokens_in, tokens_out, ttft_ms, throttled_ms = "none", "", 0, 0, None, 0.0

    def __iter__(self) -> Iterator[str]:
        raise AllGeneratorsBusy("retrieval-only mode")
        yield ""  # pragma: no cover - makes this a generator


def retrieval_only_answerer() -> Answerer:
    s = get_settings()
    retriever = load_retriever(s.retrieval_config)
    retriever.config = replace(retriever.config, rewrite=False, overlap_rewrite=False)
    return Answerer(retriever, stream_factory=lambda messages, model: NoGenerator(), degrade=True)


class DailyCap:
    """Generated answers per day, kept on disk so a restart does not reset it. Only counts."""

    def __init__(self, limit: int, path: Path = DEMO_USAGE) -> None:
        self.limit, self.path = limit, path
        self.lock = threading.Lock()

    def take(self, today: str | None = None) -> bool:
        today = today or date.today().isoformat()
        with self.lock:
            used = 0
            if self.path.exists():
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                used = saved["count"] if saved["date"] == today else 0
            if used >= self.limit:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"date": today, "count": used + 1}), encoding="utf-8")
            return True


class DemoAnswerer(Answerer):
    """Past the daily cap: busy mode, with no generation and no rewrite. Cache hits stay free."""

    def __init__(self, retriever, *, cap: DailyCap, **kwargs) -> None:  # noqa: ANN001, ANN003
        super().__init__(retriever, **kwargs)
        self.cap, self.open_factory = cap, self.stream_factory
        self.open_config = retriever.config

    def _generate(self, question: str) -> Iterator[tuple[str, dict]]:
        if self.cap.take():
            self.stream_factory, self.retriever.config = self.open_factory, self.open_config
        else:
            self.stream_factory = lambda messages, model: NoGenerator()
            self.retriever.config = replace(self.open_config, rewrite=False, overlap_rewrite=False)
        yield from super()._generate(question)


def production_answerer() -> Answerer:
    s = get_settings()
    if s.answer_mode == "retrieval_only":
        return retrieval_only_answerer()
    retriever = load_retriever(s.retrieval_config)
    chain = generator_chain()
    extra = {"cap": DailyCap(s.demo_daily_cap)} if s.demo_mode else {}
    return (DemoAnswerer if s.demo_mode else Answerer)(
        retriever,
        stream_factory=lambda messages, model: FallbackStream(messages, chain),
        cards_filter=True,
        degrade=True,
        cache=AnswerCache(
            CACHE_DIR, corpus_version(retriever.config.strategy), s.answer_cache_days
        ),
        **extra,
    )
