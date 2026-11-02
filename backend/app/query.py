"""Query rewrite for cross-lingual retrieval (pipeline step 2).

One LLM call turns a question in Malayalam, English or Manglish (romanised Malayalam)
into a Malayalam-script and an English variant; retrieval runs all variants and fuses
them with RRF. Fallback chain: Groq -> Gemini (if a key is set) -> no rewrite.

Caching: rewrites are cached by sha1(prompt version, model, query). The disk cache
stores only {lang, ml_query, en_query} - never profile hints - and is on for evals;
the API should keep it off or in memory (CLAUDE.md: no PII stored).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.llm.client import LLMError, chat_json
from backend.app.llm.prompts import QUERY_REWRITE_SYSTEM, QUERY_REWRITE_VERSION

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "rewrite"
DEFAULT_REWRITE_MODEL = "qwen/qwen3.8-27b"
ML_CHAR = re.compile("[ഀ-ൿ]")
LANGS = {"ml", "en", "manglish", "mixed"}


@dataclass
class Rewrite:
    original: str
    lang: str
    ml_query: str | None = None
    en_query: str | None = None
    profile_hints: dict = field(default_factory=dict)
    provider: str = "none"
    model: str | None = None
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None
    throttled_ms: float = 0.0  # provider rate-limit waits (not user-facing latency)
    tokens_in: int = 0
    tokens_out: int = 0

    def variants(self) -> list[str]:
        """Original first, then distinct non-empty rewrites."""
        out: list[str] = []
        for q in (self.original, self.ml_query, self.en_query):
            if q and q.strip() and q.strip() not in out:
                out.append(q.strip())
        return out


def script_lang(query: str) -> str:
    """Cheap guess without an LLM: Malayalam script or not."""
    return "ml" if ML_CHAR.search(query) else "en"


# Romanised Malayalam (Manglish) markers: function words and verb endings that English
# text does not use. A miss only means a Manglish question is treated as English.
_MANGLISH = re.compile(
    r"\b(?:enikk?u|ente|njan|nja?ngal|nammal|ningal|avan|aval|undo|und|ille?|illa|alle|"
    r"aanu|ano|aano|entha|enthu|engane|evide|ethra|eppol|vayass?u?|vayas|kudumbam|"
    r"bharthavu?|bharya|amma|achan|makan|makal|joli|panam|roopa|rupa|padhathi|apeksha|"
    r"kittumo|kittum|kittan|kitti|labhikkumo|labhikkum|cheyyan|cheyyanam|venam|venda|"
    r"\w+(?:umo|unnu|illa))\b",
    re.I,
)
_LATIN = re.compile("[A-Za-z]")


def query_kind(query: str) -> str:
    """Without an LLM: "ml" (mostly Malayalam script), "mixed", "manglish" or "en"."""
    ml, latin = len(ML_CHAR.findall(query)), len(_LATIN.findall(query))
    if ml:
        return "ml" if ml >= latin else "mixed"
    return "manglish" if _MANGLISH.search(query) else "en"


def _cache_path(query: str, model: str) -> Path:
    key = hashlib.sha1(f"{QUERY_REWRITE_VERSION}\x00{model}\x00{query}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _clean(data: dict, query: str) -> tuple[str, str | None, str | None, dict]:
    lang = data.get("lang") if data.get("lang") in LANGS else script_lang(query)
    ml = data.get("ml_query") if isinstance(data.get("ml_query"), str) else None
    en = data.get("en_query") if isinstance(data.get("en_query"), str) else None
    if ml and not ML_CHAR.search(ml):
        ml = None  # the "Malayalam" variant must actually be in Malayalam script
    hints = data.get("profile_hints") if isinstance(data.get("profile_hints"), dict) else {}
    return lang, ml, en, {k: v for k, v in hints.items() if v is not None}


def rewrite(query: str, *, model: str = DEFAULT_REWRITE_MODEL, use_cache: bool = True) -> Rewrite:
    t0 = time.perf_counter()
    use_cache = use_cache and os.environ.get("SAHAYI_REWRITE_CACHE", "1") != "0"
    path = _cache_path(query, model)
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return Rewrite(query, data["lang"], data.get("ml_query"), data.get("en_query"),
                       provider="cache", model=model, cached=True,
                       latency_ms=(time.perf_counter() - t0) * 1000)  # fmt: skip
    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": query},
    ]
    try:
        result = chat_json(messages, groq_model=model, allow_gemini=True)
    except LLMError as exc:
        return Rewrite(query, script_lang(query), error=str(exc)[:300],
                       latency_ms=(time.perf_counter() - t0) * 1000)  # fmt: skip
    lang, ml, en, hints = _clean(result.data or {}, query)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"lang": lang, "ml_query": ml, "en_query": en},
                                   ensure_ascii=False), encoding="utf-8")  # fmt: skip
    return Rewrite(query, lang, ml, en, hints, provider=result.provider, model=result.model,
                   latency_ms=result.latency_ms, throttled_ms=result.throttled_ms,
                   tokens_in=result.tokens_in, tokens_out=result.tokens_out)  # fmt: skip
