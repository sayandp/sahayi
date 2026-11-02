"""BM25 retrieval with Malayalam-aware tokenisation.

Queries and chunks go through the same normaliser as ingestion (NFC, atomic chillus,
ZWJ/ZWNJ only inside Malayalam words), so "പെന്‍ഷന്‍" in a query matches "പെൻഷൻ" in a
document. Malayalam is agglutinative ("പെൻഷൻ", "പെൻഷന്റെ", "പെൻഷനുള്ള"), so each
Malayalam token is also indexed as a light stem with common case/plural suffixes
removed; the exact token is kept too, so exact matches still score higher.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re
from rank_bm25 import BM25Okapi

from ingest.clean import normalise_text

TOKEN = re.compile(r"[\p{L}\p{M}\p{N}‌‍]+")
ML_CHAR = re.compile(r"[ഀ-ൿ]")
JOINERS = re.compile("[‌‍]")
# Longest first. Case endings, plural, conjunctive "-ഉം", copula "-ആണ്".
ML_SUFFIXES = sorted(
    [
        "ത്തിന്റെ", "ത്തിനുള്ള", "ത്തിനു", "ത്തിന്", "ത്തിൽ", "ത്തെ", "ത്താൽ",
        "യുടെ", "ിന്റെ", "ന്റെ", "ുടെ", "ിനുള്ള", "നുള്ള", "ുള്ള", "യുള്ള",
        "ിന്", "ിനു", "ക്ക്", "ക്കു", "ിൽ", "യിൽ", "ിലെ", "യിലെ", "ിലൂടെ",
        "ങ്ങൾ", "ങ്ങളുടെ", "ങ്ങളും", "ുകൾ", "കൾ", "ുകളും", "കളും",
        "ും", "യും", "യാണ്", "ാണ്", "യെ", "യോ", "ോ", "േ", "െ",
    ],
    key=len,
    reverse=True,
)  # fmt: skip
MIN_STEM_CHARS = 3


@dataclass(frozen=True)
class TokenizerConfig:
    stem: bool = True
    lowercase: bool = True


DEFAULT_TOKENIZER = TokenizerConfig()


def ml_stem(token: str) -> str:
    """Strip one common suffix if a stem of >= MIN_STEM_CHARS remains."""
    for suffix in ML_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_CHARS:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, config: TokenizerConfig = DEFAULT_TOKENIZER) -> list[str]:
    text = normalise_text(text)
    if config.lowercase:
        text = text.lower()
    out: list[str] = []
    for tok in TOKEN.findall(text):
        if not ML_CHAR.search(tok):
            out.append(tok)
            continue
        tok = JOINERS.sub("", tok)  # spelling variants with/without joiners match
        out.append(tok)
        if config.stem:
            stem = ml_stem(tok)
            if stem != tok:
                out.append(stem)
    return out


class BM25Index:
    """BM25Okapi over chunk texts; search returns (chunk_id, score) best first."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        config: TokenizerConfig = DEFAULT_TOKENIZER,
    ) -> None:
        self.chunk_ids = list(chunk_ids)
        self.config = config
        self.bm25 = BM25Okapi([tokenize(t, config) or ["∅"] for t in texts])

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        q = tokenize(query, self.config)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> BM25Index:
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - our own index file
