"""Paths, JSONL helpers and small text utilities shared by the ingestion stages."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ingest.models import Lang, PageReview, Source

ROOT = Path(__file__).resolve().parents[1]
SOURCES_YAML = ROOT / "ingest" / "sources.yaml"
PAGE_REVIEW_YAML = ROOT / "ingest" / "page_review.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST = RAW_DIR / "manifest.jsonl"
MANUAL_DIR = RAW_DIR / "manual"
LOW_CONF_DIR = DATA_DIR / "processed" / "low_conf"
REOCR_DIR = DATA_DIR / "processed" / "reocr"
OCR_CACHE_DIR = DATA_DIR / "cache" / "ocr"
PAGES_RAW = PROCESSED_DIR / "pages.jsonl"
PAGES_CLEAN = PROCESSED_DIR / "pages.clean.jsonl"
EXTRACT_ERRORS = PROCESSED_DIR / "extract_errors.jsonl"
REPORT_MD = PROCESSED_DIR / "ingest_report.md"

ML_START, ML_END = "ഀ", "ൿ"
ML_LANG_THRESHOLD = 0.3

M = TypeVar("M", bound=BaseModel)


def load_sources(path: Path = SOURCES_YAML) -> list[Source]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = [Source.model_validate(s) for s in data["sources"]]
    ids = [s.id for s in sources]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate source ids: {sorted(dupes)}")
    return sources


def load_page_reviews(path: Path = PAGE_REVIEW_YAML) -> dict[tuple[str, int], PageReview]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reviews = [PageReview.model_validate(r) for r in data.get("pages", [])]
    return {(r.doc_id, r.page): r for r in reviews}


def parse_pages(spec: str | None) -> set[int] | None:
    """'6-9,16-18,25' -> {6, 7, 8, 9, 16, 17, 18, 25}; None means all pages."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        pages.update(range(int(lo), int(hi or lo) + 1))
    return pages


def read_jsonl(path: Path, model: type[M]) -> Iterator[M]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
            n += 1
    return n


def is_malayalam(ch: str) -> bool:
    return ML_START <= ch <= ML_END


def ml_ratio(text: str) -> float:
    """Share of script characters that are Malayalam (U+0D00–U+0D7F).

    Malayalam vowel signs are combining marks, so they are counted by range rather
    than with str.isalpha().
    """
    ml = sum(1 for ch in text if is_malayalam(ch))
    other = sum(1 for ch in text if ch.isalpha() and not is_malayalam(ch))
    total = ml + other
    return ml / total if total else 0.0


def detect_lang(text: str) -> Lang:
    return "ml" if ml_ratio(text) >= ML_LANG_THRESHOLD else "en"


MYSCHEME_REQUIRED = "#details"  # present only when a myscheme page really rendered


def is_myscheme(url: str) -> bool:
    return "myscheme.gov.in/schemes/" in url


def tokens(text: str) -> list[str]:
    """Whitespace tokens; the token unit used for all chunk sizes."""
    return text.split()
