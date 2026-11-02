"""Clean extracted page text.

Per page: fix mojibake (ftfy) -> NFC -> atomic Malayalam chillus -> ZWJ/ZWNJ only
inside Malayalam words -> whitespace collapse -> drop page-number lines.
Per document: drop header/footer lines that repeat on more than half the pages.

Reads data/processed/pages.jsonl, writes data/processed/pages.clean.jsonl.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from itertools import groupby

import ftfy
import regex as re

from ingest.common import (
    PAGES_CLEAN,
    PAGES_RAW,
    detect_lang,
    is_malayalam,
    ml_ratio,
    read_jsonl,
    write_jsonl,
)
from ingest.models import Page

ZWJ, ZWNJ = "‍", "‌"
VIRAMA = "്"
# Pre-Unicode-5.1 chillu spelling (consonant + virama + ZWJ) -> atomic chillu.
# Both render identically, but mixing them splits one word into two BM25 terms
# (e.g. "പെന്‍ഷന്‍" vs "പെൻഷൻ").
CHILLU = {
    "ണ" + VIRAMA + ZWJ: "ൺ",  # ൺ
    "ന" + VIRAMA + ZWJ: "ൻ",  # ൻ
    "ര" + VIRAMA + ZWJ: "ർ",  # ർ
    "ല" + VIRAMA + ZWJ: "ൽ",  # ൽ
    "ള" + VIRAMA + ZWJ: "ൾ",  # ൾ
    "ക" + VIRAMA + ZWJ: "ൿ",  # ൿ
}
INVISIBLES = re.compile("[­​⁠﻿]")  # soft hyphen, ZWSP, WJ, BOM
DOT_LEADER = re.compile(r"(?:\s*\.){4,}")  # ". . . . ." in tables of contents
SPACES = re.compile(r"[ \t   -   　]+")
DIGIT = r"[0-9൦-൯]"
# "Page 3", "Page 3 of 10", "3 / 10", "- 3 -", "പേജ് 3" — safe to drop anywhere.
PAGE_LABEL = re.compile(
    rf"^[-–—(\[]?\s*(?:page|pg\.?|പേജ്)\s*{DIGIT}{{1,4}}"
    rf"(?:\s*(?:/|of|-)\s*{DIGIT}{{1,4}})?\s*[-–—)\]]?$"
    rf"|^{DIGIT}{{1,4}}\s*(?:/|of)\s*{DIGIT}{{1,4}}$"
    rf"|^[-–—]\s*{DIGIT}{{1,4}}\s*[-–—]$",
    re.I,
)
# A bare number is only a page number at the top/bottom edge of a page;
# in the body it may be an amount in a table ("6000").
BARE_NUMBER = re.compile(rf"^[(\[]?{DIGIT}{{1,4}}[)\]]?$")
EDGE_LINES = 3
PAGE_NUMBER_EDGE = 2
REPEAT_SHARE = 0.5
MIN_PAGES_FOR_REPEATS = 3


def fix_encoding(text: str) -> str:
    return unicodedata.normalize("NFC", ftfy.fix_text(text, normalization="NFC"))


def normalise_chillus(text: str) -> str:
    for old, new in CHILLU.items():
        text = text.replace(old, new)
    return text


def normalise_joiners(text: str) -> str:
    """Keep ZWJ/ZWNJ only between two Malayalam characters; drop elsewhere."""
    text = INVISIBLES.sub("", text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in (ZWJ, ZWNJ):
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not (prev and nxt and is_malayalam(prev) and is_malayalam(nxt)):
                continue
        out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [SPACES.sub(" ", DOT_LEADER.sub(" … ", line)).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalise_text(text: str) -> str:
    return collapse_whitespace(normalise_joiners(normalise_chillus(fix_encoding(text))))


def _edge_indices(lines: list[str], n: int) -> set[int]:
    """Indices of the first/last ``n`` non-empty lines.

    ``n`` shrinks to a third of the page so that on short pages the "edges" never
    cover the body (otherwise similar body lines look like repeated headers).
    """
    idx = [i for i, line in enumerate(lines) if line.strip()]
    n = min(n, max(1, len(idx) // 3))
    return set(idx[:n] + idx[-n:])


def remove_page_numbers(text: str) -> str:
    lines = text.split("\n")
    edges = _edge_indices(lines, PAGE_NUMBER_EDGE)
    kept = [
        line
        for i, line in enumerate(lines)
        if not PAGE_LABEL.match(line.strip())
        and not (i in edges and BARE_NUMBER.match(line.strip()))
    ]
    return "\n".join(kept)


def _line_key(line: str, page_no: int) -> str:
    """Header/footer comparison key: case/spacing-insensitive, own page number masked.

    Only the page's own number is masked ("Page 3 of 12" == "Page 4 of 12"); other
    digits must match, so numbered body lines are not mistaken for footers.
    """
    key = " ".join(line.lower().split())
    return re.sub(rf"(?<!\d){page_no}(?!\d)", "#", key, count=1)  # "page 4 of 4"


def find_repeated_edge_lines(
    texts: list[str], page_numbers: list[int] | None = None, share: float = REPEAT_SHARE
) -> set[str]:
    """Keys of lines in the top/bottom EDGE_LINES that occur on > share of pages."""
    if len(texts) < MIN_PAGES_FOR_REPEATS:
        return set()
    page_numbers = page_numbers or list(range(1, len(texts) + 1))
    counts: Counter[str] = Counter()
    for text, page_no in zip(texts, page_numbers, strict=True):
        lines = text.split("\n")
        counts.update({_line_key(lines[i], page_no) for i in _edge_indices(lines, EDGE_LINES)})
    return {k for k, c in counts.items() if k and c > share * len(texts)}


def remove_lines(text: str, keys: set[str], page_no: int) -> str:
    lines = text.split("\n")
    edges = _edge_indices(lines, EDGE_LINES)
    return "\n".join(
        line
        for i, line in enumerate(lines)
        if not (i in edges and _line_key(line, page_no) in keys)
    )


def clean_doc(pages: list[Page]) -> list[Page]:
    """Clean all pages of one document (pages must share a doc_id)."""
    texts = [remove_page_numbers(normalise_text(p.text)) for p in pages]
    repeated = find_repeated_edge_lines(texts, [p.page for p in pages])
    out: list[Page] = []
    for page, text in zip(pages, texts, strict=True):
        text = collapse_whitespace(remove_lines(text, repeated, page.page))
        out.append(
            page.model_copy(
                update={
                    "text": text,
                    "lang_detected": detect_lang(text),
                    "ml_ratio": round(ml_ratio(text), 4),
                }
            )
        )
    return out


def clean_pages(pages: list[Page]) -> list[Page]:
    ordered = sorted(pages, key=lambda p: (p.doc_id, p.page))
    out: list[Page] = []
    for _, doc_pages in groupby(ordered, key=lambda p: p.doc_id):
        out.extend(clean_doc(list(doc_pages)))
    return out


def main() -> None:
    pages = clean_pages(list(read_jsonl(PAGES_RAW, Page)))
    n = write_jsonl(PAGES_CLEAN, pages)
    empty = sum(1 for p in pages if not p.text)
    print(f"cleaned {n} pages ({empty} empty after cleaning)")


if __name__ == "__main__":
    main()
