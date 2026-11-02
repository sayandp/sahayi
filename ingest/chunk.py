"""Split cleaned pages into retrievable chunks.

Strategies (sizes are whitespace tokens):
  fixed          ~512-token windows, no overlap
  recursive      <= 800 tokens with 100-token overlap; paragraphs -> sentences -> words
  section_aware  split on headings (eligibility / benefits / how to apply / documents …,
                 ML + EN) and numbered clauses, merge small sections, fall back to
                 recursive for oversized ones. Default.

Chunks never cross a page boundary, so every chunk cites exactly one (doc, page).
chunk_id = f"{doc_id}:{page}:{idx}" with idx counted per page.

Reads data/processed/pages.clean.jsonl, writes data/processed/chunks.{strategy}.jsonl.
The default strategy is also written to data/processed/chunks.jsonl.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby

import regex as re

from ingest.common import PAGES_CLEAN, PROCESSED_DIR, detect_lang, read_jsonl, tokens, write_jsonl
from ingest.models import Chunk, Page

DEFAULT_STRATEGY = "section_aware"
FIXED_SIZE = 512
RECURSIVE_SIZE = 800
RECURSIVE_OVERLAP = 100
MAX_HEADING_TOKENS = 8
# Pages whose text layer was rejected and could not be OCR'd carry junk text;
# chunking them would put uncitable garbage in the index.
UNUSABLE_METHODS = {"ocr_unavailable", "ocr_failed"}
# Explicit section label written by extract.py, e.g. "[[section:eligibility]] Eligibility".
SECTION_MARKER_RE = re.compile(r"^\[\[section:([a-z_]+)\]\] ?", re.M)

Piece = tuple[str, str | None]  # (text, section label)

# ------------------------------------------------------------------------- fixed


def split_fixed(text: str, size: int = FIXED_SIZE) -> list[str]:
    toks = tokens(text)
    return [" ".join(toks[i : i + size]) for i in range(0, len(toks), size)]


# --------------------------------------------------------------------- recursive

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Sentence ends: '.', '?', '!', Devanagari danda '।' / '॥' (used in some ML docs).
SENTENCE_END = re.compile(r"(?<=[.?!।॥])\s+")
ABBREVIATION = re.compile(
    r"(?:\b(?:rs|no|nos|sl|dr|smt|sri|shri|mr|mrs|ms|viz|etc|govt|dept|vs|st|sc|i\.e|e\.g)"
    r"|(?:^|\s)\w)\.$",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    parts = [p for p in SENTENCE_END.split(text) if p.strip()]
    out: list[str] = []
    for part in parts:
        if out and ABBREVIATION.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _units(text: str, max_tokens: int) -> list[str]:
    """Paragraphs, else sentences, else word windows — each <= max_tokens."""
    units: list[str] = []
    for para in PARAGRAPH_BREAK.split(text):
        if not para.strip():
            continue
        if len(tokens(para)) <= max_tokens:
            units.append(para.strip())
            continue
        for sent in split_sentences(para):
            if len(tokens(sent)) <= max_tokens:
                units.append(sent.strip())
            else:
                units.extend(split_fixed(sent, max_tokens))
    return units


def split_recursive(
    text: str, size: int = RECURSIVE_SIZE, overlap: int = RECURSIVE_OVERLAP
) -> list[str]:
    """Greedy packing of units into chunks of <= size tokens.

    Each chunk after the first starts with the last ``overlap`` tokens of the
    previous chunk, so chunk[i+1].split()[:k] == chunk[i].split()[-k:].
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for unit in _units(text, size - overlap):
        n = len(tokens(unit))
        if cur and cur_len + n > size:
            chunks.append("\n\n".join(cur))
            tail = tokens(chunks[-1])[-overlap:] if overlap else []
            cur, cur_len = ([" ".join(tail)] if tail else []), len(tail)
        cur.append(unit)
        cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ----------------------------------------------------------------- section-aware

# Heading keywords per section label. Matched at the start of a short line, after
# optional numbering. Malayalam forms are post-clean (atomic chillus).
SECTION_KEYWORDS: dict[str, list[str]] = {
    "eligibility": [
        r"eligib\w*", r"who (?:can|is eligible|are eligible)", r"qualifications?",
        r"criteria", r"അർഹത\w*", r"യോഗ്യത\w*", r"മാനദണ്ഡ\w*",
    ],
    "exclusions": [r"exclusions?", r"not eligible", r"ineligib\w*", r"അർഹരല്ലാത്ത\w*"],
    "benefits": [
        r"benefits?", r"(?:financial )?assistance", r"entitlements?", r"amount of",
        r"ആനുകൂല്യ\w*", r"ധനസഹായ\w*", r"സഹായ ?തുക\w*",
    ],
    "how_to_apply": [
        r"how to apply", r"application (?:procedure|process)", r"procedure",
        r"mode of (?:application|selection)", r"selection",
        # Not bare "അപേക്ഷ…" (application/applicant): it starts ordinary sentences.
        r"അപേക്ഷിക്കേണ്ട (?:വിധം|രീതി)\w*", r"അപേക്ഷാ ?(?:നടപടി|രീതി|ക്രമം|സമർപ്പണം)\w*",
        r"അപേക്ഷ സമർപ്പിക്കേണ്ട (?:വിധം|രീതി)\w*", r"തിരഞ്ഞെടുപ്പ് (?:രീതി|നടപടി)\w*",
    ],
    "documents": [
        r"documents?(?: required| to be submitted)?", r"required documents",
        r"ആവശ്യമായ രേഖ\w*", r"ഹാജരാക്കേണ്ട രേഖ\w*", r"സമർപ്പിക്കേണ്ട രേഖ\w*",
    ],
    "objectives": [r"objectives?", r"aims?", r"purpose", r"ലക്ഷ്യ\w*", r"ഉദ്ദേശ്യ\w*"],
    "faq": [r"faqs?", r"frequently asked questions", r"സംശയങ്ങൾ\w*", r"ചോദ്യ\w*"],
}  # fmt: skip
NUMBERING = r"(?:\(?\d{1,2}(?:\.\d{1,2})*[.)]|\(?[ivxIVX]{1,4}[.)]|\(?[a-hA-H][.)]|[൦-൯]{1,2}[.)])"
NUMBERED_CLAUSE = re.compile(rf"^\s*{NUMBERING}\s+\S")
_HEADING_RES = {
    label: re.compile(rf"^\s*(?:{NUMBERING}\s*)?(?:{'|'.join(words)})\b", re.I)
    for label, words in SECTION_KEYWORDS.items()
}


def heading_label(line: str) -> str | None:
    """Section label if ``line`` looks like a keyword heading, else None."""
    stripped = line.strip()
    if not stripped or len(tokens(stripped)) > MAX_HEADING_TOKENS:
        return None
    if stripped.endswith("."):
        return None  # a short sentence, not a heading
    if stripped[0].islower():
        return None  # a wrapped continuation line ("purpose shall not be")
    for label, pattern in _HEADING_RES.items():
        if pattern.match(stripped):
            return label
    return None


@dataclass
class _Segment:
    label: str | None
    hard: bool  # starts at a keyword heading: never merged into the previous segment
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def strip_markers(text: str) -> str:
    return SECTION_MARKER_RE.sub("", text)


def _segments(text: str, section: str | None) -> tuple[list[_Segment], str | None]:
    """Split a page into segments at headings / numbered clauses.

    If the page carries explicit ``[[section:x]]`` markers (myscheme pages), they
    are the only source of section labels; keyword headings inside them are ignored.
    """
    segs: list[_Segment] = []
    cur = _Segment(section, hard=False)
    explicit = SECTION_MARKER_RE.search(text) is not None
    for line in text.split("\n"):
        marker = SECTION_MARKER_RE.match(line)
        if marker:
            label: str | None = marker.group(1)
            line = line[marker.end() :]
        else:
            label = None if explicit else heading_label(line)
        if label or NUMBERED_CLAUSE.match(line):
            if cur.text:
                segs.append(cur)
            section = label or section
            cur = _Segment(section, hard=label is not None)
        cur.lines.append(line)
    if cur.text:
        segs.append(cur)
    return segs, section


def split_sections(
    text: str,
    section: str | None = None,
    size: int = RECURSIVE_SIZE,
    overlap: int = RECURSIVE_OVERLAP,
) -> tuple[list[Piece], str | None]:
    """Return (pieces, section label in force at the end of the page)."""
    segs, section = _segments(text, section)
    merged: list[_Segment] = []
    for seg in segs:
        prev = merged[-1] if merged else None
        fits = prev is not None and len(tokens(prev.text)) + len(tokens(seg.text)) <= size
        if fits and not seg.hard and prev.label == seg.label:
            prev.lines.extend(seg.lines)
        else:
            merged.append(seg)
    pieces: list[Piece] = []
    for seg in merged:
        if len(tokens(seg.text)) <= size:
            pieces.append((seg.text, seg.label))
        else:
            pieces.extend((t, seg.label) for t in split_recursive(seg.text, size, overlap))
    return pieces, section


# ---------------------------------------------------------------------- interface


def _make_chunk(page: Page, idx: int, text: str, strategy: str, section: str | None) -> Chunk:
    return Chunk(
        chunk_id=f"{page.doc_id}:{page.page}:{idx}",
        doc_id=page.doc_id,
        scheme=page.scheme,
        department=page.department,
        source_url=page.source_url,
        page=page.page,
        lang=detect_lang(text),
        downloaded_at=page.downloaded_at,
        text=text,
        scheme_id=page.scheme_id,
        level=page.level,
        strategy=strategy,
        section=section,
        n_tokens=len(tokens(text)),
        method=page.method,
    )


def _page_pieces(
    strategy: str, kw: dict
) -> Callable[[str, str | None], tuple[list[Piece], str | None]]:
    if strategy == "fixed":
        size = kw.get("size", FIXED_SIZE)
        return lambda text, sec: (
            [(t, None) for t in split_fixed(strip_markers(text), size)],
            sec,
        )
    size = kw.get("size", RECURSIVE_SIZE)
    overlap = kw.get("overlap", RECURSIVE_OVERLAP)
    if strategy == "recursive":
        return lambda text, sec: (
            [(t, None) for t in split_recursive(strip_markers(text), size, overlap)],
            sec,
        )
    if strategy == "section_aware":
        return lambda text, sec: split_sections(text, sec, size, overlap)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


STRATEGIES = ("fixed", "recursive", "section_aware")


def chunk(pages: list[Page], strategy: str = DEFAULT_STRATEGY, **kw: int) -> list[Chunk]:
    """Chunk pages with one strategy. Section labels carry across pages of a doc."""
    split = _page_pieces(strategy, kw)
    out: list[Chunk] = []
    ordered = sorted(pages, key=lambda p: (p.doc_id, p.page))
    for _, doc_pages in groupby(ordered, key=lambda p: p.doc_id):
        section: str | None = None
        for page in doc_pages:
            if not page.text.strip() or page.method in UNUSABLE_METHODS:
                continue
            if page.personal_data or page.excluded:
                continue  # named beneficiary/rank lists: personal data, no scheme content
            pieces, section = split(page.text, section)
            if page.section_hint:  # e.g. a reviewed table page
                pieces = [(t, page.section_hint) for t, _ in pieces]
            texts = [(t.strip(), s) for t, s in pieces if t.strip()]
            out.extend(_make_chunk(page, i, t, strategy, s) for i, (t, s) in enumerate(texts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=[*STRATEGIES, "all"])
    args = ap.parse_args()
    pages = list(read_jsonl(PAGES_CLEAN, Page))
    for strategy in STRATEGIES if args.strategy == "all" else (args.strategy,):
        path = PROCESSED_DIR / f"chunks.{strategy}.jsonl"
        n = write_jsonl(path, chunk(pages, strategy))
        print(f"{strategy}: {n} chunks -> {path.name}")
        if strategy == DEFAULT_STRATEGY:
            shutil.copyfile(path, PROCESSED_DIR / "chunks.jsonl")


if __name__ == "__main__":
    main()
