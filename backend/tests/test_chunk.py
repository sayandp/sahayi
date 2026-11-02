"""Tests for ingest/chunk.py: metadata, ids, empties, overlap, sections."""

from __future__ import annotations

import pytest

from ingest.chunk import (
    STRATEGIES,
    chunk,
    heading_label,
    split_fixed,
    split_recursive,
    split_sections,
    split_sentences,
)
from ingest.models import Chunk, Page

REQUIRED = [
    "chunk_id", "doc_id", "scheme", "department", "source_url",
    "page", "lang", "downloaded_at", "text",
]  # fmt: skip

ML_DOC = """ലൈഫ് ഭവന പദ്ധതി

ലക്ഷ്യം
സംസ്ഥാനത്തെ എല്ലാ ഭൂരഹിത ഭവനരഹിതർക്കും സുരക്ഷിതമായ വീട് നൽകുക.

അർഹത മാനദണ്ഡങ്ങൾ
1. അപേക്ഷകൻ കേരളത്തിൽ സ്ഥിരതാമസക്കാരൻ ആയിരിക്കണം.
2. കുടുംബ വാർഷിക വരുമാനം മൂന്ന് ലക്ഷം രൂപയിൽ താഴെ ആയിരിക്കണം.

ആനുകൂല്യങ്ങൾ
ഭവന നിർമ്മാണത്തിന് നാല് ലക്ഷം രൂപ ധനസഹായം.

ആവശ്യമായ രേഖകൾ
റേഷൻ കാർഡ്, വരുമാന സർട്ടിഫിക്കറ്റ്, ആധാർ."""

EN_DOC = """PM-KISAN Operational Guidelines

Eligibility
All landholding farmer families are eligible. Exclusions apply to institutional landholders.

Benefits:
Rs. 6000 per year in three instalments of Rs. 2000 each.

How to apply
Farmers approach the local patwari or revenue officer or register on the portal."""


def make_page(text: str, page: int = 1, doc_id: str = "pm_kisan--og-abc123") -> Page:
    return Page(
        doc_id=doc_id,
        scheme_id="pm_kisan",
        scheme="PM-KISAN",
        department="DA&FW",
        level="central",
        source_url="https://pmkisan.gov.in/Documents/og.pdf",
        downloaded_at="2026-10-12",
        page=page,
        text=text,
        method="text",
        lang_detected="en",
        ml_ratio=0.0,
    )


def long_text(n_words: int) -> str:
    sentences = []
    for i in range(0, n_words, 10):
        sentences.append(" ".join(f"w{j}" for j in range(i, min(i + 10, n_words))) + ".")
    paras = [" ".join(sentences[k : k + 7]) for k in range(0, len(sentences), 7)]
    return "\n\n".join(paras)


@pytest.fixture
def pages() -> list[Page]:
    return [
        make_page(EN_DOC, 1),
        make_page(long_text(2000), 2),
        make_page("   \n\n  ", 3),  # empty after extraction
        make_page(ML_DOC, 1, doc_id="life_mission--life-project-def456"),
    ]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_metadata_complete(pages: list[Page], strategy: str) -> None:
    chunks = chunk(pages, strategy)
    assert chunks
    for c in chunks:
        dumped = c.model_dump()
        for key in REQUIRED:
            assert dumped[key] not in (None, ""), key
        assert c.chunk_id == f"{c.doc_id}:{c.page}:{c.chunk_id.rsplit(':', 1)[1]}"
        assert c.strategy == strategy
        Chunk.model_validate(dumped)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_ids_unique_and_contiguous(pages: list[Page], strategy: str) -> None:
    chunks = chunk(pages, strategy)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    by_page: dict[tuple[str, int], list[int]] = {}
    for c in chunks:
        by_page.setdefault((c.doc_id, c.page), []).append(int(c.chunk_id.rsplit(":", 1)[1]))
    for idxs in by_page.values():
        assert idxs == list(range(len(idxs)))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_empty_chunks_and_empty_page_skipped(pages: list[Page], strategy: str) -> None:
    chunks = chunk(pages, strategy)
    assert all(c.text.strip() and c.n_tokens > 0 for c in chunks)
    assert not any(c.page == 3 for c in chunks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_text_covered(pages: list[Page], strategy: str) -> None:
    page = pages[1]
    covered = {t for c in chunk([page], strategy) for t in c.text.split()}
    assert covered == set(page.text.split())


def test_lang_per_chunk(pages: list[Page]) -> None:
    langs = {c.doc_id.split("--")[0]: c.lang for c in chunk(pages, "section_aware")}
    assert langs == {"pm_kisan": "en", "life_mission": "ml"}


def test_fixed_sizes() -> None:
    parts = split_fixed(long_text(1100), 512)
    assert [len(p.split()) for p in parts] == [512, 512, 76]


@pytest.mark.parametrize(("size", "overlap"), [(800, 100), (120, 30), (50, 0)])
def test_recursive_overlap_is_exact(size: int, overlap: int) -> None:
    parts = split_recursive(long_text(2500), size, overlap)
    assert len(parts) > 2
    for prev, nxt in zip(parts, parts[1:], strict=False):
        p, n = prev.split(), nxt.split()
        assert len(p) <= size and len(n) <= size
        k = min(overlap, len(p))
        if k:
            assert n[:k] == p[-k:]


def test_recursive_short_text_single_chunk() -> None:
    assert split_recursive("ഒരു ചെറിയ ഖണ്ഡിക.", 800, 100) == ["ഒരു ചെറിയ ഖണ്ഡിക."]


def test_recursive_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        split_recursive("x", 100, 100)


def test_sentences_split_on_malayalam_full_stop_and_danda() -> None:
    text = "അപേക്ഷകൻ കേരളീയൻ ആയിരിക്കണം. വരുമാനം പരിധിക്കുള്ളിൽ ആയിരിക്കണം। രേഖകൾ ഹാജരാക്കണം."
    assert len(split_sentences(text)) == 3


def test_sentences_do_not_split_on_abbreviations() -> None:
    text = "Assistance of Rs. 6000 is paid. Contact Smt. Devi or Dr. Nair for details."
    assert split_sentences(text) == [
        "Assistance of Rs. 6000 is paid.",
        "Contact Smt. Devi or Dr. Nair for details.",
    ]


@pytest.mark.parametrize(
    ("line", "label"),
    [
        ("Eligibility", "eligibility"),
        ("Eligibility Criteria:", "eligibility"),
        ("3. How to Apply", "how_to_apply"),
        ("Documents Required", "documents"),
        ("Benefits:", "benefits"),
        ("അർഹത മാനദണ്ഡങ്ങൾ", "eligibility"),
        ("യോഗ്യത", "eligibility"),
        ("ആനുകൂല്യങ്ങൾ", "benefits"),
        ("അപേക്ഷിക്കേണ്ട വിധം", "how_to_apply"),
        ("ആവശ്യമായ രേഖകൾ", "documents"),
        ("ഹാജരാക്കേണ്ട രേഖകൾ", "documents"),
        ("Benefits will be paid to bank accounts.", None),  # sentence, not heading
        # Regression: bare "അപേക്ഷ…" starts ordinary lines in ML circulars.
        ("അപേക്ഷയിൽ സ്വന്തം ഇ-മെയിൽ വിലാസവും", None),
        ("അപേക്ഷ ക്ഷണിക്കുന്നു", None),
        ("അപേക്ഷാ നടപടിക്രമം", "how_to_apply"),
        ("സമർപ്പിക്കേണ്ട രേഖകൾ", "documents"),
        ("All landholding farmer families are eligible", None),
        ("The scheme provides benefits to farmers in all districts of the state", None),
    ],
)
def test_heading_label(line: str, label: str | None) -> None:
    assert heading_label(line) == label


def test_section_aware_splits_on_headings_ml() -> None:
    pieces, last = split_sections(ML_DOC)
    labels = [label for _, label in pieces]
    assert labels == [None, "objectives", "eligibility", "benefits", "documents"]
    elig = pieces[2][0]
    assert "1. അപേക്ഷകൻ" in elig and "2. കുടുംബ" in elig  # numbered clauses merged
    assert last == "documents"


def test_section_aware_splits_on_headings_en() -> None:
    pieces, _ = split_sections(EN_DOC)
    assert [label for _, label in pieces] == [None, "eligibility", "benefits", "how_to_apply"]


def test_section_label_carries_to_next_page() -> None:
    p1 = make_page("Eligibility\nApplicant must be a resident of Kerala.", 1)
    p2 = make_page("Annual family income must be below Rs. 1,00,000.", 2)
    chunks = chunk([p1, p2], "section_aware")
    assert [(c.page, c.section) for c in chunks] == [(1, "eligibility"), (2, "eligibility")]


def test_section_aware_oversized_section_falls_back_to_recursive() -> None:
    text = "Eligibility\n" + long_text(2000)
    pieces, _ = split_sections(text, size=800, overlap=100)
    assert len(pieces) >= 3
    assert all(len(t.split()) <= 800 and label == "eligibility" for t, label in pieces)


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        chunk([make_page("x")], "semantic")


def test_wrapped_lowercase_line_is_not_a_heading() -> None:
    # Regression: PDF line-wraps like "purpose shall not be" were labelled objectives.
    assert heading_label("purpose shall not be") is None
    assert heading_label("assistance under this scheme") is None
    assert heading_label("Purpose") == "objectives"


@pytest.mark.parametrize("method", ["ocr_unavailable", "ocr_failed"])
def test_pages_without_usable_text_are_not_chunked(method: str) -> None:
    junk = make_page("രജിേΒഷȟϗ നിÓേദശŅൾ " * 20, 1).model_copy(update={"method": method})
    good = make_page(EN_DOC, 2)
    chunks = chunk([junk, good], "section_aware")
    assert chunks and {c.page for c in chunks} == {2}
