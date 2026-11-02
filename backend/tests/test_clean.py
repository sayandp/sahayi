"""Tests for ingest/clean.py using real Malayalam scheme text."""

from __future__ import annotations

import unicodedata

from ingest.clean import (
    ZWJ,
    ZWNJ,
    clean_doc,
    collapse_whitespace,
    find_repeated_edge_lines,
    fix_encoding,
    normalise_chillus,
    normalise_joiners,
    normalise_text,
    remove_page_numbers,
)
from ingest.models import Page

# From welfarepension.lsgkerala.gov.in (legacy chillu spelling with ZWJ).
LEGACY = "സാമൂഹ്യ സുരക്ഷാ പെന്\u200dഷന്\u200d സ്കീമുകള്\u200d"
ATOMIC = "സാമൂഹ്യ സുരക്ഷാ പെൻഷൻ സ്കീമുകൾ"


def make_page(text: str, page: int = 1, doc_id: str = "d1") -> Page:
    return Page(
        doc_id=doc_id,
        scheme_id="s",
        scheme="S",
        department="D",
        level="state",
        source_url="https://example.gov.in/x.pdf",
        downloaded_at="2026-10-12",
        page=page,
        text=text,
        method="text",
        lang_detected="en",
        ml_ratio=0.0,
    )


def test_legacy_chillus_become_atomic() -> None:
    assert normalise_chillus(LEGACY) == ATOMIC
    assert normalise_text(LEGACY) == ATOMIC


def test_all_six_chillus() -> None:
    legacy = "".join(c + "\u0d4d" + ZWJ for c in "ണനരലളക")
    assert normalise_chillus(legacy) == "\u0d7a\u0d7b\u0d7c\u0d7d\u0d7e\u0d7f"


def test_zwnj_kept_inside_malayalam_word() -> None:
    # "ലാപ്‌ടോപ്പ്" (laptop) from an egrantz notice: ZWNJ blocks the പ്ട conjunct.
    word = "ലാപ്" + ZWNJ + "ടോപ്പ്"
    assert normalise_joiners(word) == word
    assert normalise_text(f"പട്ടികവർഗ്ഗ വിദ്യാർത്ഥികൾക്ക് {word}").count(ZWNJ) == 1


def test_joiners_dropped_outside_malayalam() -> None:
    assert normalise_joiners(f"PM{ZWJ}-KISAN") == "PM-KISAN"
    assert normalise_joiners(f"Scheme{ZWNJ} പദ്ധതി") == "Scheme പദ്ധതി"
    assert normalise_joiners(f"പദ്ധതി{ZWJ} ") == "പദ്ധതി "  # word-final, before space
    assert normalise_joiners(f"{ZWNJ}പദ്ധതി") == "പദ്ധതി"


def test_invisibles_removed() -> None:
    assert normalise_joiners("ആനുകൂല്യ\u200bങ്ങൾ\ufeff") == "ആനുകൂല്യങ്ങൾ"


def test_mojibake_fixed() -> None:
    broken = "കർഷക പെൻഷൻ".encode().decode("latin-1")
    assert broken != "കർഷക പെൻഷൻ"
    assert fix_encoding(broken) == "കർഷക പെൻഷൻ"


def test_output_is_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Café ലൈഫ് മിഷൻ")
    assert unicodedata.is_normalized("NFC", normalise_text(decomposed))


def test_collapse_whitespace_keeps_paragraphs() -> None:
    text = "അർഹത  മാനദണ്ഡങ്ങൾ\u00a0\t:\r\n\r\n\r\n\r\n1.  കേരളത്തിൽ   സ്ഥിരതാമസം  "
    assert collapse_whitespace(text) == "അർഹത മാനദണ്ഡങ്ങൾ :\n\n1. കേരളത്തിൽ സ്ഥിരതാമസം"


def test_page_number_lines_removed() -> None:
    text = "3\nEligibility\nAnnual income below Rs. 1,00,000\nPage 3 of 12\n- 4 -\nപേജ് ൫"
    assert remove_page_numbers(text) == "Eligibility\nAnnual income below Rs. 1,00,000"


def test_bare_number_in_body_is_kept() -> None:
    # Table cells come out one per line; an amount mid-page must survive.
    text = "Scheme\nAmount\nPM-KISAN\n6000\nper year\npaid in three instalments\nend"
    assert "6000" in remove_page_numbers(text).split("\n")


def test_repeated_header_and_footer_removed() -> None:
    pages = [
        make_page(
            "Government of Kerala\nസാമൂഹ്യനീതി വകുപ്പ്\n"
            + "\n".join(f"Body text {chr(96 + i)}{j} about വിധവ പെൻഷൻ." for j in range(6))
            + f"\nwww.swd.kerala.gov.in | Page {i} of 4",
            page=i,
        )
        for i in range(1, 5)
    ]
    cleaned = clean_doc(pages)
    for i, p in enumerate(cleaned, start=1):
        assert "Government of Kerala" not in p.text
        assert "സാമൂഹ്യനീതി വകുപ്പ്" not in p.text
        assert "www.swd.kerala.gov.in" not in p.text
        assert all(f"Body text {chr(96 + i)}{j}" in p.text for j in range(6))


def test_short_pages_keep_similar_body_lines() -> None:
    # Regression: on 3-line pages every line was an "edge" line, and body lines
    # differing only by digits were removed as a repeated header.
    pages = [make_page(f"Title\nBody text {i} about പെൻഷൻ.\nEnd", page=i) for i in range(1, 5)]
    for i, p in enumerate(clean_doc(pages), start=1):
        assert f"Body text {i}" in p.text


def test_line_on_half_the_pages_is_kept() -> None:
    texts = ["Eligibility\nbody a", "Eligibility\nbody b", "Benefits\nbody c", "Docs\nbody d"]
    assert find_repeated_edge_lines(texts) == set()  # 2/4 is not > 50%


def test_only_own_page_number_is_masked() -> None:
    # Footers differ only by the page number; numbered body lines differ otherwise.
    texts = [
        f"Head\nSl. No. {10 + i} widow pension\nbody\nmore\nnote {i * 7}\nKerala SWD - {i}"
        for i in range(1, 5)
    ]
    keys = find_repeated_edge_lines(texts)
    assert keys == {"head", "kerala swd - #"}


def test_no_repeat_detection_for_short_docs() -> None:
    assert find_repeated_edge_lines(["Header\nx", "Header\ny"]) == set()


def test_clean_doc_updates_lang() -> None:
    page = make_page("ലൈഫ് മിഷൻ ഭവന പദ്ധതി - അർഹത മാനദണ്ഡങ്ങൾ")
    (cleaned,) = clean_doc([page])
    assert cleaned.lang_detected == "ml"
    assert cleaned.ml_ratio > 0.9


def test_toc_dot_leaders_collapsed() -> None:
    text = (
        "Preface . . . . . . . . . . . .. . . . xvii\n"
        "Section 5. State level Administration. .. . . . 29"
    )
    assert collapse_whitespace(text) == "Preface … xvii\nSection 5. State level Administration … 29"
