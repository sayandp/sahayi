"""Offline tests for Phase 1c: page ranges, OCR preprocessing, recovery, page reviews."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pymupdf
import pytest

from ingest import recover
from ingest.chunk import chunk
from ingest.common import load_page_reviews, parse_pages
from ingest.models import Page, Source, SourceUrl
from ingest.ocr_pre import adaptive_threshold, estimate_skew, ink_fraction, remove_rules
from ingest.report import exclusion_reason


def make_page(text: str, page: int = 1, **update: object) -> Page:
    p = Page(
        doc_id="obc--doc-abc123",
        scheme_id="obc",
        scheme="OBC",
        department="BCDD",
        level="state",
        source_url="https://egrantz.kerala.gov.in/x.pdf",
        downloaded_at="2026-10-12",
        page=page,
        text=text,
        method="ocr",
        lang_detected="ml",
        ml_ratio=0.9,
    )
    return p.model_copy(update=update)


def test_parse_pages() -> None:
    assert parse_pages(None) is None
    assert parse_pages("6-9,16-18, 25") == {6, 7, 8, 9, 16, 17, 18, 25}


# ------------------------------------------------------------- preprocessing


def test_adaptive_threshold_handles_uneven_lighting() -> None:
    gray = np.tile(np.linspace(120, 250, 200, dtype=np.uint8), (100, 1))  # gradient
    gray[40:60, 50:150] -= 80  # a dark "text" block on the gradient
    binary = adaptive_threshold(gray)
    assert binary[50, 60] == 0 and binary[50, 140] == 0  # text found at both ends
    assert binary[10, 10] == 255 and binary[10, 190] == 255  # background stays white


def test_remove_rules_keeps_text() -> None:
    img = np.full((100, 200), 255, dtype=np.uint8)
    img[20, :] = 0  # horizontal table rule
    img[:, 30] = 0  # vertical table rule
    img[50:55, 80:90] = 0  # a glyph
    out = remove_rules(img)
    assert (out[20, :] == 255).all() and (out[:, 30] == 255).all()
    assert (out[50:55, 80:90] == 0).all()


def test_estimate_skew_finds_rotation() -> None:
    from PIL import Image

    img = np.full((800, 800), 255, dtype=np.uint8)
    for y in range(100, 700, 40):
        img[y : y + 8, 100:700] = 0  # text-line-like bars
    rotated = np.asarray(Image.fromarray(img).rotate(-2, fillcolor=255))
    assert estimate_skew(rotated) == pytest.approx(2, abs=0.5)
    assert estimate_skew(img) == pytest.approx(0, abs=0.3)


def test_ink_fraction_blank() -> None:
    assert ink_fraction(np.full((50, 50), 255, dtype=np.uint8)) == 0.0


# ------------------------------------------------------------------ recovery


def _pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


def test_matches_requires_expected_text() -> None:
    pdf = _pdf_bytes("Framework for Implementation of Pradhan Mantri Awaas Yojana")
    assert recover.matches(pdf, "Pradhan Mantri Awaas Yojana")[0]
    ok, why = recover.matches(pdf, "Schemes of NSAP")
    assert not ok and "do not contain" in why
    assert not recover.matches(b"\x89PNG....", None)[0]


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        for prefix, resp in routes.items():
            if str(request.url).startswith(prefix):
                return resp
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_recover_entry_falls_back_to_mirror(tmp_path: Path) -> None:
    pdf = _pdf_bytes("Salient Features of Schemes of NSAP")
    client = _client(
        {
            "https://cloudflare-dns.com": httpx.Response(200, json={"Status": 3}),
            "https://dns.google": httpx.Response(200, json={"Status": 3}),
            "https://nsap.dord.gov.in/": httpx.Response(200, content=pdf),
        }
    )
    entry = SourceUrl(
        url="https://nsap.nic.in/Guidelines/g.pdf",
        manual=True,
        expect="Schemes of NSAP",
        mirrors=["https://nsap.dord.gov.in/Guidelines/g.pdf"],
    )
    src = Source(id="nsap", scheme="NSAP", department="MoRD", level="central", lang="en",
                 urls=[entry])  # fmt: skip
    attempts = recover.recover_entry(src, entry, client, tmp_path)
    assert [(a.step, a.ok) for a in attempts] == [("doh", False), ("mirror", True)]
    assert "NXDOMAIN" in attempts[0].detail
    folder = tmp_path / "nsap"
    assert (folder / "g.pdf").read_bytes() == pdf
    assert (folder / "g.pdf.url.txt").read_text().strip() == entry.url
    assert (folder / "g.pdf.fetched_from.txt").read_text().strip() == entry.mirrors[0]
    # second run: nothing fetched again
    assert recover.recover_entry(src, entry, client, tmp_path)[0].step == "existing"


def test_recover_entry_rejects_wrong_document(tmp_path: Path) -> None:
    client = _client(
        {
            "https://cloudflare-dns.com": httpx.Response(200, json={"Status": 3}),
            "https://dns.google": httpx.Response(200, json={"Status": 3}),
            "https://mirror.gov.in/": httpx.Response(200, content=_pdf_bytes("Tender notice")),
        }
    )
    entry = SourceUrl(url="https://x.nic.in/g.pdf", manual=True, expect="Schemes of NSAP",
                      mirrors=["https://mirror.gov.in/g.pdf"])  # fmt: skip
    src = Source(id="nsap", scheme="NSAP", department="MoRD", level="central", lang="en",
                 urls=[entry])  # fmt: skip
    attempts = recover.recover_entry(src, entry, client, tmp_path)
    assert not any(a.ok for a in attempts)
    assert not (tmp_path / "nsap" / "g.pdf").exists()


# --------------------------------------------------------------- page reviews


def test_reviewed_table_and_blank_pages_in_chunks() -> None:
    table = make_page("177 കോമരൻ\n178 കോടയ\n179 കോനാർ", page=14, section_hint="table")
    blank = make_page("", page=13, excluded="blank page (reviewed)")
    prose = make_page("അർഹത മാനദണ്ഡങ്ങൾ\n1. അപേക്ഷകർ കേരളീയരായിരിക്കണം.", page=2)
    chunks = chunk([table, blank, prose], "section_aware")
    assert {(c.page, c.section) for c in chunks} == {(2, "eligibility"), (14, "table")}
    assert exclusion_reason(blank) == "blank page (reviewed)"
    assert exclusion_reason(table) is None


def test_page_review_file_loads() -> None:
    reviews = load_page_reviews()
    classes = {r.page_class for r in reviews.values()}
    assert classes <= {"table", "stamp", "blank", "bad_prose"}
    assert len(reviews) == 7
