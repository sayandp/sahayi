"""Offline tests for Phase 1b: myscheme sections, manual drops, OCR retry, low-conf dump."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest import extract
from ingest.chunk import chunk, split_sections, strip_markers
from ingest.download import dedupe_rows, scan_manual
from ingest.models import ManifestRow, Page, Source

MYSCHEME_HTML = """<html><head><title>Pradhan Mantri Kisan Samman Nidhi</title></head><body>
<nav><a href="/">Home</a></nav>
<div id="details" class="pt-10"><h3>Details</h3><p>Income support of Rs. 6000 per year.</p>
<p>Benefits are paid by DBT.</p></div>
<div id="benefits"><h3>Benefits</h3><p>Rs. 2000 every four months.</p></div>
<div id="eligibility"><h3>Eligibility</h3><p>All landholding farmers' families.</p></div>
<div id="application-process"><h3>Application Process</h3><p>Online via CSCs.</p></div>
<div id="documents-required"><h3>Documents Required</h3><ul><li>Aadhaar Card</li></ul></div>
<div id="sources"><h3>Sources And References</h3><a href="/g.pdf">Guidelines</a></div>
<div id="feedback">Was this helpful?</div></body></html>"""


def make_page(text: str, doc_id: str = "pm_kisan--pm-kisan-abc123", page: int = 1) -> Page:
    return Page(
        doc_id=doc_id,
        scheme_id="pm_kisan",
        scheme="PM-KISAN",
        department="DA&FW",
        level="central",
        source_url="https://www.myscheme.gov.in/schemes/pm-kisan",
        downloaded_at="2026-10-12",
        page=page,
        text=text,
        method="html",
        lang_detected="en",
        ml_ratio=0.0,
    )


# ------------------------------------------------------------------ myscheme


def test_extract_myscheme_labels_sections() -> None:
    text = extract.extract_myscheme(MYSCHEME_HTML)
    lines = text.split("\n")
    assert lines[:2] == ["[[section:details]] Pradhan Mantri Kisan Samman Nidhi", "Details"]
    markers = [ln for ln in lines if ln.startswith("[[section:")]
    assert markers == [
        "[[section:details]] Pradhan Mantri Kisan Samman Nidhi",
        "[[section:benefits]] Benefits",
        "[[section:eligibility]] Eligibility",
        "[[section:how_to_apply]] Application Process",
        "[[section:documents]] Documents Required",
    ]
    assert "Sources And References" not in text and "Was this helpful" not in text


def test_extract_myscheme_unrendered_page_is_empty() -> None:
    assert extract.extract_myscheme("<html><body>Something went wrong</body></html>") == ""
    assert extract.is_myscheme("https://www.myscheme.gov.in/schemes/pm-kisan")


def test_chunk_sections_from_markers() -> None:
    page = make_page(extract.extract_myscheme(MYSCHEME_HTML))
    chunks = chunk([page], "section_aware")
    assert [c.section for c in chunks] == [
        "details",
        "benefits",
        "eligibility",
        "how_to_apply",
        "documents",
    ]
    assert all("[[section:" not in c.text for c in chunks)
    # "Benefits are paid by DBT." inside Details must not open a benefits section.
    assert "Benefits are paid by DBT." in chunks[0].text
    assert chunks[0].text.startswith("Pradhan Mantri Kisan Samman Nidhi\nDetails")


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_markers_stripped_for_other_strategies(strategy: str) -> None:
    page = make_page(extract.extract_myscheme(MYSCHEME_HTML))
    assert all("[[section:" not in c.text for c in chunk([page], strategy))


def test_split_sections_explicit_markers_ignore_keyword_headings() -> None:
    text = "[[section:faq]] Frequently Asked Questions\nEligibility\nWho can apply?"
    pieces, last = split_sections(text)
    assert [label for _, label in pieces] == ["faq"]
    assert last == "faq"
    assert strip_markers("[[section:faq]] FAQ") == "FAQ"


# ---------------------------------------------------------------- manual drop


@pytest.fixture
def sources() -> dict[str, Source]:
    src = Source(
        id="nsap",
        scheme="NSAP",
        department="MoRD",
        level="central",
        lang="en",
        urls=[{"url": "https://nsap.nic.in/g.pdf", "manual": True}],
    )
    return {"nsap": src}


def test_scan_manual_reads_sidecar(tmp_path: Path, sources: dict[str, Source]) -> None:
    folder = tmp_path / "nsap"
    folder.mkdir()
    (folder / "g.pdf").write_bytes(b"%PDF-1.7 test")
    (folder / "g.pdf.url.txt").write_text("https://nsap.nic.in/g.pdf\n", encoding="utf-8")
    (row,) = scan_manual(sources, tmp_path)
    assert row.status == "ok" and row.method == "manual" and row.kind == "pdf"
    assert row.source_url == "https://nsap.nic.in/g.pdf"
    assert row.downloaded_at and row.sha256


def test_scan_manual_reports_problems(tmp_path: Path, sources: dict[str, Source]) -> None:
    (tmp_path / "nsap").mkdir()
    (tmp_path / "nsap" / "x.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "unknown").mkdir()
    (tmp_path / "unknown" / "y.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "unknown" / "y.pdf.url.txt").write_text("https://a.in/y.pdf", encoding="utf-8")
    errors = {r.scheme_id: r.error for r in scan_manual(sources, tmp_path)}
    assert "missing sidecar" in errors["nsap"]
    assert "unknown scheme_id" in errors["unknown"]


def test_dedupe_prefers_manual_ok_over_failed_download() -> None:
    base = {"doc_id": "nsap--g-1", "scheme_id": "nsap", "source_url": "u"}
    failed = ManifestRow(**base, status="failed", error="DNS", method="http")
    manual = ManifestRow(**base, status="ok", method="manual", kind="pdf", path="p")
    assert dedupe_rows([manual, failed]) == [manual]
    assert dedupe_rows([failed, manual]) == [manual]


# ------------------------------------------------------------------- OCR retry


class FakeCache:
    def __init__(self, results: dict[tuple[int, int | None], tuple[str, float | None]]) -> None:
        self.results = results
        self.calls: list[tuple[int, int | None]] = []

    def get(self, page: object, page_no: int, dpi: int, psm: int | None) -> extract.OcrResult:
        self.calls.append((dpi, psm))
        text, conf = self.results[(dpi, psm)]
        return text, conf, extract.settings_label(dpi, psm)


def test_ocr_best_skips_retry_when_confident() -> None:
    cache = FakeCache({(300, None): ("good", 85.0)})
    assert extract.ocr_best(None, 1, cache) == ("good", 85.0, "300dpi/psm3")
    assert cache.calls == [(300, None)]


def test_ocr_best_retries_low_confidence_and_keeps_better() -> None:
    cache = FakeCache({(300, None): ("weak", 41.0), (400, 6): ("better", 66.5)})
    assert extract.ocr_best(None, 1, cache) == ("better", 66.5, "400dpi/psm6")
    cache = FakeCache({(300, None): ("weak", 41.0), (400, 6): ("worse", 30.0)})
    assert extract.ocr_best(None, 1, cache) == ("weak", 41.0, "300dpi/psm3")


def test_write_low_conf(tmp_path: Path) -> None:
    pages = [
        make_page("ok text").model_copy(update={"method": "ocr", "ocr_conf": 80.0}),
        make_page("bad text", page=2).model_copy(
            update={"method": "ocr", "ocr_conf": 35.2, "ocr_settings": "400dpi/psm6"}
        ),
    ]
    assert extract.write_low_conf(pages, tmp_path / "low") == 1
    (f,) = (tmp_path / "low").iterdir()
    assert f.name == "pm_kisan--pm-kisan-abc123_p002.txt"
    content = f.read_text(encoding="utf-8")
    assert "ocr_conf: 35.2" in content and content.endswith("bad text")


# ------------------------------------------------------------- page guards

RANK_LIST = "\n".join(
    [
        "EEP- Assistance for Competitive Exam Coaching",
        "2 PARVATHY M 94.7 48000 10000 1000 9 9000 19000 1900 17100",
        "20 ABILSHA A M 86.8 84000 10000 1000 6 6000 16000 1600 14400",
        "21 VINAYA C T K 86.2 52000 10000 1000 8 8000 18000 1800 16200",
        "44 BINCY P T 84.33 54000 10000 1000 10 10000 20000 2000 18000",
        "67 BINCY 81.94 96000 10000 1000 10 10000 20000 2000 18000",
    ]
)
CASTE_LIST = "\n".join(["Sl No Caste Name", "40 Iruva", "41 Ishanattu Kallar", "45 Izhava"] * 3)


def test_personal_data_list_detected_and_not_chunked() -> None:
    assert extract.is_personal_data_list(RANK_LIST)
    assert not extract.is_personal_data_list(CASTE_LIST)
    page = make_page(RANK_LIST).model_copy(update={"personal_data": True})
    assert chunk([page, make_page(CASTE_LIST, page=2)], "section_aware")[0].page == 2
    assert all(c.page == 2 for c in chunk([page, make_page(CASTE_LIST, page=2)], "fixed"))


def test_private_use_glyphs_trigger_ocr() -> None:
    # egrantz notice: English words mixed with font glyph IDs in the Private Use Area.
    glyphs = "\ue017്\ue00b\ue062\ue029ം\ue063\ue03bർ\ue014"
    junk = f"One Time Registration {glyphs} email id/aadhaar no " * 3
    assert extract.text_layer_problem(junk) == "private_use_glyphs"


def test_egrantz_shortlist_rows_are_personal_data() -> None:
    # Regression: shortlists put an application ID before mixed-case names.
    one_row = "147 OEEP224090016537 BISMI S SERVICE ACADEMY 65000 7/8/2024 25000 22500 KOLLAM"
    assert extract.is_personal_data_list(one_row)
    two_ids = "Applicant ADV0124070000011 ... Applicant ADV0124070000016 ..."
    assert extract.is_personal_data_list(two_ids)
    notice = "Apply on www.egrantz.kerala.gov.in before 23/12/2023. Helpline 0471 2727379."
    assert not extract.is_personal_data_list(notice)


def test_download_row_carries_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: `pages:` was only applied to manual drops, not downloaded URLs.
    from ingest import download

    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download, "ROOT", tmp_path)
    src = Source(id="adv", scheme="A", department="D", level="state", lang="ml",
                 urls=[{"url": "https://e.in/a.pdf", "pages": "1-4"}])  # fmt: skip
    folder = tmp_path / "adv"
    folder.mkdir()
    (folder / f"{download.doc_id_for('adv', 'https://e.in/a.pdf')}.pdf").write_bytes(b"%PDF-1.7")
    row = download.download_one(None, None, src, src.urls[0], {})
    assert row.status == "ok" and row.pages == "1-4"
