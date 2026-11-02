"""Offline tests for the pure helpers in ingest/extract.py and ingest/download.py."""

from __future__ import annotations

import pytest

from ingest import extract
from ingest.common import detect_lang, ml_ratio
from ingest.download import doc_id_for, find_pdf_links, sniff_kind
from ingest.extract import extract_html, needs_ocr, text_layer_problem

HTML = """<html><head><title>t</title><style>.x{}</style></head><body>
<header><a href="/">Home</a></header>
<ul class="menu"><li><a href="/a">ഹോം</a></li><li><a href="/b">പദ്ധതികള്‍</a></li></ul>
<div id="content">
  <h2>കാരുണ്യ ആരോഗ്യ സുരക്ഷാ പദ്ധതി</h2>
  <p>കുടുംബത്തിന് പ്രതിവർഷം 5 ലക്ഷം രൂപ വരെ ചികിത്സാ സഹായം. <a href="/x">കൂടുതൽ</a></p>
  <p>Download the <a href="/docs/Guidelines%20KASP.pdf?v=2">guidelines</a> here.</p>
</div>
<script>var a = "<p>not text</p>";</script>
<footer>© 2026</footer></body></html>"""


def test_extract_html_main_text() -> None:
    text = extract_html(HTML)
    assert "കാരുണ്യ ആരോഗ്യ സുരക്ഷാ പദ്ധതി" in text
    assert "5 ലക്ഷം രൂപ" in text
    assert "ഹോം" not in text  # nav list dropped by link density
    assert "not text" not in text and "© 2026" not in text and ".x{}" not in text


def test_find_pdf_links() -> None:
    links = find_pdf_links(HTML, "https://sha.kerala.gov.in/page/")
    assert links == ["https://sha.kerala.gov.in/docs/Guidelines%20KASP.pdf"]


def test_doc_id_stable_and_safe() -> None:
    url = "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"
    a, b = doc_id_for("pm_kisan", url), doc_id_for("pm_kisan", url)
    assert a == b and a.startswith("pm_kisan--revisedpm-kisan")
    assert ":" not in a and "/" not in a
    assert doc_id_for("kasp", "https://sha.kerala.gov.in/?page_id=1077").startswith(
        "kasp--page-id-1077"
    )
    assert doc_id_for("x", "https://a.in/ലൈഫ്.pdf").startswith("x--index-")


def test_sniff_kind() -> None:
    assert sniff_kind(b"%PDF-1.7 ...", "application/octet-stream") == "pdf"
    assert sniff_kind(b"<!DOCTYPE html><html>", "") == "html"
    assert sniff_kind(b"\x89PNG", "image/png") is None


def test_lang_detection() -> None:
    assert detect_lang("വിധവ പെൻഷൻ അപേക്ഷ") == "ml"
    assert detect_lang("Widow pension application") == "en"
    assert detect_lang("PM-KISAN പദ്ധതി പ്രകാരം Rs. 6000") == "ml"
    assert ml_ratio("") == 0.0


def test_needs_ocr() -> None:
    assert needs_ocr("  \n 12 \n")
    assert not needs_ocr("Eligibility: " + "x" * 60)


# Real text layers seen in the corpus (2026-10-12).
BROKEN_ML = (  # kfwfb Registrationinstructions_KFWFB.pdf: broken font mapping
    "രജിേΒഷȟϗ നിÓേദശŅൾ രജിേΒഷȟϗ അേപý സമÓɠിÚȽതിȟ താെഴ പറ˖Ƚ േരഖകϕെട പകÓɡകളൾ "
    "ഓേരാ പകÓɡം 200 KB യിൽ താെഴ ആവശÊമാണ് വയθ് െതളിയിÚȽതിȟϗ േരഖ ആധാÓ "
    "വിേ̣ ജ് ഓഫീസറിൽ നിȽ് ലഭിÚȽ വാÓഷിക വ˹മാന സർƃിഫിÙϿ് Öഷി അȟബȷ"
)
GOOD_ML = (  # keralaagriculture.gov.in karshakapension page
    "ചെറുകിട നാമമാത്ര കർഷക പെൻഷൻ പദ്ധതി കാർഷികോൽപ്പാദനം മെച്ചപ്പെടുത്തുന്നതിനോടൊപ്പം "
    "കർഷക സുരക്ഷയും ഉറപ്പ് വരുത്തുന്നതിന്റെ ഭാഗമായി 60 വയസ്സ് തികഞ്ഞ നെൽകർഷകർക്ക് "
    "അവരുടെ ജീവിത സായാഹ്നത്തിൽ ഒരു കൈത്താങ്ങ് എന്ന നിലയിൽ ഒരു പെൻഷൻ പദ്ധതിക്ക് "
    "2008-09 വർഷം 31-01-2009-ലെ സർക്കാർ ഉത്തരവ് (പി) നമ്പർ 22/2009/കൃഷി (പി.ബി.) പ്രകാരം"
)
GARBLED = (  # SC/ST post-matric circular: Malayalam scan with an English OCR layer
    "(5) 630 To600ual% aôa1 o @jasola0now laleSMo, 75%75% an0R@Mle gmlaoj»s "
    "oslmnomaold olmaio8 modajm006Ti. (6) G0o0o Ao06maol 80) mnoDala Qið1D0o al0l44 "
    "90d aglwo0al&Ho nimuoJS mcmoolglw'lcð mlme2 awls mudslnoleoglo amslrunomomlos"
)
GOOD_EN = (  # stdd Unnathi Overseas Scholarship order
    "The scholarship shall be granted to a student for only ONE programme or course of "
    "study and shall be limited to the period for which it is sanctioned. Applications "
    "submitted after taking admission other than through the procedure laid down herein "
    "to any foreign university or institution will not be considered under any circumstances."
)


def test_text_layer_problem_detects_broken_malayalam() -> None:
    assert text_layer_problem(BROKEN_ML) == "broken_malayalam_text_layer"
    assert text_layer_problem(GOOD_ML) is None


def test_text_layer_problem_detects_garbled_ocr_layer() -> None:
    assert text_layer_problem(GARBLED) == "garbled_text_layer"
    assert text_layer_problem(GOOD_EN) is None
    assert (
        text_layer_problem(GOOD_EN + " Contact: bcddkerala@gmail.com www.bcdd.kerala.gov.in")
        is None
    )


def test_text_layer_problem_too_short() -> None:
    assert text_layer_problem("  12  ") == "too_little_text"


def test_extract_html_keeps_aspnet_form_content() -> None:
    # Regression: ASP.NET wraps the whole page in <form>; skipping <form> emptied
    # welfarepension.lsgkerala.gov.in pages.
    html = (
        '<html><body><form id="form1" runat="server"><div>'
        "<p>കർഷക തൊഴിലാളി പെൻഷൻ: 60 വയസ്സ് പൂർത്തിയായവർക്ക്.</p>"
        "</div></form></body></html>"
    )
    assert "കർഷക തൊഴിലാളി പെൻഷൻ" in extract_html(html)


def test_extract_html_table_rows_stay_together() -> None:
    # welfarepension.lsgkerala.gov.in norms table: name | amount | documents per row.
    html = (
        "<table><tr><th>ക്രമ നം</th><th>പെൻഷൻറെ പേര്</th><th>തുക</th></tr>"
        "<tr><td>1</td><td><p>കർഷക തൊഴിലാളി</p><p>പെൻഷൻ</p></td><td>1600</td></tr></table>"
    )
    assert extract_html(html).split("\n") == [
        "ക്രമ നം | പെൻഷൻറെ പേര് | തുക",
        "1 | കർഷക തൊഴിലാളി പെൻഷൻ | 1600",
    ]


def test_resolve_tesseract_cmd_prefers_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract.shutil, "which", lambda _: "/usr/bin/tesseract")
    assert extract.resolve_tesseract_cmd(r"D:\a\New folder (2)\tesseract.exe") == (
        r"D:\a\New folder (2)\tesseract.exe"
    )
    assert extract.resolve_tesseract_cmd("") == "/usr/bin/tesseract"
    assert extract.resolve_tesseract_cmd(None) == "/usr/bin/tesseract"


def test_configure_tesseract_fails_clearly_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract.shutil, "which", lambda _: None)
    with pytest.raises(extract.OcrSetupError, match="TESSERACT_CMD"):
        extract.configure_tesseract("")
    with pytest.raises(extract.OcrSetupError, match="missing file"):
        extract.configure_tesseract(r"Z:\nope\tesseract.exe")


def test_check_ocr_requires_malayalam(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    exe = tmp_path / "tesseract.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(extract.pytesseract, "get_languages", lambda config="": ["eng", "osd"])
    with pytest.raises(extract.OcrSetupError, match=r"\['mal'\]"):
        extract.check_ocr(str(exe))
    assert extract.pytesseract.pytesseract.tesseract_cmd == str(exe)
    monkeypatch.setattr(extract.pytesseract, "get_languages", lambda config="": ["eng", "mal"])
    assert extract.check_ocr(str(exe)) == ["eng", "mal"]
