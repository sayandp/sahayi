"""Extract per-page text from downloaded documents.

PDF pages go through PyMuPDF. A page whose text layer is missing or unusable (see
``text_layer_problem``) is rendered with PyMuPDF at 300 dpi and OCR'd with
Tesseract ``mal+eng``; no pdf2image/Poppler is needed.

Tesseract is located via TESSERACT_CMD (env or .env, read by backend/app/config.py),
falling back to ``tesseract`` on PATH. The run fails at startup if Tesseract or its
``mal``/``eng`` data is missing; pass ``--no-ocr`` to run without OCR, in which case
those pages are kept with method ``ocr_unavailable`` and excluded from chunking.

HTML documents become a single page holding the main text (navigation and other
link-dense blocks are dropped).

Writes data/processed/pages.jsonl and data/processed/extract_errors.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import pytesseract
import regex as re
from tqdm import tqdm

from backend.app.config import get_settings
from ingest.common import (
    EXTRACT_ERRORS,
    LOW_CONF_DIR,
    MANIFEST,
    OCR_CACHE_DIR,
    PAGES_RAW,
    REOCR_DIR,
    ROOT,
    detect_lang,
    is_myscheme,
    load_page_reviews,
    load_sources,
    ml_ratio,
    parse_pages,
    read_jsonl,
    write_jsonl,
)
from ingest.models import DocMeta, ExtractMethod, ManifestRow, Page, PageReview, Source
from ingest.ocr_pre import ink_fraction, preprocess

MIN_TEXT_CHARS = 50
OCR_DPI = 300
OCR_LANG = "mal+eng"
REQUIRED_LANGS = ("mal", "eng")


# ----------------------------------------------------------------------------- OCR


class OcrSetupError(RuntimeError):
    pass


def resolve_tesseract_cmd(configured: str | None) -> str | None:
    """TESSERACT_CMD if set, else `tesseract` on PATH, else None."""
    return configured.strip() if configured and configured.strip() else shutil.which("tesseract")


def configure_tesseract(configured: str | None = None) -> str:
    """Point pytesseract at the binary; must run before any OCR call."""
    if configured is None:
        configured = get_settings().tesseract_cmd
    cmd = resolve_tesseract_cmd(configured)
    if cmd is None:
        raise OcrSetupError(
            "Tesseract not found: set TESSERACT_CMD in .env to the full path of "
            "tesseract.exe, or add tesseract to PATH (or run with --no-ocr)."
        )
    if not Path(cmd).exists() and shutil.which(cmd) is None:
        raise OcrSetupError(f"TESSERACT_CMD points to a missing file: {cmd}")
    pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def check_ocr(configured: str | None = None) -> list[str]:
    """Startup check: configure Tesseract, print its languages, require mal + eng."""
    cmd = configure_tesseract(configured)
    try:
        langs = sorted(pytesseract.get_languages(config=""))
    except Exception as exc:  # noqa: BLE001 - surface any launch failure clearly
        raise OcrSetupError(f"Tesseract at {cmd} failed to run: {exc}") from exc
    print(f"Tesseract: {cmd}")
    print(f"Tesseract languages ({len(langs)}): {', '.join(langs)}")
    missing = [lang for lang in REQUIRED_LANGS if lang not in langs]
    if missing:
        raise OcrSetupError(
            f"Tesseract language data missing: {missing}. Download mal.traineddata from "
            "https://github.com/tesseract-ocr/tessdata into the tessdata folder next to "
            f"{cmd}."
        )
    return langs


def render_gray(page: pymupdf.Page, dpi: int):  # noqa: ANN201 - PIL.Image
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def ocr_page(
    page: pymupdf.Page, dpi: int = OCR_DPI, psm: int | None = None, pre: bool = False
) -> tuple[str, float | None]:
    """OCR one rendered page; returns (text, mean word confidence 0-100).

    Call ``configure_tesseract`` (or ``check_ocr``) first.
    """
    img = render_gray(page, dpi)
    if pre:
        import numpy as np

        img = preprocess(np.asarray(img))
    config = f"--psm {psm}" if psm is not None else ""
    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, config=config, output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    text_lines: list[str] = []
    prev_par: tuple[int, int] | None = None
    for (block, par, _), words in lines.items():
        if prev_par is not None and (block, par) != prev_par:
            text_lines.append("")  # paragraph break
        text_lines.append(" ".join(words))
        prev_par = (block, par)
    return "\n".join(text_lines), (sum(confs) / len(confs) if confs else None)


# ----------------------------------------------------------------------------- PDF


@dataclass(frozen=True)
class RawPage:
    page: int
    text: str
    method: ExtractMethod
    ocr_conf: float | None = None
    quality_flag: str | None = None
    ocr_settings: str | None = None
    layer_garble: float | None = None
    excluded: str | None = None
    section_hint: str | None = None


LOW_CONF = 60.0
DEFAULT_SETTINGS = (OCR_DPI, None)  # Tesseract default page segmentation (psm 3)
RETRY_SETTINGS = (400, 6)  # higher resolution, single uniform block of text
# Reviewed tables / bad prose: preprocessed image (deskew, adaptive threshold, rules
# removed) at 400 dpi with psm 4 (single column, variable sizes) and psm 6.
PREPROCESSED_SETTINGS = ((400, 4), (400, 6))
BLANK_INK = 0.002  # share of dark pixels below which a page is blank
RETRY_CLASSES = {"table", "bad_prose"}
EXCLUDE_CLASSES = {"blank": "blank page (reviewed)", "stamp": "stamp/signature/letterhead"}


def settings_label(dpi: int, psm: int | None, pre: bool = False) -> str:
    return f"{dpi}dpi/{'pre/' if pre else ''}psm{psm if psm is not None else 3}"


OcrResult = tuple[str, float | None, str]  # (text, mean confidence, settings label)


def better(a: OcrResult, b: OcrResult) -> OcrResult:
    """Keep the attempt with the higher confidence (a wins ties and unknowns)."""
    return b if (b[1] or -1.0) > (a[1] or -1.0) else a


class OcrCache:
    """OCR results keyed by (pdf sha256, page, settings) so re-runs skip Tesseract."""

    def __init__(self, directory: Path, sha256: str | None) -> None:
        self.path = directory / f"{sha256}.json" if sha256 else None
        self.data: dict[str, list] = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(
        self, page: pymupdf.Page, page_no: int, dpi: int, psm: int | None, pre: bool = False
    ) -> OcrResult:
        label = settings_label(dpi, psm, pre)
        key = f"{page_no}:{label}"
        if key not in self.data:
            text, conf = ocr_page(page, dpi, psm, pre)
            self.data[key] = [text, conf]
        text, conf = self.data[key]
        return text, conf, label

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


def ocr_best(
    page: pymupdf.Page, page_no: int, cache: OcrCache, review: str | None = None
) -> OcrResult:
    """OCR at 300 dpi; if confidence < LOW_CONF retry at 400 dpi --psm 6.

    Pages reviewed as ``table`` or ``bad_prose`` also get preprocessed attempts
    (see ingest/ocr_pre.py). The highest-confidence result is kept.
    """
    best = cache.get(page, page_no, *DEFAULT_SETTINGS)
    if best[1] is not None and best[1] >= LOW_CONF and review not in RETRY_CLASSES:
        return best
    best = better(best, cache.get(page, page_no, *RETRY_SETTINGS))
    if review in RETRY_CLASSES:
        for dpi, psm in PREPROCESSED_SETTINGS:
            best = better(best, cache.get(page, page_no, dpi, psm, pre=True))
    return best


def is_blank(page: pymupdf.Page) -> bool:
    import numpy as np

    return ink_fraction(np.asarray(render_gray(page, 72))) < BLANK_INK


# Text-layer quality. Many Kerala GO PDFs embed Malayalam with a broken font
# mapping: vowel signs come out in visual order (a word starting with "േ") and
# conjuncts as Latin/Greek/PUA junk ("രജിേΒഷȟϗ"). Scans sometimes carry an English
# OCR layer over Malayalam text ("To600ual% aôa1"). Both pass a length check but
# are useless for retrieval, so such pages are re-OCR'd.
BROKEN_ML_THRESHOLD = 0.2
GARBLED_THRESHOLD = 0.25
MIN_TOKENS_TO_JUDGE = 20
_EDGE_PUNCT = "\"'“”‘’()[]{}<>.,:;!?-–—/|*•"
_ML = re.compile(r"[ഀ-ൿ]")
_STARTS_WITH_MARK = re.compile(r"[ഀ-ഃാ-്ൗൢൣ]")
_FOREIGN_IN_ML = re.compile(r"[^ഀ-ൿ‌‍0-9.,()\-/]")
_GARBLED = re.compile(r"[a-z][A-Z]|[A-Za-z]\d+[A-Za-z]|[%»«ðô©®¤§�]")
# Private Use Area code points: glyph IDs leaked from fonts without a ToUnicode map.
_PUA = re.compile("[-]")
PUA_THRESHOLD = 0.02
# A row of a named beneficiary / rank list: serial, UPPERCASE name, >= 3 numbers
# ("20 ABILSHA A M 86.8 84000 10000 1000 ..."). Caste or course annexures
# ("45 Izhava") do not match. Such pages are personal data, never chunked.
_NAME_ROW = re.compile(
    r"^[ \t]*\d{1,4}[ \t]+[A-Z][A-Z.]+(?:[ \t]+[A-Z][A-Z.]*){0,4}[ \t]+[\d.]+(?:[ \t]+[\d.]+){2,}",
    re.M,
)
MIN_NAME_ROWS = 5
# Application / registration numbers in e-Grantz shortlists ("OEEP224090019039").
_APPLICATION_ID = re.compile(r"\b[A-Z]{2,6}\d{9,}\b")
MIN_APPLICATION_IDS = 2
# "147 OEEP224090016537 BISMI S ..." - a single such row is already a named person.
_ID_NAME_ROW = re.compile(r"^[ \t]*\d{1,4}[ \t]+[A-Z]{2,6}\d{9,}[ \t]+[A-Za-z]", re.M)
_URLISH = re.compile(r"@|www\.|https?:|\.(?:in|com|org)\b", re.I)


def needs_ocr(text: str, min_chars: int = MIN_TEXT_CHARS) -> bool:
    return len("".join(text.split())) < min_chars


def broken_ml_ratio(text: str) -> tuple[float, int]:
    """(share of Malayalam tokens that cannot occur in correct Unicode text, n tokens)."""
    words = [w.strip(_EDGE_PUNCT) for w in text.split()]
    ml_words = [w for w in words if w and _ML.search(w)]
    bad = sum(1 for w in ml_words if _STARTS_WITH_MARK.match(w) or _FOREIGN_IN_ML.search(w))
    return (bad / len(ml_words) if ml_words else 0.0), len(ml_words)


def garbled_ratio(text: str) -> tuple[float, int]:
    """(share of Latin tokens that look like OCR junk, n tokens). URLs/emails ignored."""
    words = [w for w in text.split() if len(w) >= 3 and re.search(r"[A-Za-z]", w)]
    words = [w for w in words if not _URLISH.search(w)]
    bad = sum(1 for w in words if _GARBLED.search(w))
    return (bad / len(words) if words else 0.0), len(words)


def is_personal_data_list(text: str) -> bool:
    """Named beneficiary / rank lists: serial + NAME + numbers rows, or many app IDs."""
    return (
        len(_NAME_ROW.findall(text)) >= MIN_NAME_ROWS
        or len(set(_APPLICATION_ID.findall(text))) >= MIN_APPLICATION_IDS
        or _ID_NAME_ROW.search(text) is not None
    )


def text_layer_problem(text: str) -> str | None:
    """Reason to distrust a PDF text layer, or None if it looks usable."""
    if needs_ocr(text):
        return "too_little_text"
    compact = "".join(text.split())
    if len(_PUA.findall(compact)) > PUA_THRESHOLD * len(compact):
        return "private_use_glyphs"
    ratio, n = broken_ml_ratio(text)
    if n >= MIN_TOKENS_TO_JUDGE and ratio > BROKEN_ML_THRESHOLD:
        return "broken_malayalam_text_layer"
    ratio, n = garbled_ratio(text)
    if n >= MIN_TOKENS_TO_JUDGE and ratio > GARBLED_THRESHOLD:
        return "garbled_text_layer"
    return None


def extract_pdf(
    path: Path,
    use_ocr: bool,
    force_ocr: bool = False,
    cache: OcrCache | None = None,
    pages: set[int] | None = None,
    reviews: dict[int, PageReview] | None = None,
) -> list[RawPage]:
    """Per-page text.

    ``force_ocr`` re-OCRs pages even when their text layer passes; ``pages`` limits
    which PDF pages are ingested; ``reviews`` (page -> review) applies manual
    classifications: blank/stamp pages are excluded, tables are labelled ``table``.
    """
    cache = cache or OcrCache(OCR_CACHE_DIR, None)
    reviews = reviews or {}
    out: list[RawPage] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            if pages is not None and i not in pages:
                continue
            review = reviews[i].page_class if i in reviews else None
            if review in EXCLUDE_CLASSES:
                out.append(RawPage(i, "", "text", excluded=EXCLUDE_CLASSES[review]))
                continue
            hint = "table" if review == "table" else None
            text = page.get_text("text", sort=True)
            problem = text_layer_problem(text) or ("forced_reocr" if force_ocr else None)
            garble = round(garbled_ratio(text)[0], 3) if force_ocr else None
            if problem is None:
                out.append(RawPage(i, text, "text", section_hint=hint))
            elif problem == "too_little_text" and is_blank(page):
                out.append(RawPage(i, text, "text", excluded="blank page (no ink)"))
            elif not use_ocr:
                out.append(RawPage(i, text, "ocr_unavailable", quality_flag=problem))
            else:
                try:
                    ocr_text, conf, settings = ocr_best(page, i, cache, review)
                    out.append(
                        RawPage(i, ocr_text, "ocr", conf, problem, settings, garble, None, hint)
                    )
                except Exception:  # noqa: BLE001 - record and keep going
                    out.append(RawPage(i, text, "ocr_failed", quality_flag=problem))
    cache.save()
    return out


# ---------------------------------------------------------------------------- HTML

SKIP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "svg", "button", "select", "iframe", "head", "template",
}  # fmt: skip
BLOCK_TAGS = {
    "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "main", "table", "ul", "ol", "dd", "dt", "br",
    "blockquote", "pre", "caption", "aside",
}  # fmt: skip
CELL_TAGS = {"td", "th"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
MAX_LINK_DENSITY = 0.5


class _MainTextParser(HTMLParser):
    """Collects text blocks with their link density, skipping chrome elements.

    A table row becomes one block with cells joined by " | ", so a scheme's name,
    amount and required documents stay on the same line.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, float, bool]] = []
        self._parts: list[str] = []
        self._link_chars = 0
        self._skip = 0
        self._in_link = 0
        self._rows = 0
        self._heading = False

    def _flush(self) -> None:
        text = " ".join("".join(self._parts).split()).strip(" |")
        if text:
            self.blocks.append((text, self._link_chars / len(text), self._heading))
        self._parts, self._link_chars, self._heading = [], 0, False

    def _break(self) -> None:
        """Block boundary: a new block outside tables, a space inside a row."""
        if self._rows:
            self._parts.append(" ")
        else:
            self._flush()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "tr":
            self._flush()
            self._rows += 1
        elif tag in CELL_TAGS and self._rows:
            self._parts.append(" | ")
        elif tag in BLOCK_TAGS:
            self._break()
            self._heading = tag in HEADING_TAGS and not self._rows
        if tag == "a" and not self._skip:
            self._in_link += 1

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "tr" and self._rows:
            self._flush()
            self._rows -= 1
        elif tag in BLOCK_TAGS:
            self._break()
        if tag == "a" and self._in_link:
            self._in_link -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self._parts.append(data)
        if self._in_link:
            self._link_chars += len(" ".join(data.split()))

    def close(self) -> None:
        super().close()
        self._flush()


def extract_html(html: str) -> str:
    """Main text of an HTML page: drop chrome, link-dense blocks and repeats."""
    parser = _MainTextParser()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    lines: list[str] = []
    for text, link_density, heading in parser.blocks:
        if link_density > MAX_LINK_DENSITY or text in seen:
            continue
        seen.add(text)
        lines.extend(["", text] if heading else [text])
    return "\n".join(lines).strip()


# ------------------------------------------------------------------------ myscheme

# myscheme.gov.in scheme pages (rendered) hold each section in <div id="...">.
MYSCHEME_SECTIONS = {
    "details": "details",
    "benefits": "benefits",
    "eligibility": "eligibility",
    "exclusions": "exclusions",
    "application-process": "how_to_apply",
    "documents-required": "documents",
    "faqs": "faq",
}
MYSCHEME_STOP = {"sources", "feedback"}
_SECTION_DIV = re.compile(r'<div id="([a-z-]+)"')
_TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
# Explicit section label line prefix, understood by chunk.py (and stripped there).
SECTION_MARKER = "[[section:{label}]] "


def extract_myscheme(html: str) -> str:
    """Labelled sections of a rendered myscheme page; '' if none were rendered."""
    cuts = [
        (m.start(), m.group(1))
        for m in _SECTION_DIV.finditer(html)
        if m.group(1) in MYSCHEME_SECTIONS or m.group(1) in MYSCHEME_STOP
    ]
    parts: list[str] = []
    ends = [start for start, _ in cuts[1:]] + ([len(html)] if cuts else [])
    for (start, sid), end in zip(cuts, ends, strict=True):
        if sid not in MYSCHEME_SECTIONS:
            continue
        body = extract_html(html[start:end])
        if body:
            parts.append(SECTION_MARKER.format(label=MYSCHEME_SECTIONS[sid]) + body)
    if not parts:
        return ""
    title = _TITLE.search(html)
    name = " ".join(unescape(title.group(1)).split()) if title else ""
    if name:  # in the first section, so it does not become a tiny title-only chunk
        marker_end = parts[0].index("]] ") + 3
        parts[0] = f"{parts[0][:marker_end]}{name}\n{parts[0][marker_end:]}"
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- driver


def doc_meta(row: ManifestRow, source: Source) -> DocMeta:
    return DocMeta(
        doc_id=row.doc_id,
        scheme_id=source.id,
        scheme=source.scheme,
        department=source.department,
        level=source.level,
        source_url=row.source_url,
        downloaded_at=row.downloaded_at,
    )


def to_pages(meta: DocMeta, raw_pages: list[RawPage]) -> list[Page]:
    return [
        Page(
            **meta.model_dump(),
            page=rp.page,
            text=rp.text,
            method=rp.method,
            lang_detected=detect_lang(rp.text),
            ml_ratio=round(ml_ratio(rp.text), 4),
            ocr_conf=None if rp.ocr_conf is None else round(rp.ocr_conf, 1),
            quality_flag=rp.quality_flag,
            personal_data=is_personal_data_list(rp.text),
            excluded=rp.excluded,
            section_hint=rp.section_hint,
            ocr_settings=rp.ocr_settings,
            layer_garble=rp.layer_garble,
        )
        for rp in raw_pages
    ]


def extract_doc(
    row: ManifestRow, use_ocr: bool, reviews: dict[tuple[str, int], PageReview] | None = None
) -> list[RawPage]:
    path = ROOT / row.path
    if row.kind == "pdf":
        doc_reviews = {page: r for (doc, page), r in (reviews or {}).items() if doc == row.doc_id}
        cache = OcrCache(OCR_CACHE_DIR, row.sha256)
        return extract_pdf(path, use_ocr, row.force_ocr, cache, parse_pages(row.pages), doc_reviews)
    html = path.read_bytes().decode("utf-8", errors="replace")
    text = extract_myscheme(html) if is_myscheme(row.source_url) else extract_html(html)
    return [RawPage(1, text, "html")]


def write_low_conf(pages: list[Page], directory: Path = LOW_CONF_DIR) -> int:
    """Copy text of OCR pages with confidence < LOW_CONF for manual review."""
    shutil.rmtree(directory, ignore_errors=True)
    low = [
        p for p in pages if p.method == "ocr" and (p.ocr_conf or 0) < LOW_CONF and not p.excluded
    ]
    for p in low:
        directory.mkdir(parents=True, exist_ok=True)
        header = (
            f"doc_id: {p.doc_id}\npage: {p.page}\nocr_conf: {p.ocr_conf}\n"
            f"ocr_settings: {p.ocr_settings}\nsource_url: {p.source_url}\n\n"
        )
        (directory / f"{p.doc_id}_p{p.page:03d}.txt").write_text(header + p.text, "utf-8")
    return len(low)


def write_reocr_comparison(pages: list[Page], directory: Path = REOCR_DIR) -> int:
    """Before/after notes for documents re-OCR'd with force_ocr."""
    shutil.rmtree(directory, ignore_errors=True)
    forced = [p for p in pages if p.quality_flag == "forced_reocr" or p.layer_garble is not None]
    by_doc: dict[str, list[Page]] = {}
    for p in forced:
        by_doc.setdefault(p.doc_id, []).append(p)
    for doc_id, doc_pages in by_doc.items():
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Re-OCR: {doc_id}",
            "",
            "| page | layer garbled | OCR garbled | OCR conf | settings |",
            "|---|---|---|---|---|",
        ]
        for p in doc_pages:
            after = garbled_ratio(p.text)[0] if p.method == "ocr" else None
            lines.append(
                f"| {p.page} | {p.layer_garble} | {'' if after is None else round(after, 3)} "
                f"| {p.ocr_conf} | {p.ocr_settings} |"
            )
        (directory / f"{doc_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(by_doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-ocr", action="store_true", help="skip OCR (pages needing it are excluded)"
    )
    args = ap.parse_args()
    use_ocr = not args.no_ocr
    if use_ocr:
        try:
            check_ocr()
        except OcrSetupError as exc:
            raise SystemExit(f"OCR setup error: {exc}") from exc
    else:
        print("OCR: disabled by --no-ocr")

    sources = {s.id: s for s in load_sources()}
    rows = [r for r in read_jsonl(MANIFEST, ManifestRow) if r.status == "ok"]
    reviews = load_page_reviews()

    pages: list[Page] = []
    errors: list[dict] = []
    for row in tqdm(rows, desc="extract"):
        source = sources.get(row.scheme_id)
        if source is None:
            continue  # source removed from sources.yaml
        try:
            pages.extend(to_pages(doc_meta(row, source), extract_doc(row, use_ocr, reviews)))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            errors.append({"doc_id": row.doc_id, "source_url": row.source_url, "error": repr(exc)})

    write_jsonl(PAGES_RAW, pages)
    n_low = write_low_conf(pages)
    n_forced = write_reocr_comparison(pages)
    print(f"low-confidence OCR pages: {n_low} (-> {LOW_CONF_DIR.relative_to(ROOT)})")
    print(f"forced re-OCR docs: {n_forced} (-> {REOCR_DIR.relative_to(ROOT)})")
    EXTRACT_ERRORS.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors), encoding="utf-8"
    )
    print(f"pages: {len(pages)} from {len(rows) - len(errors)} docs; {len(errors)} errors")


if __name__ == "__main__":
    main()
