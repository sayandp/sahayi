"""Pydantic models shared by the ingestion stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Lang = Literal["ml", "en"]
Level = Literal["state", "central"]
ExtractMethod = Literal["text", "ocr", "html", "ocr_unavailable", "ocr_failed"]
Strategy = Literal["fixed", "recursive", "section_aware"]
FetchMethod = Literal["http", "js", "manual"]

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class SourceUrl(BaseModel):
    url: str
    verify: bool = False
    # Rendering with headless Chromium (ingest/render.py). None -> inherit Source.render.
    render: Literal["js"] | None = None
    click: str | None = None  # CSS selector clicked after load (e.g. an ASP.NET postback)
    expect: str | None = None  # text that must be present after rendering, else fail
    select: str | None = None  # CSS selector: keep only this element's HTML
    # The host is unreachable from the build network: do not fetch; the file is
    # expected in data/raw/manual/{scheme_id}/ with a {filename}.url.txt sidecar.
    manual: bool = False
    force_ocr: bool = False  # OCR every page even if the PDF has a text layer
    pages: str | None = None  # PDF pages to ingest, 1-based, e.g. "6-9,16-18"
    # Official mirrors / successor hosts to try when the URL itself is unreachable.
    mirrors: list[str] = Field(default_factory=list)


class Source(BaseModel):
    """One entry of ingest/sources.yaml."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    scheme: str
    department: str
    level: Level
    lang: Literal["ml", "en", "both"]
    urls: list[SourceUrl] = Field(min_length=1)
    notes: str = ""
    follow_links: bool = True
    render: Literal["js"] | None = None
    # Scheme ids whose documents cover this scheme when it has no source of its own.
    covered_by: list[str] = Field(default_factory=list)

    @field_validator("urls", mode="before")
    @classmethod
    def _coerce_urls(cls, v: list) -> list:
        return [{"url": u} if isinstance(u, str) else u for u in v]


class ManifestRow(BaseModel):
    """One line of data/raw/manifest.jsonl."""

    doc_id: str
    scheme_id: str
    source_url: str
    downloaded_at: str | None = Field(default=None, pattern=DATE_PATTERN)
    sha256: str | None = None
    bytes: int | None = None
    status: Literal["ok", "failed"]
    kind: Literal["pdf", "html"] | None = None
    path: str | None = None
    parent_url: str | None = None
    tls_verified: bool = True
    error: str | None = None
    method: FetchMethod = "http"
    force_ocr: bool = False
    pages: str | None = None
    fetched_from: str | None = None  # actual URL fetched when not source_url


class DocMeta(BaseModel):
    doc_id: str
    scheme_id: str
    scheme: str
    department: str
    level: Level
    source_url: str
    downloaded_at: str = Field(pattern=DATE_PATTERN)


class Page(DocMeta):
    """Text of one PDF page (1-based) or one HTML document (page 1)."""

    page: int = Field(ge=1)
    text: str
    method: ExtractMethod
    lang_detected: Lang
    ml_ratio: float = Field(ge=0.0, le=1.0)
    ocr_conf: float | None = None
    # Why the text layer was rejected (too_little_text | broken_malayalam_text_layer |
    # garbled_text_layer); None when the PyMuPDF text was used as-is.
    quality_flag: str | None = None
    ocr_settings: str | None = None  # e.g. "300dpi/psm3" or "400dpi/psm6" (retry won)
    layer_garble: float | None = None  # garbled share of the replaced PDF text layer
    # Named beneficiary / rank list: personal data, never chunked.
    personal_data: bool = False
    excluded: str | None = None  # reason the page is never chunked
    section_hint: str | None = None  # forces the chunk section label (e.g. table)


PageClass = Literal["table", "stamp", "blank", "bad_prose"]


class PageReview(BaseModel):
    """A human/agent review of one page (ingest/page_review.yaml)."""

    doc_id: str
    page: int = Field(ge=1)
    page_class: PageClass = Field(alias="class")
    note: str = ""


class Chunk(BaseModel):
    """A retrievable unit. The first nine fields are the CLAUDE.md contract."""

    chunk_id: str
    doc_id: str
    scheme: str
    department: str
    source_url: str
    page: int = Field(ge=1)
    lang: Lang
    downloaded_at: str = Field(pattern=DATE_PATTERN)
    text: str = Field(min_length=1)
    scheme_id: str
    level: Level
    strategy: Strategy
    section: str | None = None
    n_tokens: int = Field(ge=1)
    method: ExtractMethod
