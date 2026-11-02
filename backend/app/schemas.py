"""Request/response pydantic models for the API (shape documented in CLAUDE.md)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class Citation(BaseModel):
    chunk_id: str
    source_url: str
    page: int
    link: str  # source_url, with #page=N for PDFs
    snippet: str
    scheme: str | None = None
    downloaded_at: str | None = None


class SchemeCard(BaseModel):
    scheme_id: str
    name: str
    eligible: bool | None
    reasons: list[str] = []
    benefit: str | None = None
    how_to_apply: str | None = None
    documents: list[str] = []
    citations: list[str] = []  # chunk_ids


class Dropped(BaseModel):
    text: str
    reason: str
    detail: str = ""


class ValidationInfo(BaseModel):
    dropped: list[Dropped] = []
    unknown_citations: list[str] = []
    flags: list[str] = []
    kept_sentences: int = 0


class Meta(BaseModel):
    latency_ms: int
    ttfc_ms: int  # time to first content: the sources event
    ttft_ms: int
    latency: dict
    tokens_in: int
    tokens_out: int
    cost_inr: float
    model: str
    fallbacks: list[str] = []  # generators that failed before `model` answered
    cached: bool = False
    rewrite_model: str | None = None
    rerank_backend: str | None = None
    rerank_fallback: str | None = None
    top_rerank_score: float | None = None
    prompt_version: str
    prices_checked: str
    request_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    lang: Literal["ml", "en"]
    schemes: list[SchemeCard]
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None = None
    degraded: bool = False  # generators busy: answer is a notice, cards hold cited excerpts
    follow_up_question: str | None = None
    as_of: str | None
    validation: ValidationInfo
    meta: Meta


class SchemeSummary(BaseModel):
    scheme_id: str
    name: str
    department: str
    level: str
    site: str
    n_chunks: int


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    scheme: str
    department: str
    page: int
    lang: str
    section: str | None = None
    source_url: str
    link: str
    downloaded_at: str
    text: str
