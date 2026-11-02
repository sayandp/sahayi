"""Summarise an ingestion run into data/processed/ingest_report.md."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from ingest.chunk import DEFAULT_STRATEGY, STRATEGIES
from ingest.common import (
    EXTRACT_ERRORS,
    MANIFEST,
    PAGES_CLEAN,
    PAGES_RAW,
    PROCESSED_DIR,
    REPORT_MD,
    ROOT,
    load_page_reviews,
    load_sources,
    read_jsonl,
)
from ingest.download import SIDECAR_SUFFIX, doc_id_for
from ingest.extract import garbled_ratio
from ingest.models import Chunk, ManifestRow, Page, PageReview, Source
from ingest.recover import Attempt, read_log

LOW_OCR_CONF = 60.0
SHORT_HTML_CHARS = 300
TARGET_SCHEMES = 50
TARGET_ML_SHARE = 0.35
TARGET_MAX_EXCLUDED = 0.05


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def groupby_doc(pages: list[Page]) -> dict[str, list[Page]]:
    out: dict[str, list[Page]] = defaultdict(list)
    for p in pages:
        out[p.doc_id].append(p)
    return dict(out)


def garbled_share(text: str) -> float:
    return garbled_ratio(text)[0]


def exclusion_reason(p: Page) -> str | None:
    if p.excluded:
        return p.excluded
    if p.personal_data:
        return "personal-data list (named beneficiaries)"
    if p.method in ("ocr_unavailable", "ocr_failed"):
        return f"no usable text ({p.method})"
    return None


def chunk_stats(chunks: list[Chunk]) -> dict[str, object]:
    n = len(chunks)
    ml = sum(c.lang == "ml" for c in chunks)
    toks = sorted(c.n_tokens for c in chunks)
    return {
        "chunks": n,
        "ml": ml,
        "en": n - ml,
        "ml_share": _pct(ml, n),
        "avg_tokens": f"{sum(toks) / n:.0f}" if n else "n/a",
        "median_tokens": toks[n // 2] if n else "n/a",
        "max_tokens": toks[-1] if n else "n/a",
        "lt_20_tokens": sum(t < 20 for t in toks),
    }


def build_report(
    sources: list[Source],
    manifest: list[ManifestRow],
    raw_pages: list[Page],
    clean_pages: list[Page],
    chunks_by_strategy: dict[str, list[Chunk]],
    extract_errors: list[dict],
    reviews: dict[tuple[str, int], PageReview] | None = None,
    recovery: list[Attempt] | None = None,
) -> str:
    reviews = reviews or {}
    recovery = recovery or []
    ok = [r for r in manifest if r.status == "ok"]
    failed = [r for r in manifest if r.status == "failed"]
    errored_ids = {e["doc_id"] for e in extract_errors}
    docs_with_pages = {p.doc_id for p in raw_pages}
    methods = Counter(p.method for p in raw_pages)
    n_pages = len(raw_pages)
    pdf_pages = sum(methods[m] for m in ("text", "ocr", "ocr_unavailable", "ocr_failed"))
    needs_ocr = methods["ocr"] + methods["ocr_unavailable"] + methods["ocr_failed"]
    flags = Counter(p.quality_flag for p in raw_pages if p.quality_flag)
    default_chunks = chunks_by_strategy.get(DEFAULT_STRATEGY, [])
    schemes_with_chunks = {c.scheme_id for c in default_chunks}

    n_ml = sum(c.lang == "ml" for c in default_chunks)
    ml_share = n_ml / len(default_chunks) if default_chunks else 0.0
    exclusions = Counter(exclusion_reason(p) for p in raw_pages if exclusion_reason(p))
    n_excluded = sum(exclusions.values())
    excluded = n_excluded / n_pages if n_pages else 0.0
    low_conf = [
        p
        for p in raw_pages
        if p.method == "ocr" and (p.ocr_conf or 0.0) < LOW_OCR_CONF and not p.excluded
    ]
    retried = Counter(p.ocr_settings for p in raw_pages if p.method == "ocr")

    def verdict(ok_: bool) -> str:
        return "PASS" if ok_ else "**FAIL**"

    L: list[str] = [f"# Ingest report — {date.today().isoformat()}", ""]
    L += ["## Targets", ""]
    L += _table(
        ["target", "value", "status"],
        [
            [
                f"schemes with ≥1 chunk ≥ {TARGET_SCHEMES}",
                len(schemes_with_chunks),
                verdict(len(schemes_with_chunks) >= TARGET_SCHEMES),
            ],
            [
                f"Malayalam chunks ≥ {TARGET_ML_SHARE:.0%} ({DEFAULT_STRATEGY})",
                f"{ml_share:.1%} ({n_ml}/{len(default_chunks)})",
                verdict(ml_share >= TARGET_ML_SHARE),
            ],
            [
                f"pages excluded < {TARGET_MAX_EXCLUDED:.0%}",
                f"{excluded:.1%} ({n_excluded}/{n_pages})",
                verdict(excluded < TARGET_MAX_EXCLUDED),
            ],
        ],
    )
    L += ["", "## Totals", ""]
    L += _table(
        ["metric", "value"],
        [
            ["schemes in sources.yaml", len(sources)],
            ["schemes with ≥1 chunk", len(schemes_with_chunks)],
            ["documents downloaded (ok)", len(ok)],
            [
                "  of which PDF / HTML",
                f"{sum(r.kind == 'pdf' for r in ok)} / " f"{sum(r.kind == 'html' for r in ok)}",
            ],
            *[[f"  fetched via {m}", n] for m, n in sorted(Counter(r.method for r in ok).items())],
            ["documents failed to download", len(failed)],
            ["documents failed to extract", len(errored_ids)],
            ["documents with TLS verification bypassed", sum(not r.tls_verified for r in ok)],
            ["pages (PDF pages + HTML docs)", n_pages],
            ["PDF pages", pdf_pages],
            ["PDF pages needing OCR", f"{needs_ocr} ({_pct(needs_ocr, pdf_pages)})"],
            *[[f"  reason: {k}", v] for k, v in flags.most_common()],
            ["  OCR'd", f"{methods['ocr']} ({_pct(methods['ocr'], pdf_pages)})"],
            ["  OCR unavailable (Tesseract missing)", methods["ocr_unavailable"]],
            ["  OCR failed", methods["ocr_failed"]],
            *[[f"  OCR result kept from {k}", v] for k, v in sorted(retried.items()) if k],
            [f"  OCR pages with confidence < {LOW_OCR_CONF:.0f}", len(low_conf)],
            ["pages excluded from chunking", n_excluded],
            *[[f"  {k}", v] for k, v in exclusions.most_common()],
            ["pages empty after cleaning", sum(1 for p in clean_pages if not p.text)],
        ],
    )

    L += ["", "## Chunks per strategy", ""]
    rows = []
    for s in STRATEGIES:
        st = chunk_stats(chunks_by_strategy.get(s, []))
        rows.append(
            [
                s + (" (default)" if s == DEFAULT_STRATEGY else ""),
                st["chunks"],
                f"{st['ml']} / {st['en']}",
                st["ml_share"],
                st["avg_tokens"],
                st["median_tokens"],
                st["max_tokens"],
                st["lt_20_tokens"],
            ]
        )
    L += _table(
        ["strategy", "chunks", "ML / EN", "ML share", "avg tokens", "median", "max", "<20 tok"],
        rows,
    )

    if default_chunks:
        sections = Counter(c.section or "(none)" for c in default_chunks)
        L += [
            "",
            f"Section labels ({DEFAULT_STRATEGY}): "
            + ", ".join(f"{k} {v}" for k, v in sections.most_common()),
        ]

    L += ["", "## Per scheme", ""]
    docs_by = defaultdict(list)
    for r in manifest:
        docs_by[r.scheme_id].append(r)
    pages_by = Counter(p.scheme_id for p in raw_pages)
    chunks_by = Counter(c.scheme_id for c in default_chunks)
    ml_by = Counter(c.scheme_id for c in default_chunks if c.lang == "ml")
    rows = []
    for s in sources:
        docs = docs_by.get(s.id, [])
        n_ok = sum(d.status == "ok" for d in docs)
        rows.append(
            [
                s.id,
                s.level,
                f"{n_ok}/{len(docs)}",
                pages_by[s.id],
                chunks_by[s.id],
                _pct(ml_by[s.id], chunks_by[s.id]),
                "yes" if any(u.verify for u in s.urls) else "",
            ]
        )
    L += _table(["scheme", "level", "docs ok", "pages", "chunks", "ML share", "verify"], rows)

    L += ["", "## Failed downloads", ""]
    if failed:
        L += _table(
            ["scheme", "url", "error"],
            [[r.scheme_id, r.source_url, r.error] for r in failed],
        )
    else:
        L.append("None.")

    L += ["", "## Extraction problems", ""]
    problems: list[list[object]] = [
        [e["doc_id"], "extract error", e["error"][:120]] for e in extract_errors
    ]
    for r in ok:
        if r.doc_id not in docs_with_pages and r.doc_id not in errored_ids:
            problems.append([r.doc_id, "no pages", ""])
    by_doc: dict[str, list[Page]] = defaultdict(list)
    for p in raw_pages:
        by_doc[p.doc_id].append(p)
    for doc_id, pages in sorted(by_doc.items()):
        bad = [p for p in pages if p.method in ("ocr_unavailable", "ocr_failed")]
        if bad:
            nums = [p.page for p in bad]
            reasons = ", ".join(sorted({p.quality_flag or "?" for p in bad}))
            problems.append(
                [
                    doc_id,
                    f"{len(bad)}/{len(pages)} pages need OCR, not OCR'd",
                    f"{reasons}; pages {nums[:10]}{' …' if len(nums) > 10 else ''}",
                ]
            )
        if pages[0].method == "html" and len("".join(pages[0].text.split())) < SHORT_HTML_CHARS:
            problems.append(
                [doc_id, "HTML has little main text (JS-rendered?)", f"{len(pages[0].text)} chars"]
            )
    L += _table(["doc", "problem", "detail"], problems) if problems else ["None."]

    L += ["", "## Reviewed pages (ingest/page_review.yaml)", ""]
    if reviews:
        by_key = {(p.doc_id, p.page): p for p in raw_pages}
        rows_r = []
        for (doc_id, page_no), r in sorted(reviews.items()):
            pg = by_key.get((doc_id, page_no))
            outcome = (
                "not ingested"
                if pg is None
                else pg.excluded or f"kept, section={pg.section_hint or '-'}"
            )
            conf = "" if pg is None or pg.ocr_conf is None else f"{pg.ocr_conf} ({pg.ocr_settings})"
            rows_r.append([doc_id, page_no, r.page_class, outcome, conf, r.note])
        L += _table(["doc", "page", "class", "outcome", "OCR conf (kept)", "note"], rows_r)
    else:
        L.append("None.")

    L += ["", "## Recovery of unreachable sources (ingest/recover.py)", ""]
    if recovery:
        L += _table(
            ["scheme", "step", "result", "fetched", "detail"],
            [
                [a.scheme_id, a.step, "OK" if a.ok else "failed", a.target, a.detail]
                for a in recovery
            ],
        )
    else:
        L.append("No manual sources.")

    L += ["", f"## Low-confidence OCR pages (< {LOW_OCR_CONF:.0f})", ""]
    if low_conf:
        L.append("Text copied to `data/processed/low_conf/{doc_id}_p{page}.txt` for review.")
        L.append("")
        L += _table(
            ["doc", "page", "confidence", "kept settings"],
            [[p.doc_id, p.page, p.ocr_conf, p.ocr_settings] for p in low_conf],
        )
    else:
        L.append("None.")

    forced = [p for p in raw_pages if p.layer_garble is not None]
    L += ["", "## Forced re-OCR (embedded OCR layer replaced)", ""]
    if forced:
        L.append(
            "The embedded layer has no confidence score, so it is compared by its share of "
            "garbled tokens; details in `data/processed/reocr/`."
        )
        L.append("")
        rows_f = []
        for doc_id, pages in groupby_doc(forced).items():
            before = sum(p.layer_garble or 0 for p in pages) / len(pages)
            after_pages = [p for p in pages if p.method == "ocr"]
            after = (
                sum(garbled_share(p.text) for p in after_pages) / len(after_pages)
                if after_pages
                else None
            )
            confs = [p.ocr_conf for p in after_pages if p.ocr_conf is not None]
            rows_f.append(
                [
                    doc_id,
                    len(pages),
                    f"{before:.1%}",
                    "n/a" if after is None else f"{after:.1%}",
                    f"{sum(confs) / len(confs):.0f}" if confs else "n/a",
                    f"{min(confs):.0f}" if confs else "n/a",
                ]
            )
        L += _table(
            ["doc", "pages", "garbled before", "garbled after", "mean conf after", "min conf"],
            rows_f,
        )
    else:
        L.append("None.")

    L += ["", "## Schemes with zero chunks", ""]
    zero = [s for s in sources if s.id not in schemes_with_chunks]
    if zero:
        rows_z = []
        for s in zero:
            docs = docs_by.get(s.id, [])
            errs = "; ".join(sorted({(d.error or "")[:60] for d in docs if d.status == "failed"}))
            covered = ", ".join(s.covered_by) or ""
            rows_z.append(
                [s.id, f"{sum(d.status == 'ok' for d in docs)}/{len(docs)}", covered, errs]
            )
        L += _table(["scheme", "docs ok", "covered by", "download errors"], rows_z)
    else:
        L.append("None.")

    L += ["", "## Files to drop in manually", ""]
    ok_manual = {r.doc_id for r in ok if r.method == "manual"}
    todo = []
    for s in sources:
        for u in s.urls:
            if u.manual and doc_id_for(s.id, u.url) not in ok_manual:
                name = PurePosixPath(unquote(urlparse(u.url).path)).name or "document.pdf"
                todo.append((s.id, name, u.url))
    if todo:
        L.append("Save each file as shown and put its source URL (one line) in the sidecar file.")
        L.append("")
        L += _table(
            ["file", "sidecar", "source URL"],
            [
                [f"data/raw/manual/{sid}/{name}", f"{name}{SIDECAR_SUFFIX}", url]
                for sid, name, url in todo
            ],
        )
    else:
        L.append("None outstanding.")

    verify = [(s.id, u.url) for s in sources for u in s.urls if u.verify]
    L += ["", "## URLs marked `verify: true`", ""]
    L += [f"- `{sid}` — {url}" for sid, url in verify] or ["None."]
    return "\n".join(L) + "\n"


def main() -> None:
    chunks = {s: list(read_jsonl(PROCESSED_DIR / f"chunks.{s}.jsonl", Chunk)) for s in STRATEGIES}
    errors = []
    if EXTRACT_ERRORS.exists():
        errors = [
            json.loads(x) for x in EXTRACT_ERRORS.read_text(encoding="utf-8").splitlines() if x
        ]
    report = build_report(
        load_sources(),
        list(read_jsonl(MANIFEST, ManifestRow)),
        list(read_jsonl(PAGES_RAW, Page)),
        list(read_jsonl(PAGES_CLEAN, Page)),
        chunks,
        errors,
        load_page_reviews(),
        read_log(),
    )
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"report -> {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
