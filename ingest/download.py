"""Download scheme documents listed in sources.yaml into data/raw/{scheme_id}/.

PDF URLs are saved as-is. HTML pages are saved and, unless the source sets
``follow_links: false``, the PDFs they link to are downloaded too (capped per page).
Already-downloaded files are skipped, so re-running is cheap.

Writes data/raw/manifest.jsonl with one row per document.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import time
from datetime import date
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from tqdm import tqdm

from ingest.common import (
    MANIFEST,
    MANUAL_DIR,
    MYSCHEME_REQUIRED,
    RAW_DIR,
    ROOT,
    is_myscheme,
    load_sources,
    read_jsonl,
    write_jsonl,
)
from ingest.models import ManifestRow, Source, SourceUrl
from ingest.recover import FETCHED_FROM_SUFFIX, LOG_NAME, recover_all

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 SahayiBot/0.1"
)
TIMEOUT = httpx.Timeout(60.0, connect=20.0)
RETRIES = 3
BACKOFF_S = 2.0
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BYTES = 50 * 1024 * 1024
MAX_LINKED_PDFS = 10
POLITE_DELAY_S = 0.5

PDF_HREF = re.compile(r"""href\s*=\s*["']([^"'#]+?\.pdf)(?:[?#][^"']*)?["']""", re.I)


class FetchError(Exception):
    pass


def doc_id_for(scheme_id: str, url: str) -> str:
    """Stable, filesystem- and chunk_id-safe id: '{scheme_id}--{slug}-{hash6}'."""
    parsed = urlparse(url)
    name = unquote(parsed.path).rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(r"\.(pdf|html?|aspx|php)$", "", name, flags=re.I)
    if parsed.query:
        name = f"{name}-{parsed.query}"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:40] or "index"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:6]
    return f"{scheme_id}--{slug}-{digest}"


def sniff_kind(content: bytes, content_type: str) -> str | None:
    if content[:5] == b"%PDF-":
        return "pdf"
    head = content[:512].lstrip().lower()
    if "html" in content_type.lower() or head.startswith((b"<!doctype", b"<html")):
        return "html"
    return None


def find_pdf_links(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    for href in PDF_HREF.findall(html):
        full = urljoin(base_url, unescape(href).strip())
        if full.startswith("http") and full not in links:
            links.append(full)
    return links


def _ssl_context(strict: bool) -> ssl.SSLContext | bool:
    if strict:
        return True
    # Several .gov.in / kerala.gov.in hosts use legacy TLS (weak DH, SHA-1
    # signatures) or serve incomplete chains. Integrity is still tracked by sha256.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT@SECLEVEL=0")
    return ctx


def _is_tls_error(exc: Exception) -> bool:
    msg = str(exc)
    return "SSL" in msg or "CERTIFICATE" in msg.upper()


class Fetcher:
    """httpx clients with retry + backoff and a relaxed-TLS fallback."""

    def __init__(self) -> None:
        self.clients = {
            strict: httpx.Client(
                verify=_ssl_context(strict),
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
                follow_redirects=True,
            )
            for strict in (True, False)
        }

    def close(self) -> None:
        for c in self.clients.values():
            c.close()

    def get(self, url: str) -> tuple[httpx.Response, bool]:
        """Return (response, tls_verified). Raises FetchError after retries."""
        last: Exception | None = None
        for strict in (True, False):
            for attempt in range(RETRIES):
                try:
                    resp = self.clients[strict].get(url)
                except httpx.HTTPError as exc:
                    last = exc
                    if _is_tls_error(exc):
                        break  # same TLS error on retry; try the relaxed client
                    time.sleep(BACKOFF_S * 2**attempt)
                    continue
                finally:
                    time.sleep(POLITE_DELAY_S)
                if resp.status_code in RETRY_STATUS:
                    last = FetchError(f"HTTP {resp.status_code}")
                    time.sleep(BACKOFF_S * 2**attempt)
                    continue
                if resp.status_code >= 400:
                    raise FetchError(f"HTTP {resp.status_code}")
                if len(resp.content) > MAX_BYTES:
                    raise FetchError(f"too large ({len(resp.content)} bytes)")
                return resp, strict
            else:
                break  # retries exhausted on a non-TLS error: relaxed TLS won't help
        raise FetchError(f"{type(last).__name__}: {last}".strip(": "))


def existing_file(scheme_dir: Path, doc_id: str) -> Path | None:
    for ext in ("pdf", "html"):
        p = scheme_dir / f"{doc_id}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _row_from_file(
    path: Path, source: Source, url: str, parent: str | None, prior: ManifestRow | None
) -> ManifestRow:
    data = path.read_bytes()
    downloaded_at = (
        prior.downloaded_at
        if prior and prior.downloaded_at
        else date.fromtimestamp(path.stat().st_mtime).isoformat()
    )
    return ManifestRow(
        doc_id=path.stem,
        scheme_id=source.id,
        source_url=url,
        downloaded_at=downloaded_at,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        status="ok",
        kind=path.suffix.lstrip("."),
        path=path.relative_to(ROOT).as_posix(),
        parent_url=parent,
        tls_verified=prior.tls_verified if prior else True,
    )


class LazyRenderer:
    """Starts Chromium only if a source actually needs JS rendering."""

    def __init__(self) -> None:
        self._renderer = None

    def get(self):  # noqa: ANN201 - Renderer; imported lazily since playwright is heavy
        if self._renderer is None:
            from ingest.render import Renderer

            self._renderer = Renderer().__enter__()
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.__exit__(None, None, None)


def _entry_key(entry: SourceUrl) -> str:
    """URL identity for doc_id: a click on the same page is a different document."""
    return f"{entry.url}#click={entry.click}" if entry.click else entry.url


def download_one(
    fetcher: Fetcher,
    renderer: LazyRenderer,
    source: Source,
    entry: SourceUrl,
    prior: dict[str, ManifestRow],
    parent: str | None = None,
) -> ManifestRow:
    url = entry.url
    doc_id = doc_id_for(source.id, _entry_key(entry))
    scheme_dir = RAW_DIR / source.id
    js = (entry.render or source.render) == "js"
    method = "js" if js else "http"

    def failed(error: str) -> ManifestRow:
        return ManifestRow(
            doc_id=doc_id,
            scheme_id=source.id,
            source_url=url,
            status="failed",
            parent_url=parent,
            error=error,
            method=method,
            force_ocr=entry.force_ocr,
            pages=entry.pages,
        )

    if entry.manual:
        return failed(f"manual: drop the file in data/raw/manual/{source.id}/ (see report)")

    found = existing_file(scheme_dir, doc_id)
    if found:
        row = _row_from_file(found, source, url, parent, prior.get(doc_id))
        return row.model_copy(
            update={"method": method, "force_ocr": entry.force_ocr, "pages": entry.pages}
        )

    if js:
        from ingest.render import RenderError

        try:
            html = renderer.get().render(
                url,
                click=entry.click,
                expect=entry.expect,
                select=entry.select,
                require=MYSCHEME_REQUIRED if is_myscheme(url) else None,
            )
        except RenderError as exc:
            return failed(f"render: {exc}")
        content, verified, kind = html.encode("utf-8"), True, "html"
    else:
        try:
            resp, verified = fetcher.get(url)
        except FetchError as exc:
            return failed(str(exc))
        content = resp.content
        kind = sniff_kind(content, resp.headers.get("content-type", ""))
        if kind is None:
            return failed(f"unsupported content-type {resp.headers.get('content-type')!r}")
        if urlparse(url).path.lower().endswith(".pdf") and kind != "pdf":
            return failed("expected PDF, got HTML (error or login page)")

    scheme_dir.mkdir(parents=True, exist_ok=True)
    path = scheme_dir / f"{doc_id}.{kind}"
    path.write_bytes(content)
    row = _row_from_file(path, source, url, parent, None)
    return row.model_copy(
        update={
            "downloaded_at": date.today().isoformat(),
            "tls_verified": verified,
            "method": method,
            "force_ocr": entry.force_ocr,
            "pages": entry.pages,
        }
    )


def download_source(
    fetcher: Fetcher, renderer: LazyRenderer, source: Source, prior: dict[str, ManifestRow]
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for entry in source.urls:
        row = download_one(fetcher, renderer, source, entry, prior)
        rows.append(row)
        # Rendered pages (myscheme etc.) are curated summaries; the guideline PDFs they
        # link to are often 100+ pages and would swamp the corpus, so they are not
        # followed. Add a specific PDF to sources.yaml when it is needed.
        if row.status != "ok" or row.kind != "html" or row.method == "js":
            continue
        if not source.follow_links:
            continue
        html = (ROOT / row.path).read_text(encoding="utf-8", errors="replace")
        for link in find_pdf_links(html, entry.url)[:MAX_LINKED_PDFS]:
            rows.append(
                download_one(fetcher, renderer, source, SourceUrl(url=link), prior, entry.url)
            )
    return rows


# ------------------------------------------------------------------- manual drop

SIDECAR_SUFFIX = ".url.txt"


def _repo_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else absolute."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _manual_entry(source: Source | None, url: str) -> SourceUrl | None:
    """The sources.yaml entry a manual file stands in for (for pages / force_ocr)."""
    if source is None:
        return None
    return next((u for u in source.urls if u.url == url), None)


def scan_manual(sources: dict[str, Source], manual_dir: Path = MANUAL_DIR) -> list[ManifestRow]:
    """Rows for files in data/raw/manual/{scheme_id}/, each with a {name}.url.txt sidecar."""
    rows: list[ManifestRow] = []
    if not manual_dir.exists():
        return rows
    for path in sorted(manual_dir.glob("*/*")):
        if not path.is_file() or path.name.endswith((SIDECAR_SUFFIX, FETCHED_FROM_SUFFIX)):
            continue
        if path.name == LOG_NAME:
            continue
        scheme_id = path.parent.name
        sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
        url = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
        fetched = path.with_name(path.name + FETCHED_FROM_SUFFIX)
        entry = _manual_entry(sources.get(scheme_id), url)
        base = {
            "doc_id": doc_id_for(scheme_id, url or path.name),
            "scheme_id": scheme_id,
            "source_url": url or path.name,
            "method": "manual",
            "fetched_from": fetched.read_text("utf-8").strip() if fetched.exists() else None,
            "pages": entry.pages if entry else None,
            "force_ocr": entry.force_ocr if entry else False,
        }
        error = None
        if scheme_id not in sources:
            error = f"unknown scheme_id {scheme_id!r} (not in sources.yaml)"
        elif not url:
            error = f"missing sidecar {sidecar.name} containing the source URL"
        data = path.read_bytes()
        kind = sniff_kind(data[:1024], "") if error is None else None
        if error is None and kind is None:
            error = "unsupported file type (expected PDF or HTML)"
        if error:
            rows.append(ManifestRow(**base, status="failed", error=error))
            continue
        rows.append(
            ManifestRow(
                **base,
                downloaded_at=date.fromtimestamp(path.stat().st_mtime).isoformat(),
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                status="ok",
                kind=kind,
                path=_repo_path(path),
            )
        )
    return rows


def dedupe_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """One row per doc_id, sorted for stable diffs.

    Preference: manual ok > ok > failed; among equals the later row wins.
    """

    def rank(r: ManifestRow) -> int:
        ok = r.status == "ok"
        return 2 * ok + (ok and r.method == "manual")

    merged: dict[str, ManifestRow] = {}
    for r in rows:
        if r.doc_id not in merged or rank(r) >= rank(merged[r.doc_id]):
            merged[r.doc_id] = r
    return sorted(merged.values(), key=lambda r: (r.scheme_id, r.doc_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="limit to these source ids")
    args = ap.parse_args()

    all_sources = load_sources()
    only = set(args.only or [])
    sources = [s for s in all_sources if not only or s.id in only]
    prior = {r.doc_id: r for r in read_jsonl(MANIFEST, ManifestRow)}

    fetcher, renderer = Fetcher(), LazyRenderer()
    rows: list[ManifestRow] = []
    try:
        for source in tqdm(sources, desc="sources"):
            rows.extend(download_source(fetcher, renderer, source, prior))
    finally:
        fetcher.close()
        renderer.close()

    if only:  # keep rows of sources not processed this run
        rows = [
            r for r in prior.values() if r.scheme_id not in only and r.method != "manual"
        ] + rows
    for a in recover_all(all_sources):
        print(f"recover {a.scheme_id} [{a.step}] {'OK' if a.ok else 'no'}: {a.detail[:90]}")
    rows = dedupe_rows(rows + scan_manual({s.id: s for s in all_sources}))
    write_jsonl(MANIFEST, rows)
    ok = sum(r.status == "ok" for r in rows)
    print(f"manifest: {ok} ok, {len(rows) - ok} failed -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
