"""Recover documents whose official host is unreachable from the build network.

For every sources.yaml URL marked ``manual: true`` that has no file yet in
data/raw/manual/{scheme_id}/, try in order:

1. DNS-over-HTTPS (Cloudflare, Google) to resolve the host, then fetch by IP with
   the Host header set (helps when only the local resolver is broken);
2. the official mirrors / successor hosts listed under ``mirrors``;
3. the Wayback Machine's raw ``id_`` snapshot of the URL.

A candidate is accepted only if it is a PDF/HTML whose first pages contain the
entry's ``expect`` text (e.g. the scheme name). Accepted files are saved as
``{file}`` + ``{file}.url.txt`` (the ORIGINAL official URL) +
``{file}.fetched_from.txt`` (the URL actually fetched). Every attempt is logged to
data/raw/manual/recovery_log.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

import httpx
import pymupdf

from ingest.common import MANUAL_DIR
from ingest.models import Source, SourceUrl

DOH_RESOLVERS = (
    "https://cloudflare-dns.com/dns-query",
    "https://dns.google/resolve",
)
WAYBACK = ("https://web.archive.org/web/2024id_/{url}", "https://web.archive.org/web/id_/{url}")
FETCHED_FROM_SUFFIX = ".fetched_from.txt"
URL_SUFFIX = ".url.txt"
LOG_NAME = "recovery_log.jsonl"
CHECK_PAGES = 5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass
class Attempt:
    scheme_id: str
    url: str
    step: str  # doh | mirror | wayback | existing
    target: str
    ok: bool
    detail: str
    at: str


def target_name(url: str) -> str:
    return PurePosixPath(unquote(urlparse(url).path)).name or "document.pdf"


def doc_text(content: bytes, pages: int = CHECK_PAGES) -> str | None:
    """Text of the first pages of a PDF (or the HTML source); None if neither."""
    if content[:5] == b"%PDF-":
        try:
            with pymupdf.open(stream=content, filetype="pdf") as doc:
                title = doc.metadata.get("title") or ""
                body = " ".join(doc[i].get_text() for i in range(min(pages, doc.page_count)))
                return f"{title} {body}"
        except Exception:  # noqa: BLE001 - corrupt PDF counts as not matching
            return None
    head = content[:2048].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")):
        return content.decode("utf-8", errors="replace")
    return None


def matches(content: bytes, expect: str | None) -> tuple[bool, str]:
    text = doc_text(content)
    if text is None:
        return False, "not a PDF/HTML document"
    if expect and expect.lower() not in " ".join(text.split()).lower():
        return False, f"first pages do not contain {expect!r}"
    return True, "matched" if expect else "accepted (no expect text configured)"


def doh_resolve(host: str, client: httpx.Client) -> tuple[list[str], str]:
    """A records via DoH; returns (ips, detail)."""
    details = []
    for resolver in DOH_RESOLVERS:
        try:
            r = client.get(
                resolver,
                params={"name": host, "type": "A"},
                headers={"accept": "application/dns-json"},
            )
            data = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            details.append(f"{urlparse(resolver).netloc}: {type(exc).__name__}")
            continue
        ips = [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]
        if ips:
            return ips, f"{urlparse(resolver).netloc}: {ips}"
        status = {3: "NXDOMAIN"}.get(data.get("Status"), f"status {data.get('Status')}")
        details.append(f"{urlparse(resolver).netloc}: {status}")
    return [], "; ".join(details)


def _get(client: httpx.Client, url: str, **kw: object) -> tuple[bytes | None, str]:
    try:
        r = client.get(url, **kw)
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    return r.content, f"HTTP {r.status_code}, {len(r.content)} bytes"


def candidates(entry: SourceUrl, client: httpx.Client) -> list[tuple[str, str, dict, str]]:
    """(step, fetch_url, extra request kwargs, pre-detail) in the order to try them."""
    parsed = urlparse(entry.url)
    out: list[tuple[str, str, dict, str]] = []
    ips, detail = doh_resolve(parsed.hostname or "", client)
    if ips:
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        by_ip = f"{parsed.scheme}://{ips[0]}{path}"
        out.append(("doh", by_ip, {"headers": {"Host": parsed.netloc}}, detail))
    else:
        out.append(("doh", "", {}, detail))
    out += [("mirror", m, {}, "") for m in entry.mirrors]
    out += [("wayback", w.format(url=quote(entry.url, safe=":/?=&")), {}, "") for w in WAYBACK]
    return out


def recover_entry(
    source: Source, entry: SourceUrl, client: httpx.Client, manual_dir: Path = MANUAL_DIR
) -> list[Attempt]:
    folder = manual_dir / source.id
    name = target_name(entry.url)
    now = datetime.now().isoformat(timespec="seconds")

    def log(step: str, target: str, ok: bool, detail: str) -> Attempt:
        return Attempt(source.id, entry.url, step, target, ok, detail, now)

    if (folder / name).exists():
        return [log("existing", str(folder / name), True, "file already present")]
    attempts: list[Attempt] = []
    for step, fetch_url, kwargs, pre in candidates(entry, client):
        if not fetch_url:  # DoH gave no address
            attempts.append(log(step, "", False, pre))
            continue
        content, detail = _get(client, fetch_url, **kwargs)
        if content is None:
            attempts.append(log(step, fetch_url, False, f"{pre} {detail}".strip()))
            continue
        ok, why = matches(content, entry.expect)
        attempts.append(log(step, fetch_url, ok, f"{pre} {detail}; {why}".strip()))
        if ok:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(content)
            (folder / f"{name}{URL_SUFFIX}").write_text(entry.url + "\n", encoding="utf-8")
            (folder / f"{name}{FETCHED_FROM_SUFFIX}").write_text(fetch_url + "\n", "utf-8")
            break
    return attempts


def recover_all(sources: list[Source], manual_dir: Path = MANUAL_DIR) -> list[Attempt]:
    prior: dict[str, list[Attempt]] = {}
    for a in read_log(manual_dir):
        prior.setdefault(a.url, []).append(a)
    attempts: list[Attempt] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True, verify=False
    ) as client:
        for source in sources:
            for entry in source.urls:
                if not entry.manual:
                    continue
                result = recover_entry(source, entry, client, manual_dir)
                if result[0].step == "existing" and prior.get(entry.url):
                    result = prior[entry.url]  # keep the history of how it was recovered
                attempts += result
    if attempts:
        manual_dir.mkdir(parents=True, exist_ok=True)
        (manual_dir / LOG_NAME).write_text(
            "".join(json.dumps(asdict(a), ensure_ascii=False) + "\n" for a in attempts),
            encoding="utf-8",
        )
    return attempts


def read_log(manual_dir: Path = MANUAL_DIR) -> list[Attempt]:
    path = manual_dir / LOG_NAME
    if not path.exists():
        return []
    return [Attempt(**json.loads(x)) for x in path.read_text("utf-8").splitlines() if x]
