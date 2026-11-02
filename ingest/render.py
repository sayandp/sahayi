"""Render JavaScript-driven pages with headless Chromium (Playwright).

Used by download.py for sources.yaml URLs with ``render: js``: myscheme.gov.in scheme
pages, the nha.gov.in / pmmsy SPAs, and ASP.NET pages whose content appears only
after a postback click (welfarepension.lsgkerala.gov.in pension norms).

Setup: ``pip install playwright`` then ``playwright install chromium``.
"""

from __future__ import annotations

import time
from types import TracebackType

from playwright.sync_api import Browser, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 SahayiBot/0.1"
)
TIMEOUT_MS = 90_000
REQUIRE_TIMEOUT_MS = 20_000
RETRIES = 3
BACKOFF_S = 15.0  # myscheme answers bursts with "Something went wrong" pages
SETTLE_MS = 1_500  # after network idle, for late client-side rendering


class RenderError(Exception):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class Renderer:
    """One browser for many pages; use as a context manager."""

    def __init__(self) -> None:
        self._pw: Playwright | None = None
        self._browser: Browser | None = None

    def __enter__(self) -> Renderer:
        from backend.app.config import configure_playwright_browsers

        configure_playwright_browsers()  # must be set before the driver starts
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def render(
        self,
        url: str,
        *,
        click: str | None = None,
        expect: str | None = None,
        select: str | None = None,
        require: str | None = None,
    ) -> str:
        """Render with retries; ``require`` is a CSS selector that must appear.

        Sites like myscheme return an error shell under load; such renders are
        retried with backoff instead of being saved as documents.
        """
        last: RenderError | None = None
        for attempt in range(RETRIES):
            try:
                return self._render_once(url, click, expect, select, require)
            except RenderError as exc:
                last = exc
                if not exc.retryable or attempt == RETRIES - 1:
                    break
                time.sleep(BACKOFF_S * (attempt + 1))
        raise last  # type: ignore[misc]

    def _render_once(
        self,
        url: str,
        click: str | None,
        expect: str | None,
        select: str | None,
        require: str | None,
    ) -> str:
        """Return the rendered HTML of ``url`` (optionally after a click / narrowed).

        If ``click`` matches several elements and ``expect`` is given, each is tried
        in turn (fresh page load per try) and the first result whose ``select``
        element contains ``expect`` is returned. This copes with pages whose
        links open the "wrong" panel or change their wiring between sessions.
        """
        if self._browser is None:
            raise RuntimeError("use Renderer as a context manager")
        page = self._browser.new_page(user_agent=USER_AGENT)
        try:
            n_targets = self._load(page, url, click)
            if require:
                try:
                    page.wait_for_selector(require, timeout=REQUIRE_TIMEOUT_MS)
                except PlaywrightError as exc:
                    raise RenderError(
                        f"required element {require!r} not rendered", retryable=True
                    ) from exc
            attempts = range(n_targets) if (click and expect) else range(1)
            for i in attempts:
                if i:
                    self._load(page, url, click)
                if click:
                    page.locator(click).nth(i).click()
                    page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
                page.wait_for_timeout(SETTLE_MS)
                node = page.locator(select or "body").first
                if node.count() == 0:
                    continue
                # Checked inside the selected element: the page around it (e.g. a grid
                # listing every scheme name) would make the check meaningless.
                if expect and expect not in node.inner_text():
                    continue
                if not select:
                    return page.content()
                return (
                    f'<!doctype html><html><head><meta charset="utf-8"><title>{page.title()}'
                    f"</title></head><body>{node.evaluate('e => e.outerHTML')}</body></html>"
                )
            what = f"text {expect!r}" if expect else f"element {select!r}"
            raise RenderError(f"{what} not found after render ({len(attempts)} attempt(s))")
        except PlaywrightError as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            raise RenderError(msg, retryable=True) from exc
        finally:
            page.close()

    @staticmethod
    def _load(page: Page, url: str, click: str | None) -> int:
        """Load ``url``; return how many elements match ``click`` (1 if no click)."""
        response = page.goto(url, wait_until="networkidle", timeout=TIMEOUT_MS)
        if response is not None and response.status >= 400:
            raise RenderError(f"HTTP {response.status}")
        if not click:
            return 1
        n = page.locator(click).count()
        if n == 0:
            raise RenderError(f"click target not found: {click}")
        return n
