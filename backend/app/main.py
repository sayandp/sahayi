"""FastAPI app and routes.

    POST /ask          SSE stream: event "sources", then "token" events (validated answer
                       sentences), then "result" (validated JSON + meta); "error" on failure
    POST /ask/sync     the same result as one JSON response
    GET  /schemes      schemes in the corpus
    GET  /source/{chunk_id}   one chunk with a link to the source page (#page=N for PDFs)
    GET  /health

Run: uvicorn backend.app.main:app --port 8000  (python tasks.py serve)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.answer import Answerer, scheme_directory, scheme_id_of, source_link
from backend.app.config import get_settings
from backend.app.logging import configure_logging, log_trace, trace_record
from backend.app.production import production_answerer
from backend.app.schemas import AskRequest, AskResponse, SchemeSummary, SourceChunk

log = logging.getLogger("sahayi")


class RateLimiter:
    """In-memory sliding window per client IP (single process)."""

    def __init__(self, per_minute: int, window_s: float = 60.0) -> None:
        self.per_minute, self.window = per_minute, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float | None:
        """None if allowed, else seconds until the next request is allowed."""
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return self.window - (now - q[0])
            q.append(now)
            return None


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    answerer_factory: Callable[[], Answerer] | None = None,
    chunk_loader: Callable[[], dict[str, dict]] | None = None,
    log_path: object = "default",
) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # the API keeps no per-query disk caches: they would store derived query text
        os.environ.setdefault("SAHAYI_REWRITE_CACHE", "0")
        os.environ.setdefault("SAHAYI_RERANK_CACHE", "0")
        configure_logging()
        if os.environ.get("SAHAYI_PRELOAD", "1") == "1" and answerer_factory is None:
            answerer().warm_up()  # load models + indexes before the first request
        yield
        if state["answerer"] is not None:
            state["answerer"].retriever.close()

    # the public demo link exposes the app and its sources, not the API explorer
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if settings.demo_mode else {}
    app = FastAPI(title="Sahayi", version="0.3.0", lifespan=lifespan, **docs)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    limiter = RateLimiter(
        settings.demo_rate_limit_per_min if settings.demo_mode else settings.rate_limit_per_min
    )
    state: dict = {"answerer": None, "chunks": None}
    load_lock, answer_lock = threading.Lock(), threading.Lock()
    trace_kwargs = {} if log_path == "default" else {"path": log_path}

    def answerer() -> Answerer:
        with load_lock:
            if state["answerer"] is None:
                state["answerer"] = (answerer_factory or production_answerer)()
            return state["answerer"]

    def chunks() -> dict[str, dict]:
        with load_lock:
            if state["chunks"] is None:
                if chunk_loader:
                    state["chunks"] = chunk_loader()
                elif state["answerer"] is not None:
                    state["chunks"] = state["answerer"].retriever.chunks
                else:
                    from backend.app.retrieval.search import load_chunks, load_named_config

                    state["chunks"] = load_chunks(
                        load_named_config(settings.retrieval_config).strategy
                    )
            return state["chunks"]

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):  # noqa: ANN001, ANN202
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        rid = "".join(ch for ch in rid if ch.isalnum() or ch in "-_")[:64] or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info("request", extra={"request_id": rid, "path": request.url.path,
                                   "status": response.status_code,
                                   "ms": round((time.perf_counter() - t0) * 1000)})  # fmt: skip
        return response

    def guard(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if settings.demo_mode:  # the server binds to localhost, so this comes from Cloudflare
            client = request.headers.get("cf-connecting-ip") or client
        wait = limiter.check(client)
        if wait is not None:
            raise HTTPException(429, "rate limit exceeded", {"Retry-After": str(int(wait) + 1)})

    def run(question: str, rid: str) -> Iterator[tuple[str, dict]]:
        """Answer under a lock (one CPU-bound retrieval at a time) and log the trace."""
        with answer_lock:
            retrieved: list[str] = []
            for name, data in answerer().stream(question):
                if name == "sources":
                    retrieved = [s["chunk_id"] for s in data["sources"]]
                if name == "result":
                    data["meta"]["request_id"] = rid
                    log_trace(trace_record(rid, question, retrieved, data), **trace_kwargs)
                yield name, data

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> StreamingResponse:
        guard(request)
        rid = request.state.request_id

        def events() -> Iterator[str]:
            try:
                for name, data in run(req.question, rid):
                    yield sse(name, data)
            except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
                log.exception("ask failed", extra={"request_id": rid})
                yield sse("error", {"request_id": rid, "error": type(exc).__name__})

        headers = {"Cache-Control": "no-cache", "X-Request-ID": rid}
        # Cloudflare quick tunnels do not support SSE. The UI parses the body itself, so the
        # demo link sends the same bytes under a plain type.
        media = "text/plain; charset=utf-8" if settings.demo_mode else "text/event-stream"
        return StreamingResponse(events(), media_type=media, headers=headers)

    @app.post("/ask/sync", response_model=AskResponse)
    def ask_sync(req: AskRequest, request: Request) -> dict:
        guard(request)
        result: dict = {}
        for name, data in run(req.question, request.state.request_id):
            if name == "result":
                result = data
        return result

    @app.get("/schemes", response_model=list[SchemeSummary])
    def schemes() -> list[dict]:
        counts: dict[str, int] = defaultdict(int)
        for c in chunks().values():
            counts[scheme_id_of(c)] += 1
        return [{"scheme_id": i.scheme_id, "name": i.name, "department": i.department,
                 "level": i.level, "site": i.site, "n_chunks": counts.get(i.scheme_id, 0)}
                for i in scheme_directory().values()]  # fmt: skip

    @app.get("/source/{chunk_id}", response_model=SourceChunk)
    def source(chunk_id: str) -> dict:
        c = chunks().get(chunk_id)
        if c is None:
            raise HTTPException(404, f"unknown chunk_id {chunk_id!r}")
        return {**{k: c.get(k) for k in SourceChunk.model_fields if k != "link"},
                "link": source_link(c)}  # fmt: skip

    @app.get("/health")
    def health() -> dict:
        if settings.demo_mode:
            return {"status": "ok", "answerer_loaded": state["answerer"] is not None}
        return {
            "status": "ok",
            "answerer_loaded": state["answerer"] is not None,
            "retrieval_config": settings.retrieval_config,
            "answer_model": settings.answer_model,
            "answer_mode": settings.answer_mode,
            "groq_key": bool(settings.groq_api_key),
            "gemini_key": bool(settings.gemini_api_key),
            "jina_key": bool(settings.jina_api_key or os.environ.get("JINA_API_KEY")),
        }

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse({"detail": exc.detail, "request_id": rid}, exc.status_code,
                            headers=exc.headers)  # fmt: skip

    if settings.static_dir:  # last, so API routes win; html=True serves index.html at /
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="ui")
    return app


app = create_app()
