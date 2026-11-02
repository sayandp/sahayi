"""Orchestrates retrieve -> rerank -> generate -> validate citations (answer pipeline).

``Answerer.stream(question)`` yields events in this order:
  ("sources", {...})   retrieved excerpts (before generation starts)
  ("token",   {...})   answer text, one validated sentence at a time
  ("result",  {...})   the validated response (CLAUDE.md schema) with meta
Only sentences that pass citation validation are streamed, so the client never shows
text that the final result drops. ``Answerer.ask`` returns the result alone.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml

from backend.app.config import get_settings
from backend.app.llm.client import (
    ANSWER_MAX_TOKENS,
    AllGeneratorsBusy,
    GroqStream,
    LLMError,
    chat_json,
)
from backend.app.llm.prompts import (
    ABSTAIN_MESSAGE,
    ABSTAIN_TOKEN,
    ANSWER_COMPACT,
    ANSWER_LANG_NAME,
    ANSWER_SYSTEM,
    ANSWER_USER,
    ANSWER_VERSION,
    BUSY_NOTICE,
    ENTAIL_SYSTEM,
    GENERIC_CONTACT,
    VERDICT,
)
from backend.app.profile import clean_profile, profile_text
from backend.app.query import query_kind
from backend.app.validate import (
    TAG,
    Validator,
    filter_cards,
    split_sentences,
    strip_tags,
    tags_in,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES_YAML = ROOT / "ingest" / "sources.yaml"
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
ML_CHAR = re.compile("[ഀ-ൿ]")
K_CONTEXT = 5
MAX_EXCERPT_CHARS = 2000
SNIPPET_CHARS = 280
Event = tuple[str, dict]


# ------------------------------------------------------------------ sources


@dataclass(frozen=True)
class SchemeInfo:
    scheme_id: str
    name: str
    department: str
    level: str
    site: str


@lru_cache(maxsize=1)
def scheme_directory() -> dict[str, SchemeInfo]:
    """scheme_id -> name, department and official site (from ingest/sources.yaml)."""
    raw = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8"))["sources"]
    out = {}
    for s in raw:
        urls = [u if isinstance(u, str) else u["url"] for u in s["urls"]]
        official = [u for u in urls if "myscheme.gov.in" not in u] or urls
        site = f"{urlparse(official[0]).scheme}://{urlparse(official[0]).netloc}"
        out[s["id"]] = SchemeInfo(s["id"], s["scheme"], s["department"], s["level"], site)
    return out


@lru_cache(maxsize=1)
def doc_kinds() -> dict[str, str]:
    """doc_id -> pdf | html, from the download manifest (absent in a slim deploy)."""
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["doc_id"]: r.get("kind") or "" for r in rows}


def scheme_id_of(chunk: dict) -> str:
    return chunk.get("scheme_id") or chunk["doc_id"].split("--")[0]


def source_link(chunk: dict) -> str:
    """Link to the source document, at the page for PDFs."""
    kind = doc_kinds().get(chunk["doc_id"]) or (
        "pdf" if ".pdf" in chunk["source_url"].lower() else "html"
    )
    return f"{chunk['source_url']}#page={chunk['page']}" if kind == "pdf" else chunk["source_url"]


def contact_for(scheme_id: str | None, lang: str) -> str:
    info = scheme_directory().get(scheme_id or "")
    generic = GENERIC_CONTACT[lang]
    if not info:
        return generic
    joiner = " അല്ലെങ്കിൽ " if lang == "ml" else ", or "
    return f"{info.department} ({info.site}){joiner}{generic}"


# ------------------------------------------------------------------ prompting


@dataclass
class Excerpt:
    alias: str  # C1..Cn, what the model cites
    chunk: dict


def answer_lang(rewrite_lang: str | None, question: str) -> str:
    """Answer language: Malayalam for Malayalam or Manglish questions, else English."""
    if rewrite_lang is None:  # no rewrite ran (retrieval-only mode)
        return guess_lang(question)
    if rewrite_lang in ("ml", "manglish"):
        return "ml"
    if rewrite_lang == "en":
        return "en"
    return "ml" if ML_CHAR.search(question) else "en"


def guess_lang(question: str) -> str:
    """Answer language without the rewrite: Malayalam script or Manglish -> ml."""
    return "en" if query_kind(question) == "en" else "ml"


def build_excerpts(chunks: list[dict], k: int = K_CONTEXT) -> list[Excerpt]:
    return [Excerpt(f"C{i}", c) for i, c in enumerate(chunks[:k], start=1)]


def excerpt_block(ex: Excerpt, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    c = ex.chunk
    text = c["text"] if len(c["text"]) <= max_chars else c["text"][:max_chars] + " …"
    section = f" | section: {c['section']}" if c.get("section") else ""
    return (f"[{ex.alias}] scheme_id: {scheme_id_of(c)} | scheme: {c['scheme']} | "
            f"department: {c['department']}{section} | page {c['page']}\n{text}")  # fmt: skip


def build_messages(question: str, lang: str, profile: dict, excerpts: list[Excerpt]) -> list[dict]:
    user = ANSWER_USER[lang].format(
        question=question,
        answer_lang=ANSWER_LANG_NAME[lang][lang],
        profile=profile_text(profile),
        excerpts="\n\n".join(excerpt_block(e) for e in excerpts),
    )
    return [{"role": "system", "content": ANSWER_SYSTEM[lang]}, {"role": "user", "content": user}]


def parse_json(text: str) -> dict | None:
    """The generator's JSON object, tolerating code fences or stray text around it."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ------------------------------------------------------------------ streaming


class AnswerFieldStream:
    """Incrementally decodes the "answer" string of a JSON object being streamed."""

    _START = re.compile(r'"answer"\s*:\s*"')

    def __init__(self) -> None:
        self.raw = ""
        self.pos: int | None = None  # index in raw of the next undecoded char
        self.done = False

    def feed(self, delta: str) -> str:
        self.raw += delta
        if self.done:
            return ""
        if self.pos is None:
            m = self._START.search(self.raw)
            if not m:
                return ""
            self.pos = m.end()
        i, end = self.pos, len(self.raw)
        while i < end:
            ch = self.raw[i]
            if ch == '"':
                self.done = True
                break
            if ch == "\\":
                if i + 1 >= end:
                    break
                if self.raw[i + 1] == "u":
                    if i + 6 > end:
                        break
                    code = int(self.raw[i + 2 : i + 6], 16)
                    if 0xD800 <= code < 0xDC00:  # surrogate pair: need the low half too
                        if i + 12 > end:
                            break
                        i += 12
                        continue
                    i += 6
                    continue
                i += 2
                continue
            i += 1
        piece = self.raw[self.pos : i]
        self.pos = i + (1 if self.done else 0)
        try:
            return json.loads(f'"{piece}"')
        except json.JSONDecodeError:
            return piece


class BackgroundStream:
    """Consumes a stream in a thread (so generation starts before anyone reads it).

    Iterating yields the buffered and then the live deltas; ``cancel`` stops and closes
    the underlying request. Stats (text, tokens, ttft_ms...) are read from the inner
    stream once it is exhausted.
    """

    _END = object()

    def __init__(self, inner) -> None:  # noqa: ANN001 - GroqStream-like
        import queue
        import threading

        self.inner = inner
        self.queue: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="spec-gen")
        self.thread.start()

    def _run(self) -> None:
        it = iter(self.inner)
        try:
            for delta in it:
                if self.cancelled.is_set():
                    break
                self.queue.put(delta)
        except Exception as exc:  # noqa: BLE001 - re-raised in the consumer
            self.queue.put(exc)
        finally:
            close = getattr(it, "close", None)
            if close:
                close()
            self.queue.put(self._END)

    def cancel(self) -> None:
        self.cancelled.set()

    def __iter__(self) -> Iterator[str]:
        while True:
            item = self.queue.get()
            if item is self._END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def __getattr__(self, name: str):  # noqa: ANN204 - text, tokens_in, ttft_ms, ...
        if name in ("inner", "queue", "cancelled", "thread"):
            raise AttributeError(name)
        if name in ("tokens_in", "tokens_out") and self.cancelled.is_set():
            self.thread.join(timeout=5)
        return getattr(self.inner, name)


class SentenceGate:
    """Releases answer sentences once complete and validated (citations + numbers)."""

    def __init__(self, validator: Validator) -> None:
        self.validator = validator
        self.buf = ""
        self.released = 0  # sentences of split_sentences(buf) already decided

    def _decide(self, sentences: list[str]) -> list[str]:
        out = []
        for sent in sentences:
            if sent.rstrip().endswith("?") and not tags_in(sent):
                out.append(sent)
                continue
            n_before = len(self.validator.dropped)
            res = self.validator.check(sent)
            del self.validator.dropped[n_before:]  # the final validation records drops
            if res:
                out.append(res[0] + "".join(f" [{c}]" for c in res[1]))
        return out

    def feed(self, text: str) -> list[str]:
        self.buf += text
        sentences = split_sentences(self.buf)
        if len(sentences) <= self.released + 1:
            return []
        # the last split piece may still grow (more words or trailing citation tags)
        complete = sentences[self.released : -1]
        last = sentences[-1].lstrip()
        if last.startswith("[") or not last:
            return []
        self.released += len(complete)
        return self._decide(complete)

    def flush(self) -> list[str]:
        sentences = split_sentences(self.buf)[self.released :]
        self.released += len(sentences)
        return self._decide(sentences)


# ------------------------------------------------------------------ answerer


def load_retriever(config_name: str):  # noqa: ANN201 - search.Retriever
    from backend.app.retrieval.search import Retriever, load_named_config

    settings = get_settings()
    path = settings.retrieval_configs_path
    config = load_named_config(config_name, *([Path(path)] if path else []))
    index = Path(settings.serve_qdrant_path) if settings.serve_qdrant_path else None
    return Retriever(config, index)


class Answerer:
    """One instance per process: holds the retriever (models + indexes)."""

    def __init__(
        self,
        retriever=None,  # noqa: ANN001 - search.Retriever or a test double
        *,
        model: str | None = None,
        k_context: int = K_CONTEXT,
        entail_model: str | None = None,
        stream_factory: Callable[..., GroqStream] = GroqStream,
        json_fallback: Callable[..., object] = chat_json,
        keep_raw: bool = False,
        speculative: bool = False,
        cards_filter: bool = False,
        degrade: bool = False,
        cache=None,  # noqa: ANN001 - cache.AnswerCache
        before_retry: Callable[[], None] | None = None,  # evals: wait out the output-token window
    ) -> None:
        s = get_settings()
        self.retriever = retriever or load_retriever(s.retrieval_config)
        self.model = model or s.answer_model
        self.k_context = k_context
        self.entail_model = entail_model  # optional entailment check (off by default)
        self.stream_factory = stream_factory
        self.json_fallback = json_fallback
        self.keep_raw = keep_raw
        self.last_context: list[str] = []  # chunk_ids of the excerpts of the last answer
        # start generating on the retriever's speculative top-k while the rewrite call is in
        # flight; kept only if the final top-k (as a set) and answer language match. Off by
        # default: the final top-5 equals the speculative one for only 7.6% of the gold
        # questions (evals/spec_match.py, 2026-10-20), so it would mostly waste tokens.
        self.speculative = speculative
        # production behaviour, set by production.py; evals keep all three off
        self.cards_filter = cards_filter  # drop scheme cards the answer does not rest on
        self.degrade = degrade  # all generators busy: excerpts + a notice, not an error
        self.cache = cache
        self.before_retry = before_retry

    def warm_up(self) -> None:
        """Load the embedder and reranker(s) now, not on the first (timed) request."""
        from backend.app.retrieval.dense import embed_queries, onnx_query_encoder
        from backend.app.retrieval.rerank import RerankUnavailable, load_backend

        config = getattr(self.retriever, "config", None)
        if config is None:
            return
        if config.embed_onnx_file:
            onnx_query_encoder(config.embed_onnx_file).encode(["warm-up"])
        elif config.mode != "bm25":
            embed_queries(["warm-up"], config.embed_model)
        spec = config.rerank_spec if config.rerank else None
        while spec is not None:
            with contextlib.suppress(RerankUnavailable, Exception):  # api: keyless / down
                load_backend(spec).score("warm-up", ["warm-up"])
            spec = spec.fallback

    def ask(self, question: str) -> dict:
        result: dict = {}
        for name, data in self.stream(question):
            if name == "result":
                result = data
        return result

    def _speculate(self, question: str, spec: dict) -> Callable[[list[str]], None]:
        """Retriever hook: start generating on the speculative top-k (see stream)."""

        def start(ids: list[str]) -> None:
            chunks = [self.retriever.chunks[i] for i in ids[: self.k_context]]
            lang = guess_lang(question)
            excerpts = build_excerpts(chunks, self.k_context)
            messages = build_messages(question, lang, {}, excerpts)
            t_gen = time.perf_counter()
            stream = BackgroundStream(self.stream_factory(messages, self.model))
            spec.update(ids={c["chunk_id"] for c in chunks}, lang=lang, excerpts=excerpts,
                        messages=messages, t_gen=t_gen, stream=stream)  # fmt: skip

        return start

    def stream(self, question: str) -> Iterator[Event]:
        if self.cache is None:
            yield from self._generate(question)
            return
        t0 = time.perf_counter()
        hit = self.cache.get(question, guess_lang(question))
        if hit is not None:
            result = hit["result"]
            ms = (time.perf_counter() - t0) * 1000
            yield "sources", hit["sources"]
            yield "token", {"text": result["answer"]}
            result["meta"] |= {"cached": True, "ttfc_ms": round(ms), "ttft_ms": round(ms),
                               "latency_ms": round((time.perf_counter() - t0) * 1000),
                               "latency": {"ttfc_ms": ms, "ttft_ms": ms, "total_ms": ms},
                               "tokens_in": 0, "tokens_out": 0, "cost_inr": 0.0}  # fmt: skip
            yield "result", result
            return
        sources: dict = {}
        for name, data in self._generate(question):
            if name == "sources":
                sources = data
            elif name == "result":
                self.cache.put(question, guess_lang(question), sources, data)
            yield name, data

    def _generate(self, question: str) -> Iterator[Event]:
        s = get_settings()
        t0 = time.perf_counter()
        spec: dict = {}
        hook = self.speculative and hasattr(self.retriever, "on_speculative")
        if hook:
            self.retriever.on_speculative = self._speculate(question, spec)
        try:
            hits = self.retriever.search(question)
        finally:
            if hook:
                self.retriever.on_speculative = None
        trace = self.retriever.last_trace
        rw = getattr(self.retriever, "last_rewrite", None)
        t_retrieved = time.perf_counter()
        lang = answer_lang(trace.rewrite_lang, question)
        profile = clean_profile(rw.profile_hints if rw else None)
        excerpts = build_excerpts([h.chunk for h in hits], self.k_context)
        # keep the speculative generation if it saw the same excerpts in the same language
        final_ids = {e.chunk["chunk_id"] for e in excerpts}
        spec_hit = bool(spec) and spec["ids"] == final_ids and spec["lang"] == lang
        if spec_hit:
            excerpts = spec["excerpts"]  # same chunks; keep the order the prompt used
        elif spec:
            spec["stream"].cancel()
        aliases = {e.alias: e.chunk["chunk_id"] for e in excerpts}
        self.last_context = list(aliases.values())
        chunks = {e.chunk["chunk_id"]: e.chunk for e in excerpts}
        scheme_ids = {cid: scheme_id_of(c) for cid, c in chunks.items()}
        user_text = " ".join([question, *map(str, profile.values())])
        timing_ttfc = (time.perf_counter() - t0) * 1000  # when the "sources" event goes out
        yield "sources", {
            "lang": lang,
            "sources": [self._citation(e.chunk) | {"tag": e.alias, "scheme": e.chunk["scheme"]}
                        for e in excerpts],
        }  # fmt: skip

        usage = {"tokens_in": trace.rewrite_tokens_in, "tokens_out": trace.rewrite_tokens_out,
                 "cost_inr": s.cost_inr(trace.rewrite_model or "", trace.rewrite_tokens_in,
                                        trace.rewrite_tokens_out)}  # fmt: skip
        flags: list[str] = ["speculative_hit" if spec_hit else "speculative_miss"] if spec else []
        timing: dict = {"retrieval_ms": (t_retrieved - t0) * 1000}
        data: dict | None = None
        ttft_ms: float | None = None
        gen_model = self.model
        raw_text = ""
        busy, fallbacks, truncated = False, [], False
        if not excerpts:
            flags.append("no_context")
        else:
            gate = SentenceGate(Validator(aliases, chunks, user_text))
            field_stream = AnswerFieldStream()
            if spec_hit:
                messages, t_gen, stream = spec["messages"], spec["t_gen"], spec["stream"]
            else:
                messages = build_messages(question, lang, profile, excerpts)
                t_gen = time.perf_counter()
                stream = self.stream_factory(messages, self.model)
            try:
                flushed = False
                for delta in stream:
                    ready = gate.feed(field_stream.feed(delta))
                    if field_stream.done and not flushed:  # "answer" closed: release the rest
                        ready, flushed = ready + gate.flush(), True
                    for sent in ready:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        yield "token", {"text": sent + " "}
                if field_stream.pos is not None and not flushed:  # stream ended mid-answer
                    for sent in gate.flush():
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        yield "token", {"text": sent + " "}
                raw_text = stream.text
                if getattr(stream, "finish_reason", None) == "length":
                    # never parse partial JSON: retry once, shorter, not streamed
                    flags.append("compact_retry")
                    self._add_usage(usage, getattr(stream, "model", gen_model), stream.tokens_in,
                                    stream.tokens_out)  # fmt: skip
                    if self.before_retry:
                        self.before_retry()
                    stream = self.stream_factory(
                        [{**messages[0], "content": messages[0]["content"] + ANSWER_COMPACT[lang]},
                         *messages[1:]], self.model)  # fmt: skip
                    raw_text = "".join(stream)
                    if getattr(stream, "finish_reason", None) == "length":
                        truncated = True
                        flags.append("truncated_failed")
                if not truncated:
                    data = parse_json(raw_text)
                timing["gen_first_delta_ms"] = stream.ttft_ms
                timing["throttled_ms"] = stream.throttled_ms
            except AllGeneratorsBusy as exc:
                busy = True
                flags.append(f"generators_busy: {str(exc)[:400]}")
            except LLMError as exc:
                flags.append(f"stream_failed: {str(exc)[:400]}")
            gen_model = getattr(stream, "model", gen_model)
            fallbacks = list(getattr(stream, "skipped", []))
            self._add_usage(usage, gen_model, stream.tokens_in, stream.tokens_out)
            if spec and not spec_hit:  # the cancelled speculative call still billed its prompt
                wasted = spec["stream"]
                self._add_usage(usage, gen_model, wasted.tokens_in, wasted.tokens_out)
            if data is None and not busy and not truncated:
                flags.append("json_retry")
                try:
                    # one retry only: the user is already waiting on a failed stream
                    res = self.json_fallback(messages, groq_model=self.model, retries=1,
                                             max_tokens=ANSWER_MAX_TOKENS)  # fmt: skip
                    data, gen_model, raw_text = res.data, res.model, res.text
                    self._add_usage(usage, res.model, res.tokens_in, res.tokens_out)
                except LLMError as exc:
                    flags.append(f"generation_failed: {str(exc)[:400]}")
            timing["generate_ms"] = (time.perf_counter() - t_gen) * 1000

        t_val = time.perf_counter()
        if self.degrade and data is None and excerpts:
            result = self._degraded(lang, excerpts, chunks, flags)
        else:
            result = self._finish(user_text, data, lang, excerpts, aliases, chunks, scheme_ids,
                                  flags, usage)  # fmt: skip
        if truncated and not result.get("degraded"):
            result["abstain_reason"] = "truncated"
        timing["validate_ms"] = (time.perf_counter() - t_val) * 1000
        if ttft_ms is None:  # nothing streamed (abstained): the result is the first text
            ttft_ms = (time.perf_counter() - t0) * 1000
            yield "token", {"text": result["answer"]}
        timing["ttfc_ms"] = timing_ttfc
        timing["ttft_ms"] = ttft_ms
        timing["total_ms"] = (time.perf_counter() - t0) * 1000
        retrieval = {k: round(getattr(trace, k), 1) for k in (
            "rewrite_ms", "rewrite_wait_ms", "bm25_ms", "dense_ms", "spec_rerank_ms",
            "rerank_ms")}  # fmt: skip
        result["meta"] = {
            "latency_ms": round(timing["total_ms"]),
            "ttfc_ms": round(timing_ttfc),
            "ttft_ms": round(ttft_ms),
            "latency": {k: round(v, 1) for k, v in timing.items() if v is not None}
            | {"retrieval_stages": retrieval},
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "cost_inr": round(usage["cost_inr"], 4),
            "model": gen_model,
            "fallbacks": fallbacks,
            "cached": False,
            "rewrite_model": trace.rewrite_model,
            "rerank_backend": trace.rerank_backend_used,
            "rerank_fallback": trace.rerank_fallback,
            "top_rerank_score": trace.top_rerank_score,
            "prompt_version": ANSWER_VERSION,
            "prices_checked": s.prices_checked,
        }
        if self.keep_raw:  # evals only: the unvalidated generation, for failure analysis
            result["raw_generation"] = raw_text
        yield "result", result

    @staticmethod
    def _add_usage(usage: dict, model: str, tokens_in: int, tokens_out: int) -> None:
        usage["tokens_in"] += tokens_in
        usage["tokens_out"] += tokens_out
        usage["cost_inr"] += get_settings().cost_inr(model, tokens_in, tokens_out)

    @staticmethod
    def _citation(chunk: dict) -> dict:
        text = " ".join(chunk["text"].split())
        return {
            "chunk_id": chunk["chunk_id"],
            "source_url": chunk["source_url"],
            "page": chunk["page"],
            "link": source_link(chunk),
            "scheme": chunk.get("scheme"),
            "downloaded_at": chunk.get("downloaded_at"),
            "snippet": text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else ""),
        }

    def _finish(self, user_text: str, data: dict | None, lang: str, excerpts: list[Excerpt],
                aliases: dict, chunks: dict, scheme_ids: dict, flags: list[str],
                usage: dict) -> dict:  # fmt: skip
        validator = Validator(aliases, chunks, user_text)
        model_abstained = bool(
            data is None
            or data.get("abstained") is True
            or str(data.get("answer") or "").strip().startswith(ABSTAIN_TOKEN)
        )
        v = None
        if not model_abstained:
            v = validator.validate(data, scheme_ids)
            if self.entail_model and v.sentences:
                self._entail(v, chunks, usage, flags)
            if self.cards_filter:
                answer_cited = list(dict.fromkeys(c for x in v.sentences for c in x["citations"]))
                v.schemes, gone = filter_cards(v.schemes, answer_cited, list(aliases.values()),
                                               chunks, scheme_ids)  # fmt: skip
                flags.extend(f"card_dropped:{sid}" for sid in gone)
                v.cited_ids = list(dict.fromkeys(
                    answer_cited + [c for sc in v.schemes for c in sc["citations"]]))  # fmt: skip
        if v is not None and v.empty and v.schemes:
            self._answer_from_cards(v, lang, flags)
        abstained = model_abstained or v is None or v.empty
        reason = None
        if abstained:
            reason = "model" if data is not None and model_abstained else (
                "error" if data is None else "validation")  # fmt: skip
        top_scheme = scheme_id_of(excerpts[0].chunk) if excerpts else None
        cited = [] if abstained or v is None else v.cited_ids
        dates = sorted(chunks[c]["downloaded_at"] for c in cited) or sorted(
            c["downloaded_at"] for c in chunks.values()
        )
        follow = data.get("follow_up_question") if data and not abstained else None
        return {
            "answer": (ABSTAIN_MESSAGE[lang].format(contact=contact_for(top_scheme, lang))
                       if abstained else v.answer),
            "lang": lang,
            "schemes": [] if abstained or v is None else v.schemes,
            "citations": [self._citation(chunks[c]) for c in cited],
            "abstained": abstained,
            "abstain_reason": reason,
            "degraded": False,
            "follow_up_question": strip_tags(follow) if isinstance(follow, str) and follow.strip()
            else None,
            "as_of": dates[0] if dates else None,
            "validation": {
                "dropped": [d.__dict__ for d in (v.dropped if v else [])],
                "unknown_citations": v.unknown_citations if v else [],
                "flags": flags + (v.flags if v else []),
                "kept_sentences": len(v.sentences) if v else 0,
            },
        }  # fmt: skip

    def _degraded(self, lang: str, excerpts: list[Excerpt], chunks: dict, flags: list[str]) -> dict:
        """Every generator failed: the notice plus verbatim, cited excerpts grouped by scheme."""
        cards: dict[str, dict] = {}
        for e in excerpts:
            c, sid = e.chunk, scheme_id_of(e.chunk)
            card = cards.setdefault(sid, {
                "scheme_id": sid, "name": c["scheme"], "eligible": None, "reasons": [],
                "benefit": None, "how_to_apply": None, "documents": [],
                "citations": []})  # fmt: skip
            card["citations"].append(c["chunk_id"])
            quote = " ".join(c["text"].split())[:SNIPPET_CHARS] + f" [{c['chunk_id']}]"
            section = c.get("section")
            if section == "benefits" and not card["benefit"]:
                card["benefit"] = quote
            elif section == "how_to_apply" and not card["how_to_apply"]:
                card["how_to_apply"] = quote
            elif section in ("eligibility", "exclusions"):
                card["reasons"].append(quote)
            elif section == "documents":
                card["documents"].append(quote)
        return {
            "answer": BUSY_NOTICE[lang],
            "lang": lang,
            "schemes": list(cards.values()),
            "citations": [self._citation(e.chunk) for e in excerpts],
            "abstained": False,
            "abstain_reason": None,
            "degraded": True,
            "follow_up_question": None,
            "as_of": min(c["downloaded_at"] for c in chunks.values()),
            "validation": {"dropped": [], "unknown_citations": [], "flags": flags,
                           "kept_sentences": 0},
        }  # fmt: skip

    @staticmethod
    def _answer_from_cards(v, lang: str, flags: list[str]) -> None:  # noqa: ANN001 - Validation
        """No answer sentence survived, but scheme cards did: rebuild the answer from their
        validated fields (verdict + first reason, else the benefit), each with its tags."""
        for sc in v.schemes:
            field = sc["reasons"][0] if sc["reasons"] else sc["benefit"]
            if not field:
                continue
            cites = [t for t in tags_in(field) if t in sc["citations"]] or list(sc["citations"])
            verdict = f" ({VERDICT[lang][sc['eligible']]})" if sc["reasons"] else ""
            v.sentences.append({"text": f"{sc['name']}{verdict}: {strip_tags(field)}",
                                "citations": cites})  # fmt: skip
        if v.sentences:
            v.answer = " ".join(s["text"] + "".join(f" [{c}]" for c in s["citations"])
                                for s in v.sentences)  # fmt: skip
            flags.append("answer_from_scheme_cards")

    def _entail(self, v, chunks: dict, usage: dict, flags: list[str]) -> None:  # noqa: ANN001
        """Optional LLM entailment check of kept sentences against their cited chunks."""
        from backend.app.validate import Dropped

        cited = list(dict.fromkeys(c for s in v.sentences for c in s["citations"]))
        label = {c: f"E{i}" for i, c in enumerate(cited, start=1)}
        body = "\n\n".join(f"[{label[c]}]\n{chunks[c]['text'][:MAX_EXCERPT_CHARS]}" for c in cited)
        lines = "\n".join(f"{i}. {s['text']} " + "".join(f"[{label[c]}]" for c in s["citations"])
                          for i, s in enumerate(v.sentences, start=1))  # fmt: skip
        t0 = time.perf_counter()
        try:
            res = self.json_fallback(
                [{"role": "system", "content": ENTAIL_SYSTEM},
                 {"role": "user", "content": f"Excerpts:\n{body}\n\nSentences:\n{lines}"}],
                groq_model=self.entail_model,
            )  # fmt: skip
        except LLMError as exc:
            flags.append(f"entail_failed: {str(exc)[:80]}")
            return
        self._add_usage(usage, res.model, res.tokens_in, res.tokens_out)
        flags.append(f"entail_ms={(time.perf_counter() - t0) * 1000:.0f}")
        bad = {
            int(x["i"])
            for x in (res.data or {}).get("verdicts", [])
            if isinstance(x, dict) and x.get("supported") is False and str(x.get("i")).isdigit()
        }
        if not bad:
            return
        keep = []
        for i, s in enumerate(v.sentences, start=1):
            if i in bad and s["citations"]:
                v.dropped.append(Dropped(s["text"], "entailment"))
            else:
                keep.append(s)
        v.sentences = keep
        v.answer = " ".join(s["text"] + "".join(f" [{c}]" for c in s["citations"]) for s in keep)
        scheme_cites = [c for sc in v.schemes for c in sc["citations"]]
        v.cited_ids = list(dict.fromkeys([c for s in keep for c in s["citations"]] + scheme_cites))


__all__ = ["Answerer", "AnswerFieldStream", "SentenceGate", "TAG", "split_sentences"]
