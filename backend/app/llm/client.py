"""LLM client: Groq primary, Gemini fallback, with latency and token accounting.

JSON-mode chat serves query rewrite and eval tooling. Answer generation streams
(``GroqStream``): Groq's JSON mode returns the whole completion as one delta, so the
answer call runs in plain mode with the format pinned by the prompt. Costs in INR come
from these token counts and the prices in config (Settings.cost_inr).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from backend.app.config import get_settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}
# Headroom under qwen's 1000 output-tokens/minute limit; Groq refuses larger requests outright
ANSWER_MAX_TOKENS = 900


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float  # the successful request only
    tokens_in: int = 0
    tokens_out: int = 0
    data: dict | None = field(default=None)
    throttled_ms: float = 0.0  # time spent on failed attempts / rate-limit waits


def _post(
    url: str, *, headers: dict, body: dict, timeout: float, retries: int = 3
) -> tuple[dict, float, float]:
    """POST with retries; returns (json, latency of the successful call, throttled ms)."""
    last = ""
    t_start = time.perf_counter()
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code == 200:
                ok_ms = (time.perf_counter() - t0) * 1000
                return r.json(), ok_ms, (t0 - t_start) * 1000
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in RETRY_STATUS:
                break
            wait = float(r.headers.get("retry-after", 0) or 0) or 1.5 * (attempt + 1)
            time.sleep(min(wait, 20))
    raise LLMError(last)


def reasoning_params(model: str, *, hide: bool = False) -> dict:
    """Least reasoning each Groq model allows (latency); ``hide`` drops it from output."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", **({"include_reasoning": False} if hide else {})}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def groq_chat(
    messages: list[dict], model: str, *, json_mode: bool = True, temperature: float = 0.0,
    timeout: float = 20.0, retries: int = 3, max_tokens: int | None = None,
) -> LLMResult:  # fmt: skip
    key = get_settings().groq_api_key
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    body: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if max_tokens:
        body["max_completion_tokens"] = max_tokens
    if model.startswith("openai/gpt-oss"):  # qwen: Groq's default (no reasoning tokens)
        body["reasoning_effort"] = "low"
    data, ok_ms, throttled = _post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        body=body,
        timeout=timeout,
        retries=retries,
    )
    usage = data.get("usage", {})
    return LLMResult(
        text=data["choices"][0]["message"]["content"],
        provider="groq",
        model=model,
        latency_ms=ok_ms,
        throttled_ms=throttled,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
    )


def gemini_chat(
    messages: list[dict], model: str, *, json_mode: bool = True, temperature: float = 0.0,
    timeout: float = 20.0, retries: int = 3, max_tokens: int | None = None,
) -> LLMResult:  # fmt: skip
    key = get_settings().gemini_api_key
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    body: dict = {"contents": contents, "generationConfig": {"temperature": temperature}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if max_tokens:
        body["generationConfig"]["maxOutputTokens"] = max_tokens
    data, ok_ms, throttled = _post(
        GEMINI_URL.format(model=model),
        headers={"x-goog-api-key": key},
        body=body,
        timeout=timeout,
        retries=retries,
    )
    usage = data.get("usageMetadata", {})
    return LLMResult(
        text=data["candidates"][0]["content"]["parts"][0]["text"],
        provider="gemini",
        model=model,
        latency_ms=ok_ms,
        throttled_ms=throttled,
        tokens_in=usage.get("promptTokenCount", 0),
        tokens_out=usage.get("candidatesTokenCount", 0),
    )


def chat_json(messages: list[dict], *, groq_model: str | None = None,
              gemini_model: str | None = None, timeout: float = 20.0,
              retries: int = 3, allow_gemini: bool = False,
              max_tokens: int | None = None) -> LLMResult:  # fmt: skip
    """Groq; with ``allow_gemini`` also Gemini if a key is set. Raises LLMError on failure.

    Gemini is opt-in so eval judges never silently change provider when Groq's quota runs out.
    """
    s = get_settings()
    errors = []
    calls = [("groq", groq_chat, groq_model or s.llm_model_groq, s.groq_api_key)]
    if allow_gemini:
        calls.append(
            ("gemini", gemini_chat, gemini_model or s.answer_model_gemini, s.gemini_api_key)
        )
    for name, fn, model, key in calls:
        if not key:
            continue
        try:
            result = fn(messages, model, json_mode=True, timeout=timeout, retries=retries,
                        max_tokens=max_tokens)  # fmt: skip
            result.data = json.loads(result.text)
            return result
        except (LLMError, json.JSONDecodeError, KeyError, IndexError) as exc:
            errors.append(f"{name}/{model}: {exc}")
    raise LLMError("; ".join(errors) or "no LLM provider configured")


class GroqStream:
    """Streamed Groq chat completion in plain mode; iterate for text deltas.

    After iteration: ``text``, ``tokens_in``, ``tokens_out``, ``ttft_ms`` (request start
    to first content delta), ``latency_ms`` and ``throttled_ms`` (429 waits before the
    successful attempt). Retries happen only before the first byte, so no text is
    ever emitted twice.
    """

    def __init__(
        self, messages: list[dict], model: str, *, temperature: float = 0.0,
        timeout: float = 30.0, retries: int = 2, max_tokens: int = ANSWER_MAX_TOKENS,
    ) -> None:  # fmt: skip
        self.messages, self.model, self.temperature = messages, model, temperature
        self.timeout, self.retries, self.max_tokens = timeout, retries, max_tokens
        self.text = ""
        self.tokens_in = self.tokens_out = 0
        self.ttft_ms: float | None = None
        self.latency_ms = self.throttled_ms = 0.0
        self.provider = "groq"
        self.finish_reason: str | None = None  # "length" means the output was cut off

    def __iter__(self) -> Iterator[str]:
        key = get_settings().groq_api_key
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        body = {"model": self.model, "messages": self.messages, "temperature": self.temperature,
                "stream": True, "max_completion_tokens": self.max_tokens,
                **reasoning_params(self.model, hide=True)}  # fmt: skip
        t_start = time.perf_counter()
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                with httpx.stream("POST", GROQ_URL, json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {key}"}) as r:  # fmt: skip
                    if r.status_code != 200:
                        last = f"HTTP {r.status_code}: {r.read()[:200]!r}"
                        if r.status_code not in RETRY_STATUS or attempt == self.retries:
                            break
                        wait = float(r.headers.get("retry-after", 0) or 0) or 1.5 * (attempt + 1)
                        time.sleep(min(wait, 10))
                        continue
                    self.throttled_ms = (t0 - t_start) * 1000
                    yield from self._read(r, t_start)
                    self.latency_ms = (time.perf_counter() - t0) * 1000
                    return
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                if self.text:  # failed mid-stream: cannot retry without duplicating text
                    raise LLMError(f"stream broke: {last}") from exc
        raise LLMError(last)

    def _read(self, r: httpx.Response, t_start: float) -> Iterator[str]:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
            if usage:
                self.tokens_in = usage.get("prompt_tokens", 0)
                self.tokens_out = usage.get("completion_tokens", 0)
            for choice in chunk.get("choices", []):
                self.finish_reason = choice.get("finish_reason") or self.finish_reason
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if self.ttft_ms is None:
                        self.ttft_ms = (time.perf_counter() - t_start) * 1000
                    self.text += delta
                    yield delta


class GeminiStream:
    """Streamed Gemini generateContent; same surface as GroqStream."""

    def __init__(self, messages: list[dict], model: str, *, temperature: float = 0.0,
                 timeout: float = 30.0, max_tokens: int = ANSWER_MAX_TOKENS) -> None:  # fmt: skip
        self.messages, self.model, self.temperature = messages, model, temperature
        self.timeout, self.max_tokens = timeout, max_tokens
        self.text = ""
        self.tokens_in = self.tokens_out = 0
        self.ttft_ms: float | None = None
        self.latency_ms = self.throttled_ms = 0.0
        self.provider = "gemini"
        self.finish_reason: str | None = None  # "length" means the output was cut off

    def __iter__(self) -> Iterator[str]:
        key = get_settings().gemini_api_key
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        system = "\n".join(m["content"] for m in self.messages if m["role"] == "system")
        body: dict = {
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]}
                         for m in self.messages if m["role"] != "system"],
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_tokens},
        }  # fmt: skip
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = GEMINI_URL.format(model=self.model).replace(
            ":generateContent", ":streamGenerateContent"
        )
        t0 = time.perf_counter()
        try:
            with httpx.stream("POST", url, params={"alt": "sse"}, json=body, timeout=self.timeout,
                              headers={"x-goog-api-key": key}) as r:  # fmt: skip
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.read()[:200]!r}")
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usageMetadata") or {}
                    self.tokens_in = usage.get("promptTokenCount", self.tokens_in)
                    self.tokens_out = usage.get("candidatesTokenCount", self.tokens_out)
                    for cand in chunk.get("candidates", []):
                        if cand.get("finishReason") == "MAX_TOKENS":
                            self.finish_reason = "length"
                        for part in (cand.get("content") or {}).get("parts", []):
                            if part.get("text"):
                                if self.ttft_ms is None:
                                    self.ttft_ms = (time.perf_counter() - t0) * 1000
                                self.text += part["text"]
                                yield part["text"]
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        self.latency_ms = (time.perf_counter() - t0) * 1000


class AllGeneratorsBusy(LLMError):
    """Every generator in the fallback chain failed before producing any text."""


class FallbackStream:
    """Tries generators in order, moving on when one fails before its first text.

    A 429 (including Groq's output-tokens-per-minute limit), a timeout or a 5xx all raise
    LLMError from the underlying stream. Once text has been sent the stream cannot switch
    without repeating itself, so a later failure propagates. ``model`` is the generator
    that answered; ``skipped`` says which ones failed and why.
    """

    def __init__(self, messages: list[dict], chain: list[tuple[str, str]], *,
                 timeout: float = 8.0, factories: dict | None = None) -> None:  # fmt: skip
        self.messages, self.chain, self.timeout = messages, chain, timeout
        self.factories = factories or {
            "groq": lambda m, model, t: GroqStream(m, model, timeout=t, retries=0),
            "gemini": lambda m, model, t: GeminiStream(m, model, timeout=t),
        }
        self.model = chain[0][1] if chain else ""
        self.skipped: list[str] = []
        self.inner = None

    def __iter__(self) -> Iterator[str]:
        for provider, model in self.chain:
            stream = self.factories[provider](self.messages, model, self.timeout)
            started = False
            try:
                for delta in stream:
                    if not started:
                        started, self.inner, self.model = True, stream, model
                    yield delta
                if started:
                    return
                self.skipped.append(f"{model}: empty response")
            except LLMError as exc:
                if started:
                    raise
                self.skipped.append(f"{model}: {str(exc)[:160]}")
        raise AllGeneratorsBusy("; ".join(self.skipped) or "no generator configured")

    def __getattr__(self, name: str):  # noqa: ANN204 - text, tokens_in, ttft_ms, ...
        if name in ("text", "tokens_in", "tokens_out", "ttft_ms", "throttled_ms", "finish_reason"):
            inner = self.__dict__.get("inner")
            if inner is not None:
                return getattr(inner, name, None)
            return "" if name == "text" else (None if name in ("ttft_ms", "finish_reason") else 0)
        raise AttributeError(name)
