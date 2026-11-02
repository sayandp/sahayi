"""Cross-encoder reranking behind one interface, with several backends.

Backends (``RerankSpec.backend``):
  st    sentence-transformers CrossEncoder (fp32 torch), e.g. BAAI/bge-reranker-v2-m3
  onnx  onnxruntime session on an ONNX file (int8 or fp32): our bge export in
        data/cache/onnx/, or the files Jina ships in its repo (onnx/model_int8.onnx)
  api   hosted rerank API (Cohere / Jina). Without a key it raises RerankUnavailable.
        With ``fallback`` set (e.g. hosted Jina -> local Jina int8 ONNX), any API error,
        a missing key, or a call slower than ``timeout_s`` falls back to that spec.
        Jina's API returns sigmoid scores; they are mapped back to logits so thresholds
        tuned on the local model (abstain_threshold) apply to both.

On this CPU the fp32 bge model costs ~1 s per (query, chunk) pair, so scores are cached
on disk by sha1(backend spec, max_length, query, text).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BATCH_SIZE = 16
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "cache" / "rerank"
ONNX_DIR = ROOT / "data" / "cache" / "onnx"
PHYSICAL_CORES = 8


class RerankUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankSpec:
    model: str = "BAAI/bge-reranker-v2-m3"
    backend: str = "st"  # st | onnx | api
    onnx_file: str | None = None  # path under data/cache/onnx/ or "hub:onnx/model_int8.onnx"
    max_length: int = 384
    provider: str | None = None  # api backend: cohere | jina
    timeout_s: float | None = None  # api backend: deadline for the whole call
    fallback: RerankSpec | None = None  # used when this backend fails / times out

    @property
    def tag(self) -> str:
        return f"{self.backend}:{self.model}:{self.onnx_file or ''}:{self.max_length}"


DEFAULT_SPEC = RerankSpec()


def _threads() -> int:
    return int(os.environ.get("SAHAYI_TORCH_THREADS", PHYSICAL_CORES))


# ------------------------------------------------------------------ backends


class _TorchCrossEncoder:
    def __init__(self, spec: RerankSpec) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        from backend.app.config import configure_hf_cache

        torch.set_num_threads(_threads())
        self.model = CrossEncoder(
            spec.model,
            device="cpu",
            max_length=spec.max_length,
            cache_folder=str(configure_hf_cache()),
            trust_remote_code=spec.model.startswith("jinaai/"),
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        import torch

        with torch.inference_mode():
            scores = self.model.predict(
                [(query, t) for t in texts], batch_size=BATCH_SIZE, show_progress_bar=False
            )
        return [float(s) for s in scores]


class _OnnxCrossEncoder:
    def __init__(self, spec: RerankSpec) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        from backend.app.config import configure_hf_cache

        cache = configure_hf_cache()
        if spec.onnx_file and spec.onnx_file.startswith("hub:"):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(spec.model, spec.onnx_file[4:], cache_dir=cache)
            tok_source = spec.model
        else:
            default_file = f"{spec.model.split('/')[-1]}/model_int8.onnx"
            path = str(ONNX_DIR / (spec.onnx_file or default_file))
            tok_source = str(Path(path).parent)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = _threads()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if os.environ.get("SAHAYI_ORT_SPIN", "0") == "0":
            # idle ORT workers otherwise busy-wait and slow the torch query encoder that
            # runs right after a rerank call on the same cores
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = AutoTokenizer.from_pretrained(tok_source, cache_dir=cache)
        self.max_length = spec.max_length

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[i : i + BATCH_SIZE])
            enc = self.tok(
                [query] * len(batch),
                batch,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {k: v.astype("int64") for k, v in enc.items() if k in self.inputs}
            logits = self.session.run(None, feed)[0]
            out.extend(float(x) for x in logits.reshape(len(batch), -1)[:, 0])
        return out


class _ApiReranker:
    """Hosted rerankers; same contract. Keys: COHERE_API_KEY / JINA_API_KEY."""

    ENDPOINTS = {
        "cohere": ("https://api.cohere.com/v2/rerank", "COHERE_API_KEY", "rerank-v3.5"),
        "jina": (
            "https://api.jina.ai/v1/rerank",
            "JINA_API_KEY",
            "jina-reranker-v2-base-multilingual",
        ),
    }  # fmt: skip

    def __init__(self, spec: RerankSpec) -> None:
        provider = spec.provider or "cohere"
        url, key_name, default_model = self.ENDPOINTS[provider]
        from backend.app.config import get_settings

        key = os.environ.get(key_name) or getattr(get_settings(), key_name.lower(), "")
        if not key:
            raise RerankUnavailable(f"{provider} rerank API disabled: {key_name} is not set")
        self.url, self.key, self.provider = url, key, provider
        self.model = spec.model if "/" not in spec.model else default_model
        self.timeout = spec.timeout_s or 20.0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        import httpx

        r = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "query": query, "documents": list(texts),
                  "top_n": len(texts)},
            timeout=self.timeout,
        )  # fmt: skip
        r.raise_for_status()
        scores = [0.0] * len(texts)
        for item in r.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        if self.provider == "jina":  # sigmoid -> logit, comparable with the ONNX model
            scores = [math.log(max(p, 1e-9) / max(1 - p, 1e-9)) for p in scores]
        return scores


@lru_cache(maxsize=4)
def load_backend(spec: RerankSpec):  # noqa: ANN201 - one of the backend classes
    if spec.backend == "st":
        return _TorchCrossEncoder(spec)
    if spec.backend == "onnx":
        return _OnnxCrossEncoder(spec)
    if spec.backend == "api":
        return _ApiReranker(spec)
    raise ValueError(f"unknown rerank backend {spec.backend!r}")


def load_reranker(model_name: str):  # noqa: ANN201 - kept for callers/tests
    return load_backend(RerankSpec(model=model_name)).model


# --------------------------------------------------------------------- cache


class ScoreCache:
    """Append-only JSONL of {key: score} per backend spec."""

    def __init__(self, tag: str, directory: Path = CACHE_DIR) -> None:
        slug = hashlib.sha1(tag.encode()).hexdigest()[:10]
        name = tag.split(":")[1].split("/")[-1] if ":" in tag else tag.split("/")[-1]
        self.path = directory / f"{name}-{slug}.jsonl"
        self.tag = tag
        self.scores: dict[str, float] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    self.scores.update(json.loads(line))

    def key(self, query: str, text: str) -> str:
        return hashlib.sha1(f"{self.tag}\x00{query}\x00{text}".encode()).hexdigest()

    def add(self, new: dict[str, float]) -> None:
        if not new:
            return
        self.scores.update(new)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(new) + "\n")


@lru_cache(maxsize=8)
def score_cache(tag: str) -> ScoreCache:
    return ScoreCache(tag)


# ------------------------------------------------------------------ interface


_API_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rerank-api")


def _score(spec: RerankSpec, query: str, texts: list[str], info: dict) -> list[float]:
    """Score with ``spec``; on failure or deadline use ``spec.fallback`` if set."""
    t0 = time.perf_counter()
    try:
        backend = load_backend(spec)
        if spec.backend == "api" and spec.timeout_s:
            future = _API_POOL.submit(backend.score, query, texts)
            scores = future.result(timeout=spec.timeout_s)
        else:
            scores = backend.score(query, texts)
        info["backend"] = spec.backend
        return scores
    except Exception as exc:  # noqa: BLE001 - any API/network failure means "fall back"
        if spec.fallback is None:
            raise
        reason = "timeout" if isinstance(exc, FutureTimeout) else type(exc).__name__
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = f"HTTP {status}" if status else reason
        info["fallback"] = f"{reason} after {(time.perf_counter() - t0) * 1000:.0f} ms"
        return _score(spec.fallback, query, texts, info)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, str]],
    spec: RerankSpec | str = DEFAULT_SPEC,
    top_k: int | None = None,
    use_cache: bool = True,
    info: dict | None = None,
) -> list[tuple[str, float]]:
    """Score (chunk_id, text) pairs against the query; return best first.

    ``info`` (optional) receives {"backend": ..., "fallback": reason} for traces.
    Cached scores are keyed by the spec that was asked for; a fallback's scores are
    not cached under the hosted spec.
    """
    info = {} if info is None else info
    if isinstance(spec, str):
        spec = RerankSpec(model=spec)
    if not candidates:
        return []
    use_cache = use_cache and os.environ.get("SAHAYI_RERANK_CACHE", "1") != "0"
    cache = score_cache(spec.tag) if use_cache else None
    keys = [cache.key(query, text) if cache else "" for _, text in candidates]
    missing = [i for i, k in enumerate(keys) if not cache or k not in cache.scores]
    fresh: dict[int, float] = {}
    if not missing:
        info.setdefault("backend", "cache")
    if missing:
        scores = _score(spec, query, [candidates[i][1] for i in missing], info)
        fresh = dict(zip(missing, scores, strict=True))
        if cache and "fallback" not in info:
            cache.add({keys[i]: s for i, s in fresh.items()})
    ranked = sorted(
        (
            (cid, fresh[i] if i in fresh else cache.scores[keys[i]])
            for i, (cid, _) in enumerate(candidates)
        ),
        key=lambda kv: -kv[1],
    )
    return ranked[:top_k] if top_k else ranked
