"""Micro-benchmark reranker backends on one query's candidates (no cache).

    python evals/bench_rerankers.py [--pool 10]

Prints load time, scoring time for the pool, peak RSS, and the Spearman agreement of
each backend's ranking with the fp32 bge-reranker-v2-m3 reference.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.rerank import RerankSpec, load_backend  # noqa: E402
from backend.app.retrieval.search import RetrievalConfig, Retriever  # noqa: E402

SPECS = {
    "bge fp32 (st)": RerankSpec("BAAI/bge-reranker-v2-m3", "st", None, 384),
    "bge int8 (onnx)": RerankSpec("BAAI/bge-reranker-v2-m3", "onnx", "bge-reranker-v2-m3/model_int8.onnx", 384),
    # jina's trust_remote_code torch model does not load on transformers 5.x; use its fp32 ONNX
    "jina fp32 (onnx)": RerankSpec("jinaai/jina-reranker-v2-base-multilingual", "onnx", "hub:onnx/model.onnx", 384),
    "jina int8 (onnx)": RerankSpec("jinaai/jina-reranker-v2-base-multilingual", "onnx", "hub:onnx/model_int8.onnx", 384),
    "jina int8 (onnx) len256": RerankSpec("jinaai/jina-reranker-v2-base-multilingual", "onnx", "hub:onnx/model_int8.onnx", 256),
}  # fmt: skip


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> list[int]:
        order = sorted(range(len(x)), key=lambda i: -x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb, strict=True)) / (n * (n * n - 1))


def main() -> None:
    import psutil

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--query", default="വിധവ പെൻഷന് അപേക്ഷിക്കാനുള്ള വരുമാന പരിധി എത്ര?")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    r = Retriever(RetrievalConfig(mode="hybrid", rerank=False))
    texts = [r.chunks[c]["text"] for c, _ in r.candidates(args.query)[: args.pool]]
    r.close()
    proc = psutil.Process()
    reference: list[float] | None = None
    for name, spec in SPECS.items():
        if args.only and not any(o in name for o in args.only):
            continue
        t0 = time.perf_counter()
        backend = load_backend(spec)
        load_s = time.perf_counter() - t0
        backend.score(args.query, texts[:2])  # warm-up
        t0 = time.perf_counter()
        scores = backend.score(args.query, texts)
        run_s = time.perf_counter() - t0
        reference = reference or scores
        rss = proc.memory_info().rss / 1e9
        peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 1e9
        print(f"## {name:18} load {load_s:5.1f}s | {len(texts)} pairs {run_s:5.2f}s "
              f"({run_s / len(texts):.2f}s/pair) | RSS {rss:.1f} GB, peak {peak:.1f} GB | "
              f"spearman vs bge fp32 {spearman(reference, scores):+.2f}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
