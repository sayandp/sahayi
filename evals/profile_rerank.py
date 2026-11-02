"""Profile cross-encoder reranking for one query (Phase 2b, step 1).

    python evals/profile_rerank.py [--model BAAI/bge-reranker-v2-m3] [--query "..."]

Measures tokenisation vs forward pass, batching (one call vs batches of 8 vs one by
one), and the effect of max_length / inference_mode / thread count.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import configure_hf_cache  # noqa: E402
from backend.app.retrieval.search import RetrievalConfig, Retriever  # noqa: E402


def timed(fn, repeat: int = 2) -> float:  # noqa: ANN001
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--query", default="വിധവ പെൻഷന് അപേക്ഷിക്കാനുള്ള വരുമാന പരിധി എത്ര?")
    ap.add_argument("--pool", type=int, default=20)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache = configure_hf_cache()
    r = Retriever(RetrievalConfig(mode="hybrid", rerank=False))
    cands = r.candidates(args.query)[: args.pool]
    texts = [r.chunks[c]["text"] for c, _ in cands]
    r.close()

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=cache)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, cache_dir=cache)
    model.eval()
    full = [len(tok(args.query, t)["input_ids"]) for t in texts]
    print(f"model {args.model}: dtype {next(model.parameters()).dtype}, "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")  # fmt: skip
    print(f"torch threads {torch.get_num_threads()} (physical cores 8, logical 16)")
    print(f"pair lengths (subword tokens, untruncated): median {statistics.median(full):.0f}, "
          f"max {max(full)}, >512: {sum(n > 512 for n in full)}/{len(full)}")  # fmt: skip

    def tokenize(max_len: int):  # noqa: ANN202
        return tok([args.query] * len(texts), texts, padding=True, truncation="only_second",
                   max_length=max_len, return_tensors="pt")  # fmt: skip

    for max_len in (512, 384):
        enc = tokenize(max_len)
        t_tok = timed(lambda m=max_len: tokenize(m))
        with torch.inference_mode():
            t_one_batch = timed(lambda e=enc: model(**e).logits, repeat=1)

            def batches(size: int, e=enc) -> None:  # noqa: ANN001
                for i in range(0, len(texts), size):
                    _ = model(**{k: v[i : i + size] for k, v in e.items()}).logits

            t_b8 = timed(lambda: batches(8), repeat=1)
            t_single = timed(lambda: batches(1), repeat=1)
        print(f"max_length {max_len}: tokenise {t_tok * 1000:.0f} ms | forward: one batch of "
              f"{len(texts)} {t_one_batch:.1f}s, batches of 8 {t_b8:.1f}s, one-by-one "
              f"{t_single:.1f}s (padded to {enc['input_ids'].shape[1]})")  # fmt: skip

    for threads in (8, 16):
        torch.set_num_threads(threads)
        enc = tokenize(384)
        with torch.inference_mode():
            t = timed(lambda e=enc: model(**e).logits, repeat=1)
        print(f"threads {threads}: one batch, max_length 384: {t:.1f}s")
    torch.set_num_threads(8)


if __name__ == "__main__":
    main()
