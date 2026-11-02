"""Build BM25 + Qdrant indexes from data/processed/chunks.{strategy}.jsonl.

    python -m ingest.build_index                       # section_aware, bge-m3
    python -m ingest.build_index --strategy all --embed-model intfloat/multilingual-e5-base

Embeddings are cached in data/cache/emb/, so rebuilding is cheap after the first run.
"""

from __future__ import annotations

import argparse
import time

from backend.app.retrieval.bm25 import BM25Index, TokenizerConfig
from backend.app.retrieval.dense import DenseIndex, EmbeddingCache, embed_passages
from backend.app.retrieval.search import INDEX_DIR, RetrievalConfig, index_text, load_chunks
from ingest.chunk import STRATEGIES
from ingest.common import DATA_DIR

EMB_CACHE_DIR = DATA_DIR / "cache" / "emb"


def build(strategy: str, embed_model: str, dense: bool = True, context: bool = False) -> None:
    chunks = load_chunks(strategy)
    ids = list(chunks)
    texts = [index_text(chunks[i], context) for i in ids]
    for stem in (True, False):
        cfg = RetrievalConfig(strategy=strategy, embed_model=embed_model, bm25_stem=stem,
                              index_context=context)  # fmt: skip
        BM25Index(ids, texts, TokenizerConfig(stem=stem)).save(cfg.bm25_path)
    print(f"[{strategy}] BM25: {len(ids)} chunks -> bm25.{strategy}.pkl (+ .nostem)")
    if not dense:
        return
    t0 = time.time()
    vectors = embed_passages(texts, embed_model, EmbeddingCache(EMB_CACHE_DIR, embed_model))
    index = DenseIndex(INDEX_DIR / "qdrant", cfg.collection)
    index.build(ids, vectors)
    index.close()
    print(f"[{strategy}] dense: {vectors.shape} -> {cfg.collection} ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="section_aware", choices=[*STRATEGIES, "all"])
    ap.add_argument("--embed-model", default=RetrievalConfig().embed_model)
    ap.add_argument("--no-dense", action="store_true", help="BM25 only")
    ap.add_argument("--context", action="store_true", help="contextual chunk headers")
    ap.add_argument(
        "--embed-only",
        action="store_true",
        help="fill the embedding cache without touching Qdrant (it allows one process)",
    )
    args = ap.parse_args()
    for strategy in STRATEGIES if args.strategy == "all" else (args.strategy,):
        if args.embed_only:
            chunks = load_chunks(strategy)
            texts = [index_text(c, args.context) for c in chunks.values()]
            embed_passages(texts, args.embed_model, EmbeddingCache(EMB_CACHE_DIR, args.embed_model))
            print(f"[{strategy}] cached {len(texts)} embeddings (context={args.context})")
            continue
        build(strategy, args.embed_model, dense=not args.no_dense, context=args.context)


if __name__ == "__main__":
    main()
