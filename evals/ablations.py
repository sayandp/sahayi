"""Chunking x retrieval x reranker grid: which part earns its place. No LLM calls.

    python evals/ablations.py

Rewrite and contextual headers stay off, so each row shows only the parts named in it; the
rewrite and header comparisons are in evals/results/phase2b_summary.md. Uses its own index folder
(data/index/qdrant_ablate), because local Qdrant locks the main one while an eval runs. Runs at
low priority. Writes evals/results/ablation_chunking.md and the per-config JSON next to it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.dense import DenseIndex, EmbeddingCache, embed_passages  # noqa: E402
from backend.app.retrieval.search import index_text, load_chunks  # noqa: E402
from evals.run_eval import evaluate, load_config, load_gold  # noqa: E402
from ingest.common import DATA_DIR  # noqa: E402

RESULTS = ROOT / "evals" / "results"
INDEX = DATA_DIR / "index" / "qdrant_ablate"
EMB_CACHE = DATA_DIR / "cache" / "emb"
STRATEGIES = ["fixed", "recursive", "section_aware"]
MODES = ["bm25", "dense", "hybrid"]
HEAD = ["chunking", "retrieval", "rerank", "hit@1", "hit@5", "hit@10", "MRR@10", "EN", "ML",
        "Manglish", "p95 ms"]  # fmt: skip


def build_dense(strategy: str, config) -> None:  # noqa: ANN001 - RetrievalConfig
    chunks = load_chunks(strategy)
    texts = [index_text(c, False) for c in chunks.values()]
    vectors = embed_passages(
        texts, config.embed_model, EmbeddingCache(EMB_CACHE, config.embed_model)
    )
    index = DenseIndex(INDEX, config.collection)
    index.build(list(chunks), vectors)
    index.close()


def grid() -> list:
    plain = {
        "rewrite": False, "overlap_rewrite": False, "index_context": False, "skip_rewrite_kinds": [],
    }  # fmt: skip
    configs = []
    for strategy in STRATEGIES:
        for mode in MODES:
            configs.append(replace(load_config(mode), name=f"abl_{strategy}_{mode}",
                                   strategy=strategy, **plain))  # fmt: skip
        configs.append(replace(load_config("default"), name=f"abl_{strategy}_hybrid_jina",
                               strategy=strategy, abstain_threshold=None, **plain))  # fmt: skip
    return configs


def row(config, s: dict) -> list[object]:  # noqa: ANN001 - RetrievalConfig
    a = s["all"]
    return [config.strategy, config.mode, "Jina int8 top-6" if config.rerank else "none",
            a["hit@1"], f"**{a['hit@5']}**", a["hit@10"], a["mrr@10"], s["en"]["hit@5"],
            s["ml"]["hit@5"], s["manglish"]["hit@5"], round(s["latency_ms"]["p95"])]  # fmt: skip


def main() -> int:
    import psutil

    sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.setdefault("SAHAYI_TORCH_THREADS", "4")
    os.environ["SAHAYI_RERANK_CACHE"] = "0"
    questions = load_gold(None)
    configs = grid()
    for strategy in STRATEGIES:
        build_dense(
            strategy, next(c for c in configs if c.strategy == strategy and c.mode == "dense")
        )
    rows = []
    for config in configs:
        print(f"== {config.name}", flush=True)
        result = evaluate(config, questions, INDEX)
        (RESULTS / f"ablation_{config.name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        rows.append(row(config, result["summary"]))
    n = sum(q["type"] == "answerable" for q in questions)
    lines = [f"# Ablation: chunking x retrieval x rerank ({n} answerable gold questions)", "",
             "No rewrite and no contextual headers, so these sit below the default's 0.897. "
             "Latency was measured while other work was running, so only hit rates compare fairly.",
             "", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]  # fmt: skip
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    (RESULTS / "ablation_chunking.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
