"""Retrieval-only benchmark of a deploy config on the gold set. No generation, no LLM calls.

    python deploy/bench_retrieval.py --config deploy_local --gold <gold.jsonl> --cpus 2 --out x.json

Run it from the Space folder. Rewrites come only from data/cache/rewrite (cached, so they take
no time: live rewrites would add waiting); the rerank cache is off. Reports any cache miss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.serve import configure_threads, pin_cpus  # noqa: E402


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cpus", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cpus:
        pin_cpus(args.cpus)
    threads = configure_threads()
    os.environ["SAHAYI_RERANK_CACHE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import psutil

    from backend.app.answer import Answerer
    from backend.app.retrieval.search import Retriever, load_named_config

    retriever = Retriever(load_named_config(args.config, ROOT / "deploy" / "configs.yaml"))
    Answerer(retriever, model="none").warm_up()
    cold_s = time.perf_counter() - T0
    gold = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines()]
    rows = []
    for q in gold:
        pages = {(g["doc_id"], g["page"]) for g in q.get("gold") or []}
        hits = retriever.search(q["question"])
        tr = retriever.last_trace
        got = {(h.chunk["doc_id"], h.chunk["page"]) for h in hits[:5]}
        rows.append({
            "id": q["id"], "lang": q["lang"], "split": q.get("split", "main"), "type": q["type"],
            "hit5": bool(pages & got) if pages else None, "ms": tr.total_ms,
            "rewrite_missing": bool(retriever.last_rewrite and retriever.last_rewrite.error),
            "backend": tr.rerank_backend_used, "fallback": tr.rerank_fallback,
        })  # fmt: skip
    retriever.close()

    def hit(rs: list[dict]) -> float:
        rs = [r for r in rs if r["hit5"] is not None]
        return round(sum(r["hit5"] for r in rs) / len(rs), 3)

    ans = [r for r in rows if r["type"] == "answerable"]
    mem = psutil.Process().memory_info()
    fallbacks: dict[str, int] = {}
    for r in rows:
        if r["fallback"]:
            key = r["fallback"].split(" after")[0]
            fallbacks[key] = fallbacks.get(key, 0) + 1
    out = {
        "config": args.config, "threads": threads, "affinity": psutil.Process().cpu_affinity(),
        "n": len(rows), "cold_start_s": round(cold_s, 1),
        "hit@5_main": hit([r for r in ans if r["split"] == "main"]), "hit@5_all": hit(ans),
        **{f"hit@5_{lang}": hit([r for r in ans if r["lang"] == lang])
           for lang in ("en", "ml", "manglish")},
        "p50_ms": pct([r["ms"] for r in rows], 0.5), "p95_ms": pct([r["ms"] for r in rows], 0.95),
        "peak_ram_gb": round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2),
        "rewrite_cache_misses": sum(r["rewrite_missing"] for r in rows),
        "fallback_queries": fallbacks,
    }  # fmt: skip
    Path(args.out).write_text(
        json.dumps({"summary": out, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
