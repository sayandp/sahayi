"""Live end-to-end retrieval latency: real rewrite call, no caches (Phase 2b).

    python evals/bench_e2e.py --config <name> [--n 40] [--gap 6] [--threads 8]

Disables the rewrite and rerank score caches, samples questions across languages,
paces calls to stay under the Groq rate limit, and reports p50/p95 of wall-clock
retrieval time plus the per-stage breakdown and peak RAM. --threads 2 approximates a
2-vCPU HF Spaces CPU instance (network latency is the local one).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gap", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.environ["SAHAYI_REWRITE_CACHE"] = "0"
    os.environ["SAHAYI_RERANK_CACHE"] = "0"
    os.environ["SAHAYI_TORCH_THREADS"] = str(args.threads)

    import psutil

    from backend.app.retrieval.search import Retriever
    from evals.metrics import percentile
    from evals.run_eval import TIMING_KEYS, load_config, load_gold

    qs = load_gold()
    rng = random.Random(9)
    by_lang: dict[str, list[dict]] = {}
    for q in qs:
        by_lang.setdefault(q["lang"], []).append(q)
    sample = []
    while len(sample) < args.n and any(by_lang.values()):
        for lang in sorted(by_lang):
            if by_lang[lang] and len(sample) < args.n:
                sample.append(by_lang[lang].pop(rng.randrange(len(by_lang[lang]))))
    retriever = Retriever(load_config(args.config))
    retriever.search("warm-up")
    rows = []
    try:
        for q in sample:
            retriever.search(q["question"])
            tr = retriever.last_trace
            rows.append({"lang": q["lang"], "total_ms": tr.total_ms, "throttled": tr.rewrite_provider == "none",
                         **{k: getattr(tr, k) for k in TIMING_KEYS}})  # fmt: skip
            time.sleep(args.gap)
    finally:
        retriever.close()
    mem = psutil.Process().memory_info()
    totals = [r["total_ms"] for r in rows]
    out = {
        "config": args.config, "threads": args.threads, "n": len(rows),
        "rewrite_failures": sum(r["throttled"] for r in rows),
        "total_ms": {"p50": round(percentile(totals, 0.5)), "p95": round(percentile(totals, 0.95)),
                     "max": round(max(totals))},
        "by_lang_p95": {lang: round(percentile([r["total_ms"] for r in rows if r["lang"] == lang], 0.95))
                        for lang in sorted({r["lang"] for r in rows})},
        "stages_p50": {k: round(percentile([r[k] for r in rows], 0.5)) for k in TIMING_KEYS},
        "stages_p95": {k: round(percentile([r[k] for r in rows], 0.95)) for k in TIMING_KEYS},
        "peak_ram_gb": round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2),
    }  # fmt: skip
    print(json.dumps(out, indent=1))
    res = ROOT / "evals" / "results" / f"e2e_{args.config}_t{args.threads}.json"
    res.write_text(json.dumps({**out, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
