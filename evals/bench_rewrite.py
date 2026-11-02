"""Live query-rewrite latency, paced to stay under the provider rate limit.

    python evals/bench_rewrite.py [--n 20] [--gap 6]

Bypasses the rewrite cache. Reports request latency of successful calls (what one user
waits) and, separately, time lost to rate-limit retries during the run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.query import DEFAULT_REWRITE_MODEL, rewrite  # noqa: E402
from evals.metrics import percentile  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gap", type=float, default=6.0, help="seconds between calls")
    ap.add_argument("--model", default=DEFAULT_REWRITE_MODEL)
    args = ap.parse_args()
    with GOLD.open(encoding="utf-8") as f:
        qs = [q["question"] for q in map(json.loads, f)]
    sample = random.Random(5).sample(qs, args.n)
    lat, thr, fails = [], [], 0
    for q in sample:
        rw = rewrite(q, model=args.model, use_cache=False)
        if rw.provider == "none":
            fails += 1
        else:
            lat.append(rw.latency_ms)
            thr.append(rw.throttled_ms)
        time.sleep(args.gap)
    print(json.dumps({"model": args.model, "n": len(lat), "failed": fails,
                      "latency_ms": {"p50": round(percentile(lat, 0.5)), "p95": round(percentile(lat, 0.95)),
                                     "max": round(max(lat)) if lat else None},
                      "throttled_ms_total": round(sum(thr))}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
