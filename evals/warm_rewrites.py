"""Fill the rewrite cache for eval questions, waiting out Groq's daily token window.

    python evals/warm_rewrites.py --eligibility [--gold]

Rewrites are cached by (prompt version, model, question) in data/cache/rewrite, so a
non-live answer_eval run then uses exactly the default config's rewrite model instead
of silently falling back to "no rewrite" when the quota is exhausted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.query import DEFAULT_REWRITE_MODEL, _cache_path, rewrite  # noqa: E402
from evals.answer_eval import GOLD, load_eligibility  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligibility", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--wait", type=float, default=300, help="seconds between failed tries")
    args = ap.parse_args()
    qs: list[str] = []
    if args.eligibility:
        qs += [c["question"] for c in load_eligibility()]
    if args.gold:
        qs += [json.loads(line)["question"] for line in GOLD.open(encoding="utf-8")]
    todo = [q for q in qs if not _cache_path(q, DEFAULT_REWRITE_MODEL).exists()]
    print(f"{len(qs)} questions, {len(todo)} rewrites to fetch", flush=True)
    for i, q in enumerate(todo, start=1):
        while True:
            rw = rewrite(q)
            if rw.provider not in ("none",):
                print(f"  [{i}/{len(todo)}] ok ({rw.provider}, {rw.latency_ms:.0f} ms)", flush=True)
                break
            print(f"  [{i}/{len(todo)}] {str(rw.error)[:120]}; retry in {args.wait:.0f}s",
                  flush=True)  # fmt: skip
            time.sleep(args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
