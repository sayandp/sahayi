"""Judge the saved answers of an answer_eval run again with the current judge.

    python evals/rejudge.py RUN.json --only q118,h017 [--judge openai/gpt-oss-20b]

Makes before/after prompt comparisons fair: both sides get the same judge and judge prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.search import load_chunks  # noqa: E402
from evals.answer_eval import judge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--only", required=True)
    ap.add_argument("--judge", default="openai/gpt-oss-20b")
    args = ap.parse_args()
    rows = json.loads(Path(args.run).read_text(encoding="utf-8"))["items"]
    chunks = load_chunks("section_aware")
    tokens: dict = {}
    for r in rows:
        if r["id"] not in args.only.split(",") or r["abstained"]:
            continue
        j = judge(r, r, [chunks[c] for c in r["context"]], args.judge, None, True, tokens)
        st = j.get("statements") or []
        ok, cite = sum(s["supported"] for s in st), sum(s["cited_ok"] for s in st)
        print(f"{r['id']}: supported {ok}/{len(st)}, cited_ok {cite}/{len(st)}", flush=True)
        for s in st:
            if not s["supported"]:
                print(f"    unsupported: {s['text'][:170]}")
    print("tokens", json.dumps(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
