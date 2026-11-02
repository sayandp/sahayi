"""How often the speculative top-5 (original query only, reranked while the rewrite call
is in flight) equals the final top-5, by language. Decides whether starting answer
generation speculatively pays off.

    python evals/spec_match.py [--config default]

Uses the rewrite / rerank caches (rankings, not latency, are measured).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.answer import answer_lang, guess_lang  # noqa: E402
from backend.app.retrieval.search import Retriever, load_named_config  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    args = ap.parse_args()
    r = Retriever(load_named_config(args.config))
    spec: list[str] = []
    r.on_speculative = lambda ids: spec.__setitem__(slice(None), ids[:5])
    stats: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    try:
        for line in GOLD.open(encoding="utf-8"):
            q = json.loads(line)
            hits = r.search(q["question"])
            final = [h.chunk_id for h in hits[:5]]
            lang_ok = guess_lang(q["question"]) == answer_lang(r.last_trace.rewrite_lang,
                                                               q["question"])  # fmt: skip
            stats[q["lang"]].append((set(spec) == set(final), lang_ok))
    finally:
        r.close()
    out = {}
    for lang, rows in sorted(stats.items()) + [("all", [x for v in stats.values() for x in v])]:
        out[lang] = {"n": len(rows),
                     "same_top5": round(sum(a for a, _ in rows) / len(rows), 3),
                     "lang_guess_ok": round(sum(b for _, b in rows) / len(rows), 3),
                     "usable": round(sum(a and b for a, b in rows) / len(rows), 3)}  # fmt: skip
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
