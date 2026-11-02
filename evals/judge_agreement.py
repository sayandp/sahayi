"""Re-judge a sample of a finished answer_eval run with a second judge; report agreement.

    python evals/judge_agreement.py evals/results/<run>.json --second openai/gpt-oss-20b

Same judge prompt and cited excerpts, so only the judge model differs. The second judge must
not be the run's generator. Writes evals/results/<run>_judge_agreement.json.
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

from backend.app.retrieval.search import load_chunks  # noqa: E402
from evals.answer_eval import Pacer, judge  # noqa: E402

MAX_QUOTA_WAITS = 96


def kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--second", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    if args.second == run["model"]:
        raise SystemExit("the second judge must not be the generator")
    rows = [r for r in run["items"] if (r.get("judge") or {}).get("statements")]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    chunks = load_chunks("section_aware")
    pacer = Pacer()
    pairs: dict[str, tuple[list, list]] = {"supported": ([], []), "cited_ok": ([], [])}
    per_item, disagreements = [], []
    for r in sample:
        for _ in range(MAX_QUOTA_WAITS):
            j = judge(r, r, [chunks[c] for c in r["context"]], args.second, pacer, True)
            if not j or "429" not in j.get("error", ""):
                break
            print(f"{r['id']}: quota, waiting an hour", flush=True)  # Gemini free tier: 20/day
            time.sleep(3600)
        if not j or "error" in j:
            per_item.append({"id": r["id"], "error": (j or {}).get("error")})
            print(r["id"], "error", (j or {}).get("error"), flush=True)
            continue
        first = r["judge"]["statements"]
        for field, (a, b) in pairs.items():
            a.extend(x[field] for x in first)
            b.extend(x[field] for x in j["statements"])
        for x, y in zip(first, j["statements"], strict=True):
            if x["supported"] != y["supported"]:
                disagreements.append({"id": r["id"], "text": x["text"][:200],
                                      "first": x["supported"], "second": y["supported"]})  # fmt: skip
        per_item.append({"id": r["id"], "statements": len(first)})
        print(r["id"], "ok", flush=True)
    out = {"run": Path(args.run).name, "first_judge": run["judge"], "second_judge": args.second,
           "answers": len(sample), "disagreements": disagreements, "items": per_item}  # fmt: skip
    for field, (a, b) in pairs.items():
        out[field] = {
            "statements": len(a),
            "agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 3)
            if a else None,
            "kappa": kappa(a, b),
            "rate_first_judge": round(sum(a) / len(a), 3) if a else None,
            "rate_second_judge": round(sum(b) / len(b), 3) if b else None,
        }  # fmt: skip
    path = Path(args.run).with_name(Path(args.run).stem + "_judge_agreement.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("supported", "cited_ok")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
