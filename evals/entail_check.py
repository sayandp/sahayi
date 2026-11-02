"""Measure the optional LLM entailment check before enabling it (spec item 2).

    python evals/entail_check.py RUN.json [--model openai/gpt-oss-20b]

For every judged statement of an answer_eval run, asks the entailment model (one call per
answer, cited excerpts only, as Answerer._entail does) whether the cited excerpts support
it, and compares with the eval judge's labels: how many unsupported statements it would
catch, how many good ones it would wrongly drop, the faithfulness after dropping its
flags, and the added latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.llm.client import LLMError, chat_json  # noqa: E402
from backend.app.llm.prompts import ENTAIL_SYSTEM  # noqa: E402
from backend.app.retrieval.search import load_chunks  # noqa: E402
from evals.answer_eval import Pacer, statements  # noqa: E402
from evals.metrics import percentile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    chunks = load_chunks("section_aware")
    pacer = Pacer()
    rows, lat = [], []
    for it in run["items"]:
        j = it.get("judge") or {}
        if it["abstained"] or not j.get("statements"):
            continue
        stmts = statements({"answer": it["answer"], "schemes": it["schemes"]})
        if len(stmts) != len(j["statements"]):
            continue
        cited = list(dict.fromkeys(c for s in stmts for c in s["citations"]))
        label = {c: f"E{i}" for i, c in enumerate(cited, start=1)}
        body = "\n\n".join(f"[{label[c]}]\n{chunks[c]['text'][:2000]}" for c in cited)
        lines = "\n".join(f"{i}. {s['text']} " + "".join(f"[{label[c]}]" for c in s["citations"])
                          for i, s in enumerate(stmts, start=1))  # fmt: skip
        user = f"Excerpts:\n{body}\n\nSentences:\n{lines}"
        pacer.wait(args.model, len(user) // 2 + 300)
        t0 = time.perf_counter()
        try:
            res = chat_json([{"role": "system", "content": ENTAIL_SYSTEM},
                             {"role": "user", "content": user}], groq_model=args.model)  # fmt: skip
        except LLMError as exc:
            print(f"{it['id']}: {str(exc)[:200]}")
            break
        lat.append((time.perf_counter() - t0) * 1000 - res.throttled_ms)
        pacer.record(args.model, res.tokens_in + res.tokens_out)
        bad = {int(v["i"]) for v in (res.data or {}).get("verdicts", [])
               if isinstance(v, dict) and v.get("supported") is False
               and str(v.get("i", "")).isdigit()}  # fmt: skip
        for i, js in enumerate(j["statements"], start=1):
            rows.append({"id": it["id"], "judge_supported": js["supported"], "flagged": i in bad})
        print(f"  {it['id']}: {len(bad)} flagged of {len(stmts)}", flush=True)
    unsup = [r for r in rows if not r["judge_supported"]]
    flagged = [r for r in rows if r["flagged"]]
    kept = [r for r in rows if not r["flagged"]]
    out = {
        "model": args.model, "run": args.run, "statements": len(rows),
        "judge_unsupported": len(unsup),
        "caught": sum(r["flagged"] for r in unsup),
        "false_drops": sum(r["judge_supported"] for r in flagged),
        "faithfulness_micro_before": round(sum(r["judge_supported"] for r in rows) / len(rows), 3),
        "faithfulness_micro_after": round(sum(r["judge_supported"] for r in kept) / len(kept), 3)
        if kept else None,
        "statements_kept": len(kept),
        "latency_ms": {"p50": round(percentile(lat, 0.5)), "p95": round(percentile(lat, 0.95)),
                       "n": len(lat)},
    }  # fmt: skip
    print(json.dumps(out, indent=1))
    stem = Path(args.run).stem
    (ROOT / "evals" / "results" / f"{stem}_entail.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
