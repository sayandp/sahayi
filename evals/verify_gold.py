"""Check that each gold page actually answers its question (LLM judge).

    python evals/verify_gold.py --sample 20            # random 20 answerable questions
    python evals/verify_gold.py --split heldout        # all held-out questions
    python evals/verify_gold.py --ids q050 h003

The judge sees the question and the text of its gold pages and returns
yes / partial / no with a one-line reason. "no" items need fixing (rephrase the
question or correct the evidence). Writes evals/results/verify_gold.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.llm.client import LLMError, chat_json  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"
PAGES = ROOT / "data" / "processed" / "pages.clean.jsonl"
OUT = ROOT / "evals" / "results" / "verify_gold.md"
JUDGE_MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 6000

PROMPT = """You check a test set for a government-scheme search assistant.
Question (may be Malayalam, English or Manglish = Malayalam in Latin letters):
{question}

Official text that is supposed to answer it:
---
{text}
---
Does the text contain the information needed to answer the question?
Answer JSON: {{"verdict": "yes" | "partial" | "no", "reason": "<one short sentence>"}}"""


def load_pages() -> dict[tuple[str, int], str]:
    with PAGES.open(encoding="utf-8") as f:
        return {(p["doc_id"], p["page"]): p["text"] for p in map(json.loads, f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, help="random N answerable questions")
    ap.add_argument("--split", choices=["main", "heldout"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()
    with GOLD.open(encoding="utf-8") as f:
        qs = [q for q in map(json.loads, f) if q["type"] == "answerable"]
    if args.split:
        qs = [q for q in qs if q.get("split", "main") == args.split]
    if args.ids:
        qs = [q for q in qs if q["id"] in set(args.ids)]
    if args.sample:
        qs = random.Random(args.seed).sample(qs, min(args.sample, len(qs)))
    pages = load_pages()
    rows = []
    for q in qs:
        text = "\n\n".join(pages.get((g["doc_id"], g["page"]), "") for g in q["gold"])[:MAX_CHARS]
        try:
            res = chat_json([{"role": "user", "content": PROMPT.format(question=q["question"], text=text)}],
                            groq_model=args.model)  # fmt: skip
            verdict, reason = res.data.get("verdict", "?"), res.data.get("reason", "")
        except LLMError as exc:
            verdict, reason = "error", str(exc)[:120]
        rows.append((q["id"], q["lang"], q.get("split", "main"), verdict, reason, q["question"]))
        print(f"{q['id']:5} {verdict:8} {reason[:90]}", flush=True)
    counts = {v: sum(r[3] == v for r in rows) for v in ("yes", "partial", "no", "error")}
    lines = [
        f"# Gold verification ({args.model})",
        "",
        f"{len(rows)} questions checked: {counts}",
        "",
        "| id | lang | split | verdict | reason | question |",
        "|---|---|---|---|---|---|",
        *[f"| {i} | {lang} | {sp} | {v} | {r} | {qt} |" for i, lang, sp, v, r, qt in rows],
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
