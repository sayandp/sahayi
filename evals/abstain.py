"""Pick the abstain threshold on the top-1 retrieval/rerank score (Phase 2b, step 5).

    python evals/abstain.py --result evals/results/<stamp>_<config>.json [--save]

Positive class = abstain (no_answer questions). A query is refused when its top-1
score < threshold. Chooses the threshold maximising F1 over the midpoints between
sorted scores, reports precision / recall for both classes, and how many answerable
questions would be wrongly refused (split by whether retrieval had found the gold).
--save writes abstain_threshold into evals/configs.yaml for that config.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "evals" / "configs.yaml"


def best_threshold(scores: list[tuple[float, bool]]) -> dict:
    """scores: (top-1 score, is_no_answer). Returns metrics at the F1-optimal cut."""
    values = sorted({s for s, _ in scores})
    cuts = [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] or values
    best: dict = {}
    for t in cuts:
        tp = sum(s < t and neg for s, neg in scores)
        fp = sum(s < t and not neg for s, neg in scores)
        fn = sum(s >= t and neg for s, neg in scores)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best.get("f1", -1):
            best = {"threshold": round(t, 4), "f1": round(f1, 3),
                    "abstain_precision": round(precision, 3), "abstain_recall": round(recall, 3),
                    "wrongly_refused": fp, "missed_no_answer": fn}  # fmt: skip
    return best


def save_threshold(config: str, threshold: float) -> None:
    text = CONFIGS.read_text(encoding="utf-8")
    block = re.search(rf"^{re.escape(config)}:\n((?:  .*\n?)*)", text, re.M)
    if not block:
        raise SystemExit(f"config {config!r} not found in {CONFIGS}")
    body = re.sub(r"^  abstain_threshold:.*\n?", "", block.group(1), flags=re.M)
    body = body.rstrip("\n") + f"\n  abstain_threshold: {threshold}\n"
    CONFIGS.write_text(text[: block.start(1)] + body + text[block.end(1) :], encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    rows = result["questions"]
    field = (
        "top_rerank_score"
        if any(r.get("top_rerank_score") is not None for r in rows)
        else "top_score"
    )
    scored = [(r[field], r["type"] == "no_answer") for r in rows if r[field] is not None]
    best = best_threshold(scored)
    t = best["threshold"]
    answerable = [r for r in rows if r["type"] == "answerable" and r[field] is not None]
    refused = [r for r in answerable if r[field] < t]
    report = {
        "config": result["config"]["name"],
        "score_field": field,
        "n_answerable": len(answerable),
        "n_no_answer": sum(r["type"] == "no_answer" for r in rows),
        **best,
        "answer_recall": round(1 - len(refused) / len(answerable), 3) if answerable else None,
        "refused_with_gold_in_top5": sum(1 for r in refused if r["rank"] and r["rank"] <= 5),
        "refused_with_gold_missing": sum(1 for r in refused if not r["rank"] or r["rank"] > 5),
    }
    print(json.dumps(report, indent=1))
    if args.save:
        save_threshold(result["config"]["name"], t)
        print(f"saved abstain_threshold: {t} for {result['config']['name']} in {CONFIGS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
