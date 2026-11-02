"""Gate 2 side by side for two generators, from answer_eval run files.

    python evals/gate2_compare.py <qwen_run>.json <gptoss_run>.json [--live <qwen_live>.json ...]

TTFT comes only from live runs (--live): full runs use cached rewrites and rerank scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(run: dict, live: list[dict]) -> list[str]:
    s = run["summary"]
    g, t, e, a = s["gate2"], s["truncation"], s["eligibility"], s["abstention"]
    lat = next((x["summary"]["latency_ms"] for x in live if x["model"] == run["model"]), None)
    ttft = f"{lat['ttft_p50']} / {lat['ttft_p95']} ms (n={lat['n_live']})" if lat else "pending"
    return [
        run["model"], str(len(run["items"])), str(g["answerable_gold_n"]),
        "yes" if g["counts"] else "not yet", str(g["faithfulness"]), str(g["citation_accuracy"]),
        f"{e['accuracy']} ({e['n']})", f"{a['recall']} / {a['precision']}",
        f"{t['failure_rate']} ({t['failures']} failed, {t['compact_retries']} retries)",
        ttft, str(s["cost_inr"]["mean"]),
    ]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--live", nargs="*", default=[])
    args = ap.parse_args()
    live = [load(p) for p in args.live]
    head = ["generator", "items", "answerable gold n", "n ≥ 120", "faithfulness", "citation acc.",
            "eligibility acc. (n)", "abstain recall / precision", "truncation rate",
            "TTFT p50 / p95", "₹ / query"]  # fmt: skip
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for path in args.runs:
        print("| " + " | ".join(row(load(path), live)) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
