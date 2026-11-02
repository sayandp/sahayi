"""Tables for evals/results/phase3_summary.md from answer_eval runs.

    python evals/summary_3.py RUN.json [RUN2.json ...] > table.md

Prints: Gate 2 metrics by slice per run, latency / cost, worst faithfulness items and
the content citation validation dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SLICES = ["all", "lang:en", "lang:ml", "lang:manglish", "slice:main", "slice:heldout",
          "slice:eligibility", "slice:no_answer"]  # fmt: skip


def fmt(x: object) -> str:
    return "-" if x is None else str(x)


def gate_table(runs: list[dict]) -> list[str]:
    out = ["| generator | slice | n | answered | faithfulness | citation acc. | relevancy "
           "| context hit@5 | language match |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in runs:
        for sl in SLICES:
            b = r["summary"]["by_slice"][sl]
            if not b["n"]:
                continue
            f, c = b["faithfulness"], b["citation_accuracy"]
            out.append(f"| {r['model']} | {sl} | {b['n']} | {b['answered']} | "
                       f"{'**' + str(f) + '**' if sl == 'all' else fmt(f)} | "
                       f"{'**' + str(c) + '**' if sl == 'all' else fmt(c)} | "
                       f"{fmt(b['answer_relevancy'])} | {fmt(b['context_hit@5'])} | "
                       f"{fmt(b['language_match'])} |")  # fmt: skip
    return out


def other_table(runs: list[dict]) -> list[str]:
    out = ["| generator | eligibility acc. (n) | card found | abstain recall (no-answer n) "
           "| abstain precision | false-abstain rate | TTFT p50 / p95 ms | full p50 / p95 ms "
           "| ₹ / query mean (p95) | tokens in / out |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in runs:
        s = r["summary"]
        e, a, lat, cost, tok = (s["eligibility"], s["abstention"], s["latency_ms"],
                                s["cost_inr"], s["tokens"])  # fmt: skip
        live = (f"{fmt(lat['ttft_p50'])} / {fmt(lat['ttft_p95'])} (n={lat['n_live']})"
                if lat["n_live"] else "not live")  # fmt: skip
        full = f"{fmt(lat['full_p50'])} / {fmt(lat['full_p95'])}" if lat["n_live"] else "-"
        out.append(f"| {r['model']} | {fmt(e['accuracy'])} ({e['n']}) | {fmt(e['card_found'])} | "
                   f"{fmt(a['recall'])} ({a['no_answer_n']}) | {fmt(a['precision'])} | "
                   f"{fmt(a['false_abstain_rate'])} | {live} | {full} | "
                   f"{fmt(cost['mean'])} ({fmt(cost['p95'])}) | {fmt(tok['in_mean'])} / "
                   f"{fmt(tok['out_mean'])} |")  # fmt: skip
    return out


def worst(run: dict, k: int = 5) -> list[str]:
    rows = []
    for it in run["items"]:
        j = it.get("judge") or {}
        st = j.get("statements")
        if not st:
            continue
        bad = [s for s in st if not s["supported"] or not s["cited_ok"]]
        if bad:
            rows.append((sum(s["supported"] for s in st) / len(st), it, bad))
    rows.sort(key=lambda r: r[0])
    out = [
        "| id | lang | faithfulness | unsupported / mis-cited statement(s) |",
        "|---|---|---|---|",
    ]
    for score, it, bad in rows[:k]:
        texts = " · ".join(
            f"{'unsupported' if not s['supported'] else 'mis-cited'}: {s['text'][:160]}"
            for s in bad[:2]
        )
        out.append(f"| {it['id']} | {it['lang']} | {score:.2f} | {texts.replace('|', '/')} |")
    if len(out) == 2:
        out.append("| - | - | - | none: every judged statement was supported and correctly cited |")
    return out


def drops(run: dict, k: int = 12) -> list[str]:
    out = ["| id | lang | reason | detail | dropped text |", "|---|---|---|---|---|"]
    n = 0
    for it in run["items"]:
        for d in it["dropped"]:
            if n < k:
                out.append(f"| {it['id']} | {it['lang']} | {d['reason']} | {d['detail'] or '-'} | "
                           f"{d['text'][:150].replace('|', '/')} |")  # fmt: skip
            n += 1
    out.append(f"\n{n} statements dropped in {sum(bool(i['dropped']) for i in run['items'])} of "
               f"{len(run['items'])} items; reasons: {run['summary']['validation']['dropped_by_reason']}")  # fmt: skip
    return out


def main() -> int:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:]]
    print("\n".join(gate_table(runs)), "\n")
    print("\n".join(other_table(runs)), "\n")
    for r in runs:
        print(f"### Worst faithfulness: {r['model']}\n")
        print("\n".join(worst(r)), "\n")
        print(f"### Validation drops: {r['model']}\n")
        print("\n".join(drops(r)), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
