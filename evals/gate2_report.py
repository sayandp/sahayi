"""Write the Gate 2 report: both generators side by side, from the run files.

    python evals/gate2_report.py evals/results/<run1>.json evals/results/<run2>.json

Writes evals/results/gate2_final.md. Works on partial runs too and then says so plainly: the
gate counts only with at least 120 answerable gold questions, and the primary is picked only
when every run has finished.

Primary rule, decided before the results: gpt-oss-20b if it passes (cheaper, and Groq lets it
answer far more per minute); otherwise qwen if it passes; otherwise none is declared.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.gate2_finish import live_tag  # noqa: E402
from evals.run_until_done import TOTAL_ITEMS, finished  # noqa: E402

RESULTS = ROOT / "evals" / "results"
SLICES = ["main", "heldout", "eligibility", "no_answer"]
LANGS = ["en", "ml", "manglish"]
PREFERENCE = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table(head: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join("–" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def live_run(model: str) -> dict | None:
    files = sorted(RESULTS.glob(f"*_answers_{live_tag(model)}.json"))
    return load(files[-1]) if files else None


def agreement_of(path: Path) -> dict | None:
    file = path.with_name(path.stem + "_judge_agreement.json")
    return load(file) if file.exists() else None


def pick_primary(runs: list[dict]) -> tuple[str | None, str]:
    if not all(finished(r) for r in runs):
        return None, "not picked: the runs have not finished"
    passing = {r["model"] for r in runs if r["summary"]["gate2"]["pass"]}
    for model in PREFERENCE:
        if model in passing:
            return model, "first model in the agreed order that passes Gate 2"
    return None, "not picked: no generator passes Gate 2"


def report(paths: list[Path]) -> str:
    runs = [load(p) for p in paths]
    names = [r["model"] for r in runs]
    done = all(finished(r) for r in runs)
    out = ["# Gate 2 report", ""]
    if not done:
        out += ["**INTERIM, does not count.** Progress: "
                + ", ".join(f"{r['model']} {len(r['items'])}/{TOTAL_ITEMS}" for r in runs), ""]  # fmt: skip
    primary, why = pick_primary(runs)
    out += [f"**Production primary: {primary or 'none'}** ({why}).", ""]

    def row(label: str, get) -> list[object]:  # noqa: ANN001
        return [label, *(get(r["summary"]) for r in runs)]

    out += ["## Side by side", "", table(["metric", *names], [
        ["items run", *(len(r["items"]) for r in runs)],
        row("answerable gold n (needs ≥ 120)", lambda s: s["gate2"]["answerable_gold_n"]),
        row("faithfulness (≥ 0.90)", lambda s: s["gate2"]["faithfulness"]),
        row("citation accuracy (≥ 0.90)", lambda s: s["gate2"]["citation_accuracy"]),
        row("Gate 2", lambda s: "PASS" if s["gate2"]["pass"] else
            ("FAIL" if s["gate2"]["counts"] else "does not count yet")),
        row("eligibility accuracy (≥ 0.80)", lambda s: f"{s['eligibility']['accuracy']} (n={s['eligibility']['n']})"),
        row("abstention recall (≥ 0.90)", lambda s: s["abstention"]["recall"]),
        row("abstention precision", lambda s: s["abstention"]["precision"]),
        row("false abstain rate", lambda s: s["abstention"]["false_abstain_rate"]),
        row("₹ / query, mean", lambda s: s["cost_inr"]["mean"]),
        row("provider failures excluded", lambda s: s["excluded_provider_failures"]),
    ]), ""]  # fmt: skip

    out += ["## Truncation", "", table(["generator", "compact retries", "failed after retry", "failure rate"], [
        [r["model"], r["summary"]["truncation"]["compact_retries"],
         r["summary"]["truncation"]["failures"], r["summary"]["truncation"]["failure_rate"]]
        for r in runs]), ""]  # fmt: skip

    slice_rows = []
    for key, label in [*((f"slice:{s}", s) for s in SLICES), *((f"lang:{x}", x) for x in LANGS)]:
        cells: list[object] = [label]
        for r in runs:
            b = r["summary"]["by_slice"][key]
            cells += [b["n"], b["faithfulness"], b["citation_accuracy"]]
        slice_rows.append(cells)
    head = ["slice"] + [f"{n.split('/')[-1]} {m}" for n in names for m in ("n", "faith.", "cite")]
    out += ["## By slice and language", "", table(head, slice_rows), ""]

    lat_rows = []
    for name in names:
        live = live_run(name)
        if live is None:
            lat_rows.append([name, "pending", None, None, None])
            continue
        lat = live["summary"]["latency_ms"]
        lat_rows.append([name, lat["n_live"], f"{lat['ttfc_p50']} / {lat['ttfc_p95']}",
                         f"{lat['ttft_p50']} / {lat['ttft_p95']}",
                         f"{lat['full_p50']} / {lat['full_p95']}"])  # fmt: skip
    out += ["## Live latency (ms, p50 / p95)", "",
            table(["generator", "n", "first source (TTFC)", "first answer token (TTFT)", "full answer"], lat_rows),
            "", "Free-tier throttling is included.", ""]  # fmt: skip

    agree_rows = []
    for path, name in zip(paths, names, strict=True):
        a = agreement_of(path)
        if a is None:
            agree_rows.append([name, "pending", None, None, None, None])
            continue
        s, c = a["supported"], a["cited_ok"]
        agree_rows.append([name, f"{a['first_judge']} vs {a['second_judge']}", a["answers"],
                           s["statements"], f"{s['agreement']} (κ {s['kappa']})",
                           f"{c['agreement']} (κ {c['kappa']})"])  # fmt: skip
    out += ["## Judge agreement", "",
            table(["generator", "judges", "answers", "statements", "supported", "cited ok"], agree_rows),
            "", "The main judge, gpt-oss-120b, comes from the same model family as gpt-oss-20b, so its "
            "scores for that generator may be less independent. The Gemini check is the guard, on "
            "a small sample.", ""]  # fmt: skip
    return "\n".join(out)


def update_readme(text: str, readme: Path = ROOT / "README.md") -> None:
    """Replace the block between the gate2 markers, dropping the report's own title."""
    body = text.removeprefix("# Gate 2 report\n\n").replace("\n## ", "\n#### ")
    page = readme.read_text(encoding="utf-8")
    head, rest = page.split("<!-- gate2:start -->")
    _, tail = rest.split("<!-- gate2:end -->")
    new = f"{head}<!-- gate2:start -->\n{body}\n<!-- gate2:end -->{tail}"
    readme.write_text(new, encoding="utf-8", newline="\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    text = report([Path(p).resolve() for p in sys.argv[1:]])
    path = RESULTS / "gate2_final.md"
    path.write_text(text, encoding="utf-8")
    update_readme(text)
    print(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
