"""Assemble evals/results/phase2b_summary.md from the latest result per config."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"
OUT = RES / "phase2b_summary.md"

ROWS = [
    ("dense", "dense, no rewrite (baseline on the rephrased gold)"),
    ("hybrid_rw", "hybrid + rewrite"),
    ("hybrid_rw_vw05", "hybrid + rewrite, variants weighted 0.5"),
    ("hybrid_rw_vw05_jina10_sel", "… + Jina int8 top-10, selective (margin 0.05)"),
    ("hybrid_ctx", "hybrid + contextual headers, no rewrite"),
    ("dense_rw_ctx", "dense + rewrite + context"),
    ("hybrid_rw_ctx", "hybrid + rewrite + context"),
    ("hybrid_rw_ctx_jina10", "… + Jina int8 top-10 @384"),
    ("hybrid_rw_ctx_jina10_sel", "… + Jina int8 top-10 @384, selective"),
    ("hybrid_rw_ctx_jina8", "… + Jina int8 top-8 @384"),
    ("ctx_jina8_l256_original", "… + Jina int8 top-8 @256, overlap, rerank original"),
    ("ctx_jina6_l256_clearest", "… + Jina int8 top-6 @256, overlap, rerank clearest"),
    ("default", "**default**: … + Jina int8 top-6 @256, overlap, rerank original"),
    ("ctx_bge10", "… + bge-reranker-v2-m3 fp32 top-10 @384 (reference)"),
    ("ctx_bgeint8_10", "… + bge-reranker-v2-m3 int8 ONNX top-10 @384 (reference)"),
]
# Live end-to-end p95 (real Groq rewrite, no caches), from evals/bench_e2e.py
E2E = ["e2e_ctx_jina8_l256_original_t8", "e2e_ctx_jina6_l256_original_t8",
       "e2e_ctx_jina6_l256_original_t2"]  # fmt: skip


def latest(name: str) -> dict | None:
    files = sorted(glob.glob(str(RES / f"2026-*_{name}.json")))
    return json.loads(Path(files[-1]).read_text(encoding="utf-8")) if files else None


def main() -> int:
    lines = [
        "| config | what | hit@1 | hit@5 | hit@10 | MRR@10 | EN | ML | Manglish | held-out | skip rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, what in ROWS:
        r = latest(name)
        if not r:
            lines.append(f"| {name} | {what} | (not run) |||||||||")
            continue
        s = r["summary"]
        held = s["heldout"]["hit@5"] if s.get("heldout") else "-"
        skip = s.get("rerank_skip_rate")
        lines.append(
            f"| {name} | {what} | {s['all']['hit@1']} | **{s['all']['hit@5']}** | "
            f"{s['all']['hit@10']} | {s['all']['mrr@10']} | {s['en']['hit@5']} | "
            f"{s['ml']['hit@5']} | {s['manglish']['hit@5']} | {held} | "
            f"{'-' if skip is None else skip} |"
        )
    e2e = ["| live run | threads | n | p50 ms | p95 ms | p95 EN / ML / Manglish | peak RAM GB |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    stages = []
    for name in E2E:
        path = RES / f"{name}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        bl = d["by_lang_p95"]
        e2e.append(f"| {d['config']} | {d['threads']} | {d['n']} | {d['total_ms']['p50']} | "
                   f"**{d['total_ms']['p95']}** | {bl.get('en')} / {bl.get('ml')} / "
                   f"{bl.get('manglish')} | {d['peak_ram_gb']} |")  # fmt: skip
        stages.append((name, d["stages_p50"], d["stages_p95"]))
    st = ["| live run | stat | rewrite (overlapped) | wait on rewrite | bm25 | dense | "
          "speculative rerank | rerank after rewrite | total |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, p50, p95 in stages:
        for label, d in (("p50", p50), ("p95", p95)):
            st.append(f"| {name.replace('e2e_', '')} | {label} | {d['rewrite_ms']} | "
                      f"{d['rewrite_wait_ms']} | {d['bm25_ms']} | {d['dense_ms']} | "
                      f"{d['spec_rerank_ms']} | {d['rerank_ms']} | {d['total_ms']} |")  # fmt: skip
    text = (ROOT / "evals" / "results" / "phase2b_summary.template.md").read_text(encoding="utf-8")
    text = text.replace("{{ABLATION}}", "\n".join(lines)).replace("{{E2E}}", "\n".join(e2e))
    text = text.replace("{{STAGES}}", "\n".join(st))
    worst = ROOT / "data" / "cache" / "tmp" / "worst.txt"
    if worst.exists():
        rows = worst.read_text(encoding="utf-8").splitlines()[1:]
        table = ["| id | lang | split | rank | top-1 rerank score | question | top hit | gold |",
                 "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        table += ["| " + r.replace("rank ", "").replace("top1 score ", "").replace("got: ", "")
                  .replace("gold: ", "") + " |" for r in rows]  # fmt: skip
        text = text.replace("{{WORST}}", "\n".join(table))
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
