"""Retrieval evaluation on evals/gold/questions.jsonl (Gate 1: hit@5 >= 0.85).

    python evals/run_eval.py --config default
    python evals/run_eval.py --config bm25 dense hybrid default   # comparison table
    python evals/run_eval.py --config default --gate              # exit 1 if below target

Writes evals/results/{date}_{config}.json (per-question detail) and
evals/results/{date}_retrieval.md (one row per config run in this invocation).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.search import RetrievalConfig, Retriever  # noqa: E402
from evals.metrics import first_relevant_rank, hit_at_k, mrr_at_k, percentile  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"
CONFIGS = ROOT / "evals" / "configs.yaml"
RESULTS = ROOT / "evals" / "results"
GATE_HIT5 = 0.85
TARGET_MRR10 = 0.70


def load_config(name: str) -> RetrievalConfig:
    raw = yaml.safe_load(CONFIGS.read_text(encoding="utf-8"))
    if name not in raw:
        raise SystemExit(f"unknown config {name!r}; choose from {sorted(raw)}")
    return RetrievalConfig(name=name, **raw[name])


def load_gold(limit: int | None = None) -> list[dict]:
    with GOLD.open(encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    return gold[:limit] if limit else gold


def evaluate(
    config: RetrievalConfig, questions: list[dict], qdrant_path: Path | None = None
) -> dict:
    import psutil

    retriever = Retriever(config, qdrant_path)
    proc = psutil.Process()
    rows = []
    by_page: dict[tuple[str, int], list[dict]] = {}
    for c in retriever.chunks.values():
        by_page.setdefault((c["doc_id"], c["page"]), []).append(c)
    try:
        retriever.search("warm-up: load models before timing")  # loads embedder/reranker
        for q in questions:
            hits = retriever.search(q["question"])
            tr = retriever.last_trace
            if len(rows) % 10 == 9:
                print(f"  {len(rows) + 1}/{len(questions)} ({tr.total_ms:.0f} ms)", flush=True)
            pages = [(h.chunk["doc_id"], h.chunk["page"]) for h in hits]
            gold = [(g["doc_id"], g["page"]) for g in q["gold"]]
            top5 = [h.chunk for h in hits[:5]]
            rows.append(
                {
                    "id": q["id"],
                    "lang": q["lang"],
                    "type": q["type"],
                    "split": q.get("split", "main"),
                    "rank": first_relevant_rank(pages, gold) if gold else None,
                    "top_score": hits[0].score if hits else None,
                    "top_rerank_score": tr.top_rerank_score,
                    "top": [f"{d}:{p}" for d, p in pages[:5]],
                    "latency_ms": round(tr.total_ms, 1),
                    "timings": {k: round(getattr(tr, k), 1) for k in TIMING_KEYS},
                    "rewrite_cached": tr.rewrite_cached,
                    "rewrite_lang": tr.rewrite_lang,
                    "reranked": tr.reranked,
                    "rerank_skipped": tr.rerank_skipped,
                    "rerank_backend": tr.rerank_backend_used,
                    "rerank_fallback": tr.rerank_fallback,
                    "scheme_hit5": scheme_hit(top5, q) if gold else None,
                    "section_hit5": section_hit(top5, gold, by_page) if gold else None,
                }
            )
    finally:
        retriever.close()
    mem = proc.memory_info()
    peak_gb = getattr(mem, "peak_wset", mem.rss) / 1e9
    return {"config": config.to_dict(), "summary": summarize(rows, peak_gb), "questions": rows}


def _sid(chunk: dict) -> str:
    return chunk["doc_id"].split("--")[0]


def scheme_hit(top: list[dict], q: dict) -> bool:
    """Some top chunk belongs to one of the question's gold schemes."""
    return any(_sid(c) in set(q.get("schemes") or []) for c in top)


def section_hit(top: list[dict], gold: list[tuple[str, int]], by_page: dict) -> bool:
    """Same scheme and same section as a gold-page chunk (same page if it has no section)."""
    keys = set()
    for page in gold:
        for c in by_page.get(tuple(page), []):
            keys.add((_sid(c), c["section"]) if c.get("section") else (c["doc_id"], c["page"]))
    return any(((_sid(c), c.get("section")) in keys) or ((c["doc_id"], c["page"]) in keys)
               for c in top)  # fmt: skip


TIMING_KEYS = (
    "rewrite_ms",
    "rewrite_wait_ms",
    "bm25_ms",
    "dense_ms",
    "spec_rerank_ms",
    "rerank_ms",
    "total_ms",
)


def block(subset: list[dict]) -> dict:
    ranks = [r["rank"] for r in subset]
    out = {
        "n": len(subset),
        "hit@1": round(hit_at_k(ranks, 1), 3),
        "hit@5": round(hit_at_k(ranks, 5), 3),
        "hit@10": round(hit_at_k(ranks, 10), 3),
        "mrr@10": round(mrr_at_k(ranks, 10), 3),
    }
    for key in ("scheme_hit5", "section_hit5"):
        vals = [r[key] for r in subset if r.get(key) is not None]
        if vals:
            out[key.replace("_hit5", "_hit@5")] = round(sum(vals) / len(vals), 3)
    return out


def summarize(rows: list[dict], peak_gb: float | None = None) -> dict:
    answerable = [r for r in rows if r["type"] == "answerable" and r.get("split", "main") == "main"]
    heldout = [r for r in rows if r["type"] == "answerable" and r.get("split") == "heldout"]
    # latency: exclude cached-rewrite effects by reporting the live rewrite time separately
    timings = {
        k: {
            "p50": round(percentile([r["timings"][k] for r in rows if "timings" in r], 0.5), 1),
            "p95": round(percentile([r["timings"][k] for r in rows if "timings" in r], 0.95), 1),
        }
        for k in TIMING_KEYS
    }
    live_rw = [
        r["timings"]["rewrite_ms"] for r in rows if "timings" in r and not r["rewrite_cached"]
    ]
    if live_rw:
        timings["rewrite_ms_live"] = {"p50": round(percentile(live_rw, 0.5), 1),
                                      "p95": round(percentile(live_rw, 0.95), 1),
                                      "n": len(live_rw)}  # fmt: skip
    reranked = [r for r in rows if r.get("reranked") or r.get("rerank_skipped")]
    latencies = [r["latency_ms"] for r in rows]
    score = (
        "top_rerank_score"
        if any(r.get("top_rerank_score") is not None for r in rows)
        else "top_score"
    )
    no_ans = [r[score] for r in rows if r["type"] == "no_answer" and r[score] is not None]
    ans = [r[score] for r in answerable if r[score] is not None]
    return {
        "all": block(answerable),
        **{lang: block([r for r in answerable if r["lang"] == lang]) for lang in ("en", "ml", "manglish")},
        "heldout": block(heldout) if heldout else None,
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "timings_ms": timings,
        "rerank_skip_rate": round(sum(r.get("rerank_skipped", False) for r in reranked) / len(reranked), 3)
        if reranked else None,
        "peak_ram_gb": round(peak_gb, 2) if peak_gb else None,
        "rerank_fallback_rate": round(sum(bool(r.get("rerank_fallback")) for r in rows) / len(rows), 3)
        if rows else None,
        "top_score_field": score,
        "top_score_median": {
            "answerable": round(statistics.median(ans), 4) if ans else None,
            "no_answer": round(statistics.median(no_ans), 4) if no_ans else None,
        },
    }  # fmt: skip


def markdown(results: list[dict], n_questions: int) -> str:
    head = (
        "| config | mode | rewrite | rerank | hit@1 | hit@5 | hit@10 | MRR@10 | hit@5 en "
        "| hit@5 ml | hit@5 manglish | hit@5 held-out | p50 ms | p95 ms | skip rate | peak RAM GB |"
    )
    lines = [
        f"# Retrieval eval — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Gold: {n_questions} answerable questions (evals/gold/questions.jsonl); "
        f"Gate 1 = hit@5 ≥ {GATE_HIT5}, MRR@10 target ≥ {TARGET_MRR10}.",
        "",
        head,
        "|" + "---|" * (head.count("|") - 1),
    ]
    for r in results:
        c, s = r["config"], r["summary"]
        rr = (f"{c['rerank_model'].split('/')[-1]} {c['rerank_backend']} top-{c['rerank_pool']}"
              if c["rerank"] else "no")  # fmt: skip
        held = s["heldout"]["hit@5"] if s.get("heldout") else "-"
        lines.append(
            f"| {c['name']} | {c['mode']} | {'yes' if c.get('rewrite') else 'no'} | {rr} "
            f"| {s['all']['hit@1']} | **{s['all']['hit@5']}** | {s['all']['hit@10']} "
            f"| {s['all']['mrr@10']} | {s['en']['hit@5']} | {s['ml']['hit@5']} "
            f"| {s['manglish']['hit@5']} | {held} | {s['latency_ms']['p50']:.0f} "
            f"| {s['latency_ms']['p95']:.0f} | {s.get('rerank_skip_rate') if s.get('rerank_skip_rate') is not None else '-'} "
            f"| {s.get('peak_ram_gb') or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", default=["default"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gate", action="store_true", help="exit 1 if the first config fails Gate 1")
    args = ap.parse_args()

    questions = load_gold(args.limit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    results = []
    for name in args.config:
        config = load_config(name)
        print(f"== {name}: {config}", flush=True)
        result = evaluate(config, questions)
        results.append(result)
        (RESULTS / f"{stamp}_{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(result["summary"], indent=1), flush=True)
    n_ans = sum(q["type"] == "answerable" for q in questions)
    md = markdown(results, n_ans)
    (RESULTS / f"{stamp}_retrieval.md").write_text(md, encoding="utf-8")
    print(md)
    first = results[0]["summary"]["all"]
    passed = first["hit@5"] >= GATE_HIT5
    print(f"Gate 1 ({args.config[0]}): hit@5 {first['hit@5']} -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed or not args.gate else 1


if __name__ == "__main__":
    sys.exit(main())
