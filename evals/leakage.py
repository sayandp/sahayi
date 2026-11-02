"""Gold-set leakage report: lexical overlap between each question and its gold text.

    python evals/leakage.py [--threshold 0.4]

overlap = share of the question's content words (BM25 tokens without stemming, minus
stopwords and 1-2 character tokens) that occur in the best-matching gold chunk.
Questions that BM25 alone ranks at 1 with overlap > threshold were probably written
by copying the document's wording; they flatter lexical retrieval and should be
rephrased the way a citizen would ask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.bm25 import BM25Index, TokenizerConfig, tokenize  # noqa: E402
from backend.app.retrieval.search import RetrievalConfig, load_chunks  # noqa: E402
from evals.metrics import first_relevant_rank  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"
OUT = ROOT / "evals" / "results" / "gold_leakage.md"
STOP = set(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "is",
        "are",
        "be",
        "can",
        "i",
        "my",
        "me",
        "what",
        "which",
        "who",
        "how",
        "much",
        "many",
        "does",
        "do",
        "under",
        "and",
        "or",
        "with",
        "get",
        "any",
        "there",
        "this",
        "that",
        "it",
        "as",
        "at",
        "by",
        "from",
        "will",
        "would",
        "should",
        "per",
        "year",
        "scheme",
        "yojana",
    ]
)
NO_STEM = TokenizerConfig(stem=False)


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text, NO_STEM) if len(t) > 2 and t not in STOP}


def overlap(question: str, chunk_text: str) -> float:
    q = content_tokens(question)
    return len(q & content_tokens(chunk_text)) / len(q) if q else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()
    cfg = RetrievalConfig(mode="bm25", rerank=False)
    chunks = load_chunks(cfg.strategy)
    by_page: dict[tuple[str, int], list[str]] = {}
    for c in chunks.values():
        by_page.setdefault((c["doc_id"], c["page"]), []).append(c["text"])
    bm25 = BM25Index.load(cfg.bm25_path)
    rows = []
    with GOLD.open(encoding="utf-8") as f:
        for q in map(json.loads, f):
            if q["type"] != "answerable":
                continue
            gold = [(g["doc_id"], g["page"]) for g in q["gold"]]
            texts = [t for p in gold for t in by_page.get(p, [])]
            ov = max((overlap(q["question"], t) for t in texts), default=0.0)
            hits = bm25.search(q["question"], 10)
            pages = [(chunks[c]["doc_id"], chunks[c]["page"]) for c, _ in hits]
            rows.append({"id": q["id"], "lang": q["lang"], "overlap": round(ov, 2),
                         "bm25_rank": first_relevant_rank(pages, gold),
                         "question": q["question"]})  # fmt: skip
    leaky = [r for r in rows if r["bm25_rank"] == 1 and r["overlap"] > args.threshold]
    lines = [
        "# Gold leakage report",
        "",
        f"{len(rows)} answerable questions; mean overlap "
        f"{sum(r['overlap'] for r in rows) / len(rows):.2f}; "
        f"BM25 rank-1 with overlap > {args.threshold}: **{len(leaky)}**.",
        "",
        "| id | lang | overlap | BM25 rank | question |",
        "|---|---|---|---|---|",
        *[f"| {r['id']} | {r['lang']} | {r['overlap']} | {r['bm25_rank']} | {r['question']} |"
          for r in sorted(rows, key=lambda r: -r["overlap"])],  # fmt: skip
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"mean overlap {sum(r['overlap'] for r in rows) / len(rows):.2f}; leaky: {len(leaky)}")
    for r in leaky:
        print(f"  {r['id']} {r['lang']} ov={r['overlap']} | {r['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
