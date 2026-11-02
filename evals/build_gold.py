"""Resolve evals/gold/questions_src.yaml into evals/gold/questions.jsonl.

Gold is a set of (doc_id, page) pairs, independent of chunking strategy: every
chunkable page of the listed schemes whose text contains any evidence snippet.
Fails if a snippet matches nothing (a typo, or the corpus changed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import regex as re
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.clean import normalise_text  # noqa: E402

SRC = ROOT / "evals" / "gold" / "questions_src.yaml"
OUT = ROOT / "evals" / "gold" / "questions.jsonl"
PAGES = ROOT / "data" / "processed" / "pages.clean.jsonl"
UNUSABLE = {"ocr_unavailable", "ocr_failed"}


def norm(text: str) -> str:
    return " ".join(normalise_text(text).lower().split())


def chunkable_pages() -> list[dict]:
    pages = []
    with PAGES.open(encoding="utf-8") as f:
        for p in map(json.loads, f):
            usable = p["text"] and p["method"] not in UNUSABLE
            if usable and not p.get("excluded") and not p.get("personal_data"):
                pages.append({**p, "norm": norm(p["text"])})
    return pages


def resolve(question: dict, pages: list[dict]) -> tuple[list[dict], list[str]]:
    schemes = set(question.get("schemes", []))
    pool = [p for p in pages if p["scheme_id"] in schemes]
    gold: dict[tuple[str, int], dict] = {}
    missing = []
    for snippet in question.get("evidence", []):
        needle = norm(snippet)
        hits = [p for p in pool if needle in p["norm"]]
        if not hits:
            missing.append(snippet)
        for p in hits:
            gold[(p["doc_id"], p["page"])] = {"doc_id": p["doc_id"], "page": p["page"]}
    return sorted(gold.values(), key=lambda g: (g["doc_id"], g["page"])), missing


def main() -> int:
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))["questions"]
    pages = chunkable_pages()
    corpus = " ".join(p["norm"] for p in pages)
    ids = [q["id"] for q in data]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    out, problems = [], []
    rejected = [q["id"] for q in data if q.get("status") == "rejected"]
    data = [q for q in data if q.get("status") != "rejected"]
    for q in data:
        gold, missing = ([], [])
        if q["type"] == "answerable":
            gold, missing = resolve(q, pages)
            if missing or not gold:
                problems.append(f"{q['id']}: evidence not found: {missing}")
        out.append(
            {
                "id": q["id"],
                "lang": q["lang"],
                "type": q["type"],
                "question": q["question"],
                "schemes": q.get("schemes", []),
                "gold": gold,
                "answer": q.get("answer"),
                "split": q.get("split", "main"),
                "author": q.get("author", "claude"),
            }
        )
    # Reviewed 2026-10-12: passport / PAN card (ID-proof lists), ലോട്ടറി (KASP funding
    # note) and NORKA (a GO copy list) occur only incidentally; the no-answer questions
    # about them stay unanswerable. Anything else is flagged for review.
    reviewed = {"passport", "ലോട്ടറി", "norka", "pan card"}
    for term in ("sukanya", "passport", "lottery", "ലോട്ടറി", "norka", "agniveer", "pan card"):
        if term not in reviewed and re.search(re.escape(term), corpus):
            problems.append(f"no-answer term {term!r} occurs in the corpus - review the set")
    if problems:
        print("\n".join(problems))
        return 1
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out), encoding="utf-8")
    by = {}
    for q in out:
        by.setdefault((q["type"], q["lang"]), 0)
        by[(q["type"], q["lang"])] += 1
    print(f"{len(out)} questions -> {OUT.relative_to(ROOT)}: {dict(sorted(by.items()))}")
    if rejected:
        print(f"skipped {len(rejected)} rejected: {rejected}")
    with_gold = [q for q in out if q["gold"]]
    avg = sum(len(q["gold"]) for q in with_gold) / len(with_gold)
    print(f"avg gold pages per answerable question: {avg:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
