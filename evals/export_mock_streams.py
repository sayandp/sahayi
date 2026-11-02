"""Export recorded answers as the SSE events /ask sends, for the UI's mock mode.

    python evals/export_mock_streams.py

Reads answer_eval results only (no LLM or Qdrant); writes frontend/mock/streams.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.answer import ANSWER_VERSION, Answerer  # noqa: E402
from backend.app.retrieval.search import load_chunks  # noqa: E402
from backend.app.validate import split_sentences  # noqa: E402

RES = ROOT / "evals" / "results"
GATE2 = "2026-10-30_2041_answers_gate2.json"  # current prompt; preferred where it has the item
QWEN = "2026-10-20_1445_answers_bench_qwen_full.json"
ELIG = "2026-10-20_1531_answers_elig_gptoss20b_v4.json"
# (id, file, what it demonstrates)
PICKS = [
    ("h013", QWEN, "Malayalam answer"),
    ("q086", QWEN, "Malayalam answer"),
    ("h020", QWEN, "English answer"),
    ("e002", ELIG, "English eligibility, not eligible"),
    ("e007", ELIG, "English eligibility with a follow-up question"),
    ("h026", QWEN, "Manglish answer"),
    ("q058", QWEN, "Manglish answer"),
    ("n014", QWEN, "No answer in the documents: abstains"),
]


def events(row: dict, chunks: dict) -> list[dict]:
    lang = "en" if row["lang"] == "en" else "ml"
    sources = [Answerer._citation(chunks[c]) | {"tag": f"C{i}"}
               for i, c in enumerate(row["context"], start=1)]  # fmt: skip
    out = [{"event": "sources", "data": {"lang": lang, "sources": sources}}]
    for sent in split_sentences(row["answer"]) if not row["abstained"] else [row["answer"]]:
        out.append({"event": "token", "data": {"text": sent + " "}})
    meta = row["latency"]
    cited = row["citations"]
    dates = sorted(chunks[c]["downloaded_at"] for c in cited or row["context"])
    out.append({"event": "result", "data": {
        "answer": row["answer"], "lang": lang, "schemes": row["schemes"],
        "citations": [Answerer._citation(chunks[c]) for c in cited],
        "abstained": row["abstained"], "abstain_reason": row["abstain_reason"],
        "degraded": False, "follow_up_question": row["follow_up_question"],
        "as_of": dates[0], "validation": {"dropped": row["dropped"], "flags": row["flags"]},
        "meta": {"ttfc_ms": round(meta["retrieval_ms"]), "ttft_ms": row["ttft_ms"],
                 "latency_ms": row["latency_ms"], "latency": meta,
                 "tokens_in": row["tokens_in"], "tokens_out": row["tokens_out"],
                 "cost_inr": row["cost_inr"], "model": row["model"], "fallbacks": [],
                 "cached": False, "prompt_version": ANSWER_VERSION},
    }})  # fmt: skip
    return out


def main() -> int:
    chunks = load_chunks("section_aware")
    streams = []
    newest = {r["id"]: r for r in json.loads((RES / GATE2).read_text(encoding="utf-8"))["items"]}
    for rid, file, note in PICKS:
        rows = json.loads((RES / file).read_text(encoding="utf-8"))["items"]
        row = newest.get(rid) or next(r for r in rows if r["id"] == rid)
        streams.append({"id": rid, "lang": row["lang"], "note": note, "question": row["question"],
                        "events": events(row, chunks)})  # fmt: skip
    path = ROOT / "frontend" / "mock" / "streams.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(streams, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in streams:
        print(s["id"], s["lang"], s["note"], "|", s["question"][:60])
    return 0


if __name__ == "__main__":
    sys.exit(main())
