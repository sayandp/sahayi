"""Replay recorded generations through the card filter (no LLM): what would it drop?

python evals/card_filter_impact.py RUN.json [RUN2.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.answer import parse_json  # noqa: E402
from backend.app.retrieval.search import load_chunks  # noqa: E402
from backend.app.validate import Validator, filter_cards  # noqa: E402
from evals.answer_eval import load_eligibility  # noqa: E402


def main() -> int:
    chunks = load_chunks("section_aware")
    expected = {c["id"]: c["expected"] for c in load_eligibility()}
    for path in sys.argv[1:]:
        cards = dropped = lost_expected = answers = 0
        for r in json.loads(Path(path).read_text(encoding="utf-8"))["items"]:
            data = parse_json(r.get("raw_generation") or "")
            if not data or r["abstained"]:
                continue
            answers += 1
            aliases = {f"C{i}": c for i, c in enumerate(r["context"], start=1)}
            ids = {c: c.split("--")[0] for c in r["context"]}
            ctx = {c: chunks[c] for c in r["context"]}
            v = Validator(aliases, ctx, r["question"]).validate(data, ids)
            cited = list(dict.fromkeys(c for s in v.sentences for c in s["citations"]))
            kept, gone = filter_cards(v.schemes, cited, r["context"], ctx, ids)
            cards += len(v.schemes)
            dropped += len(gone)
            need = {e["scheme_id"] for e in expected.get(r["id"], [])}
            if need & set(gone):
                lost_expected += 1
                print(f"  {r['id']}: dropped the expected card {sorted(need & set(gone))}")
        print(f"{Path(path).name}: {answers} answers, {cards} cards, {dropped} dropped, "
              f"{lost_expected} expected eligibility cards lost")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
