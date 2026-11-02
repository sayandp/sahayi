"""Generate held-out gold questions with a different model family (Phase 2b).

    python evals/gold/make_heldout.py [--n 20] [--model openai/gpt-oss-120b] [--seed 7]

The original gold set was drafted by Claude and the query rewriter is qwen, so these
are written by gpt-oss (or Gemini if GEMINI_API_KEY is set and --gemini is passed).
For each sampled chunk the model writes one citizen-style question in a requested
language and an exact quote from the chunk that answers it. The quote must occur
verbatim (after normalisation) or the item is retried / dropped. Appends ids h001.. to
questions_src.yaml with split: heldout and author: <model>.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.llm.client import LLMError, chat_json, gemini_chat  # noqa: E402
from backend.app.retrieval.search import load_chunks  # noqa: E402
from evals.build_gold import norm  # noqa: E402

SRC = ROOT / "evals" / "gold" / "questions_src.yaml"
SECTIONS = {"eligibility", "benefits", "documents", "how_to_apply", "exclusions", None}
# target mix: mostly Manglish, plus cross-lingual both ways
PLAN = ["manglish"] * 12 + ["ml_about_en"] * 4 + ["en_about_ml"] * 4

PROMPT = """You help build a test set for a Kerala government-scheme search assistant.
Below is a passage from an official document (scheme: {scheme}).

Write ONE question a real Kerala resident might ask, which this passage answers.
Language: {lang_rule}
Ask about their own situation in everyday words; do not copy the passage's phrasing
or long names from it.

Also give "evidence": an EXACT quote (6-25 words) copied character-for-character from
the passage that answers the question.

Return JSON: {{"question": "...", "evidence": "..."}}

PASSAGE:
{text}
"""
LANG_RULES = {
    "manglish": "Manglish - Malayalam written in Latin letters, as typed on a phone "
                "(e.g. 'ente amma kku pension kittumo?').",
    "ml_about_en": "Malayalam script.",
    "en_about_ml": "Simple English.",
}  # fmt: skip


def pick_chunks(chunks: dict, n_plan: list[str], seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    pool = [c for c in chunks.values() if c["n_tokens"] >= 40 and c.get("section") in SECTIONS]
    en_pool = [c for c in pool if c["lang"] == "en"]
    ml_pool = [c for c in pool if c["lang"] == "ml"]
    rng.shuffle(en_pool)
    rng.shuffle(ml_pool)
    picked, used_schemes = [], set()

    def take(src: list[dict]) -> dict:
        for c in src:
            if c["scheme_id"] not in used_schemes:
                used_schemes.add(c["scheme_id"])
                src.remove(c)
                return c
        return src.pop()

    for kind in n_plan:
        if kind == "ml_about_en":
            picked.append((kind, take(en_pool)))
        elif kind == "en_about_ml":
            picked.append((kind, take(ml_pool)))
        else:  # Manglish: alternate Malayalam and English source documents
            picked.append((kind, take(ml_pool if len(picked) % 2 else en_pool)))
    return picked


def ask(model: str, prompt: str, use_gemini: bool) -> dict:
    messages = [{"role": "user", "content": prompt}]
    if use_gemini:
        res = gemini_chat(messages, "gemini-2.0-flash")
        return json.loads(res.text)
    return chat_json(messages, groq_model=model).data or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--append", action="store_true", help="keep existing held-out items")
    ap.add_argument("--plan", nargs="*", help="override the language plan, e.g. en_about_ml")
    args = ap.parse_args()
    author = "gemini-2.0-flash" if args.gemini else args.model
    chunks = load_chunks("section_aware")
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    existing = [q for q in data["questions"] if q.get("split") == "heldout"]
    if not args.append:
        data["questions"] = [q for q in data["questions"] if q.get("split") != "heldout"]
        existing = []
    plan = args.plan or (
        PLAN[: args.n] if args.n <= len(PLAN) else PLAN + ["manglish"] * (args.n - len(PLAN))
    )
    taken = {s for q in existing for s in q["schemes"]}
    start = max((int(q["id"][1:]) for q in existing), default=0) + 1
    out = []
    pool = {k: c for k, c in chunks.items() if c["scheme_id"] not in taken}
    for i, (kind, chunk) in enumerate(pick_chunks(pool, plan, args.seed), start=start):
        prompt = PROMPT.format(scheme=chunk["scheme"], lang_rule=LANG_RULES[kind],
                               text=chunk["text"][:2500])  # fmt: skip
        for attempt in range(3):
            try:
                item = ask(args.model, prompt, args.gemini)
            except (LLMError, json.JSONDecodeError) as exc:
                print(f"  h{i:03d} attempt {attempt + 1}: {exc}")
                continue
            ev, qtext = item.get("evidence", ""), item.get("question", "")
            if qtext and ev and norm(ev) in norm(chunk["text"]):
                lang = {"manglish": "manglish", "ml_about_en": "ml", "en_about_ml": "en"}[kind]
                out.append({"id": f"h{i:03d}", "lang": lang, "type": "answerable",
                            "split": "heldout", "author": author, "kind": kind,
                            "schemes": [chunk["scheme_id"]], "question": qtext,
                            "evidence": [ev]})  # fmt: skip
                print(f"  h{i:03d} {kind:12} {chunk['scheme_id']:28} | {qtext}")
                break
            print(f"  h{i:03d} attempt {attempt + 1}: evidence not verbatim, retrying")
    data["questions"].extend(out)
    header = SRC.read_text(encoding="utf-8").split("questions:")[0]
    SRC.write_text(header + yaml.safe_dump({"questions": data["questions"]}, allow_unicode=True,
                                           sort_keys=False, width=4096), encoding="utf-8")  # fmt: skip
    print(f"added {len(out)}/{len(plan)} held-out questions by {author}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
