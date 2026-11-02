"""Answer evaluation (Gate 2: faithfulness >= 0.90, citation accuracy >= 0.90).

    python evals/answer_eval.py --all --tag gate2            # every item, cached rewrites
    python evals/answer_eval.py --all --resume <run.json>    # continue after a quota stop
    python evals/answer_eval.py --all --live --no-judge --limit 30 --tag gate2   # live latency

Results are saved after every item. On a Groq daily-quota 429 the run stops cleanly, prints
the resume command and exits 2; a resume skips finished items and judges unjudged ones.
Tokens used are recorded per run in the report and in evals/results/token_log.jsonl.

Items: a stratified sample of the gold set (answerable main / held-out by language,
no-answer) plus the 15 eligibility cases (evals/gold/eligibility.yaml). Each item runs
the full pipeline (Answerer: retrieval -> generation -> validation).

Metrics (judge = a different model from the generator, default openai/gpt-oss-120b; it is
sent only the excerpts the answer cites, to save tokens):
  faithfulness       RAGAS-style: share of user-visible statements (answer sentences and
                     scheme-card fields) the judge finds supported by the cited excerpts
  citation accuracy  share of statements whose OWN cited excerpts support them (judge),
                     with numbers matching (judge + the programmatic number check)
  answer relevancy   judge's 1-5 rating / 5 (RAGAS uses question regeneration +
                     embeddings; this is an LLM-rated stand-in)
  eligibility acc.   expected (scheme, eligible) found in the result's scheme cards
  abstention         precision / recall of abstaining, positive class = no-answer
  language match     answer script matches the expected language (Manglish -> Malayalam)
RAGAS itself is not installed: it pulls langchain + OpenAI-style clients and its prompts
assume one judge LLM provider; the metrics are re-implemented with one judge call per
answer to fit Groq's free tier (8K tokens/min, 200K tokens/day per model).

--live: no rewrite / rerank / generation caches and a token pacer, so latency (TTFT and
full answer) is real. Otherwise generations and judgements are cached on disk
(data/cache/answers, data/cache/judge) and latency is not meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.llm.client import (  # noqa: E402
    ANSWER_MAX_TOKENS,
    GroqStream,
    LLMError,
    chat_json,
    gemini_chat,
)
from backend.app.llm.prompts import ANSWER_VERSION, JUDGE_SYSTEM, JUDGE_VERSION  # noqa: E402
from evals.metrics import percentile  # noqa: E402

GOLD = ROOT / "evals" / "gold" / "questions.jsonl"
ELIGIBILITY = ROOT / "evals" / "gold" / "eligibility.yaml"
RESULTS = ROOT / "evals" / "results"
GEN_CACHE = ROOT / "data" / "cache" / "answers"
JUDGE_CACHE = ROOT / "data" / "cache" / "judge"
ML_CHAR = re.compile("[ഀ-ൿ]")
LATIN = re.compile("[A-Za-z]")
GATE_FAITH, GATE_CITE = 0.90, 0.90
GATE_MIN_ANSWERABLE = 120  # Gate 2 counts only with at least this many answerable questions
TOKEN_LOG = RESULTS / "token_log.jsonl"
# Gemini's free tier does not publish per-model limits; these held in testing on 2026-10-30
GEMINI_GAP_S, GEMINI_BACKOFF_S, GEMINI_TRIES = 13.0, 65.0, 4
TPM_BUDGET = 7000  # stay under Groq's 8K tokens/minute per model
OTPM_BUDGET = 900  # qwen also has a limit of 1000 output tokens/minute (seen 2026-10-30)


class BudgetExhausted(RuntimeError):
    pass


# ------------------------------------------------------------------ pacing / caching


class Pacer:
    """Sliding-window token pacing per model, so free-tier TPM limits never throttle."""

    def __init__(self, budget: int = TPM_BUDGET) -> None:
        self.budget = budget
        self.calls: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self.last: dict[str, int] = {}

    def wait(self, model: str, estimate: int | None = None, budget: int | None = None) -> float:
        est = estimate or self.last.get(model, 3500)
        budget = budget or self.budget
        waited = 0.0
        while True:
            now = time.monotonic()
            window = [(t, n) for t, n in self.calls[model] if now - t < 60]
            self.calls[model] = window
            if sum(n for _, n in window) + est <= budget or not window:
                return waited
            pause = 60 - (now - window[0][0]) + 0.5
            time.sleep(pause)
            waited += pause

    def record(self, model: str, tokens: int) -> None:
        if tokens:
            self.calls[model].append((time.monotonic(), tokens))
            self.last[model] = tokens


_RETRY_IN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")


def retry_after_s(err: str) -> float | None:
    """Seconds from Groq's "try again in 1h2m3s", if the message has one."""
    m = _RETRY_IN.search(err)
    if not m or not any(m.groups()):
        return None
    h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mi * 60 + sec


class CachedStream:
    """GroqStream with an on-disk cache of the completion (evals only, not --live)."""

    def __init__(self, messages: list[dict], model: str) -> None:
        key = hashlib.sha1(json.dumps([ANSWER_VERSION, ANSWER_MAX_TOKENS, model, messages],
                                      ensure_ascii=False)
                           .encode()).hexdigest()  # fmt: skip
        self.path = GEN_CACHE / f"{key}.json"
        self.inner = GroqStream(messages, model)
        self.text, self.tokens_in, self.tokens_out = "", 0, 0
        self.ttft_ms = None
        self.throttled_ms = 0.0
        self.cached = False
        self.finish_reason: str | None = None

    def __iter__(self):
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.text, self.tokens_in, self.tokens_out = d["text"], d["tokens_in"], d["tokens_out"]
            self.finish_reason = d.get("finish_reason")
            self.cached = True
            yield self.text
            return
        yield from self.inner
        self.text, self.tokens_in, self.tokens_out = (
            self.inner.text, self.inner.tokens_in, self.inner.tokens_out)  # fmt: skip
        self.ttft_ms, self.throttled_ms = self.inner.ttft_ms, self.inner.throttled_ms
        self.finish_reason = self.inner.finish_reason
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"text": self.text, "tokens_in": self.tokens_in,
                                         "tokens_out": self.tokens_out, "finish_reason": self.finish_reason}, ensure_ascii=False),
                             encoding="utf-8")  # fmt: skip


# ------------------------------------------------------------------ items


def load_items(n: int, seed: int, no_answer: int, eligibility: bool) -> list[dict]:
    """Stratified by (split, lang): deterministic order, so a larger n extends a smaller one."""
    with GOLD.open(encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for q in gold:
        if q["type"] == "answerable":
            strata[(q.get("split", "main"), q["lang"])].append(q)
    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    picked: list[dict] = []
    keys = sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop(0))
    nos = [q for q in gold if q["type"] == "no_answer"]
    rng.shuffle(nos)
    items = [dict(q, slice=q.get("split", "main")) for q in picked]
    items += [dict(q, slice="no_answer") for q in nos[:no_answer]]
    if eligibility:
        items += load_eligibility()
    return items


def load_eligibility() -> list[dict]:
    from backend.app.retrieval.search import load_chunks

    chunks = load_chunks("section_aware")
    cases = yaml.safe_load(ELIGIBILITY.read_text(encoding="utf-8"))["cases"]
    out = []
    for c in cases:
        for ev in c["evidence"]:
            text = " ".join(chunks[ev["chunk_id"]]["text"].split())
            if ev["quote"] not in text:
                raise SystemExit(f"{c['id']}: evidence quote not in {ev['chunk_id']}")
        gold = list({(chunks[e["chunk_id"]]["doc_id"], chunks[e["chunk_id"]]["page"])
                     for e in c["evidence"]})  # fmt: skip
        out.append({"id": c["id"], "lang": c["lang"], "type": "answerable", "split": "eligibility",
                    "slice": "eligibility", "question": c["question"],
                    "schemes": [x["scheme_id"] for x in c["expected"]],
                    "gold": [{"doc_id": d, "page": p} for d, p in gold],
                    "expected": c["expected"]})  # fmt: skip
    return out


# ------------------------------------------------------------------ judging


def statements(result: dict) -> list[dict]:
    """User-visible claims with their citations: answer sentences + scheme-card fields."""
    from backend.app.validate import split_sentences, strip_tags, tags_in

    out = []
    for sent in split_sentences(result["answer"]):
        cites = [t for t in tags_in(sent) if ":" in t]
        if cites:
            out.append({"text": strip_tags(sent), "citations": cites})
    for sc in result["schemes"]:
        fields = [*sc["reasons"], sc["benefit"], sc["how_to_apply"], *sc["documents"]]
        for f in fields:
            if f:
                cites = [t for t in tags_in(f) if ":" in t] or sc["citations"]
                out.append({"text": f"{sc['name']}: {strip_tags(f)}", "citations": cites})
    return out


def add_tokens(tokens: dict | None, model: str, tokens_in: int, tokens_out: int) -> None:
    """Accumulate tokens actually spent this run, by model."""
    if tokens is None or not (tokens_in or tokens_out):
        return
    t = tokens.setdefault(model, {"in": 0, "out": 0, "calls": 0})
    t["in"] += tokens_in
    t["out"] += tokens_out
    t["calls"] += 1


def judge_call(messages: list[dict], model: str):  # noqa: ANN201 - LLMResult
    """One judge request, Gemini or Groq; never falls back to another provider."""
    if not model.startswith("gemini"):
        return chat_json(messages, groq_model=model, timeout=60)
    for attempt in range(GEMINI_TRIES):
        time.sleep(GEMINI_GAP_S)
        try:
            res = gemini_chat(messages, model, json_mode=True, timeout=60, retries=0)
            break
        except LLMError as exc:  # free tier: per-minute limits, "high demand" 503s, stalls
            transient = any(x in str(exc) for x in ("429", "503", "Timeout"))
            if not transient or attempt == GEMINI_TRIES - 1:
                raise
            time.sleep(GEMINI_BACKOFF_S)
    try:
        res.data = json.loads(res.text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"judge returned invalid JSON: {exc}") from exc
    return res


def judge(item: dict, result: dict, context: list[dict], model: str, pacer: Pacer | None,
          use_cache: bool, tokens: dict | None = None) -> dict | None:  # fmt: skip
    stmts = statements(result)
    if not stmts:
        return None
    cited = {c for s in stmts for c in s["citations"]}
    shown = [c for c in context if c["chunk_id"] in cited]
    alias = {c["chunk_id"]: f"C{i}" for i, c in enumerate(shown, start=1)}
    excerpts = "\n\n".join(
        f"[{alias[c['chunk_id']]}] {c['scheme']} | page {c['page']}\n{c['text'][:2000]}"
        for c in shown
    )
    lines = "\n".join(
        f"{i}. {s['text']} " + "".join(f"[{alias.get(c, 'C?')}]" for c in s["citations"])
        for i, s in enumerate(stmts, start=1)
    )
    user = f"Question: {item['question']}\n\nExcerpts:\n{excerpts}\n\nAnswer statements:\n{lines}"
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    key = hashlib.sha1(json.dumps([JUDGE_VERSION, model, messages], ensure_ascii=False)
                       .encode()).hexdigest()  # fmt: skip
    path = JUDGE_CACHE / f"{key}.json"
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        if pacer:
            pacer.wait(model, len(user) // 2 + 600)
        try:
            res = judge_call(messages, model)
        except LLMError as exc:
            return {"error": str(exc)[:500]}
        if pacer:
            pacer.record(model, res.tokens_in + res.tokens_out)
        data = {"verdict": res.data, "tokens": res.tokens_in + res.tokens_out}
        add_tokens(tokens, model, res.tokens_in, res.tokens_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    verdict = data.get("verdict") or {}
    by_i = {int(v["i"]): v for v in verdict.get("statements", [])
            if isinstance(v, dict) and str(v.get("i", "")).isdigit()}  # fmt: skip
    rows = []
    for i, s in enumerate(stmts, start=1):
        v = by_i.get(i, {})
        rows.append({"text": s["text"], "supported": v.get("supported") is True,
                     "cited_ok": v.get("cited_ok") is True,
                     "numbers_ok": v.get("numbers_ok") is not False,
                     "judged": i in by_i})  # fmt: skip
    try:
        relevancy = float(verdict.get("relevancy"))
    except (TypeError, ValueError):
        relevancy = None
    return {"statements": rows, "relevancy": relevancy, "tokens": data.get("tokens")}


# ------------------------------------------------------------------ metrics


def answer_script(text: str) -> str:
    ml, latin = len(ML_CHAR.findall(text)), len(LATIN.findall(text))
    return "ml" if ml >= 0.3 * max(1, ml + latin) else "en"


def eligibility_correct(item: dict, result: dict) -> list[dict]:
    cards = {s["scheme_id"]: s for s in result["schemes"]}
    out = []
    for exp in item.get("expected", []):
        card = cards.get(exp["scheme_id"])
        out.append({"scheme_id": exp["scheme_id"], "expected": exp["eligible"],
                    "got": card["eligible"] if card else "missing",
                    "correct": bool(card) and card["eligible"] == exp["eligible"]})  # fmt: skip
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def failed(row: dict) -> bool:
    return any(f.startswith("generation_failed") for f in row["flags"])


def summarize(rows: list[dict]) -> dict:
    n_failed = sum(failed(r) for r in rows)
    rows = [r for r in rows if not failed(r)]  # provider failures are not results

    def block(rs: list[dict]) -> dict:
        answered = [r for r in rs if not r["abstained"] and r.get("judge")
                    and "statements" in r["judge"]]  # fmt: skip
        faith = [sum(s["supported"] for s in r["judge"]["statements"]) / len(r["judge"]["statements"])
                 for r in answered]  # fmt: skip
        stm = [s for r in answered for s in r["judge"]["statements"]]
        cite = [s["cited_ok"] and s["numbers_ok"] for s in stm]
        rel = [r["judge"]["relevancy"] / 5 for r in answered if r["judge"].get("relevancy")]
        ans = [r for r in rs if r["type"] == "answerable"]
        lang_ok = [r["lang_match"] for r in rs if not r["abstained"]]
        return {
            "n": len(rs),
            "answered": len([r for r in rs if not r["abstained"]]),
            "faithfulness": _mean(faith),
            "faithfulness_micro": _mean([s["supported"] for s in stm]),
            "citation_accuracy": _mean(cite),
            "answer_relevancy": _mean(rel),
            "statements": len(stm),
            "answerable_answered": _mean([not r["abstained"] for r in ans]),
            "context_hit@5": _mean([r["context_hit"] for r in ans if r["context_hit"] is not None]),
            "language_match": _mean(lang_ok),
        }

    gate_rows = [r for r in rows if r["slice"] in ("main", "heldout", "eligibility", "no_answer")]
    gate_block = block(gate_rows)
    n_gold = sum(r["slice"] in ("main", "heldout") for r in rows)
    no_ans = [r for r in rows if r["type"] == "no_answer"]
    abst = [r for r in rows if r["abstained"]]
    elig = [e for r in rows for e in r.get("eligibility", [])]
    by = {
        "all": block(rows),
        **{f"lang:{lang}": block([r for r in rows if r["lang"] == lang])
           for lang in ("en", "ml", "manglish")},
        **{f"slice:{s}": block([r for r in rows if r["slice"] == s])
           for s in ("main", "heldout", "eligibility", "no_answer")},
    }  # fmt: skip
    lat = [r for r in rows if r.get("live")]
    return {
        "excluded_provider_failures": n_failed,
        "truncation": {
            "compact_retries": sum("compact_retry" in r["flags"] for r in rows),
            "failures": sum("truncated_failed" in r["flags"] for r in rows),
            "failure_rate": _mean(["truncated_failed" in r["flags"] for r in rows]),
        },
        "gate2": {
            "answerable_gold_n": n_gold,  # main + held-out; eligibility cases are extra
            "counts": n_gold >= GATE_MIN_ANSWERABLE,
            "faithfulness": gate_block["faithfulness"],
            "citation_accuracy": gate_block["citation_accuracy"],
            "pass": n_gold >= GATE_MIN_ANSWERABLE
            and (gate_block["faithfulness"] or 0) >= GATE_FAITH
            and (gate_block["citation_accuracy"] or 0) >= GATE_CITE,
        },
        "by_slice": by,
        "abstention": {
            "no_answer_n": len(no_ans),
            "recall": _mean([r["abstained"] for r in no_ans]),
            "precision": _mean([r["type"] == "no_answer" for r in abst]),
            "false_abstain_rate": _mean([r["abstained"] for r in rows if r["type"] == "answerable"]),
        },
        "eligibility": {"n": len(elig), "accuracy": _mean([e["correct"] for e in elig]),
                        "card_found": _mean([e["got"] != "missing" for e in elig])},
        "validation": {
            "items_with_drops": sum(bool(r["dropped"]) for r in rows),
            "dropped_by_reason": dict(sorted(
                {k: sum(d["reason"] == k for r in rows for d in r["dropped"])
                 for k in {d["reason"] for r in rows for d in r["dropped"]}}.items())),
            "abstain_reason": {k: sum(r.get("abstain_reason") == k for r in rows)
                               for k in ("model", "validation", "error")},
            "json_retry": sum(any("json_retry" in f for f in r["flags"]) for r in rows),
        },
        "latency_ms": {
            "n_live": len(lat),
            "ttfc_p50": percentile([r["ttfc_ms"] for r in lat], 0.5) if lat else None,
            "ttfc_p95": percentile([r["ttfc_ms"] for r in lat], 0.95) if lat else None,
            "ttft_p50": percentile([r["ttft_ms"] for r in lat], 0.5) if lat else None,
            "ttft_p95": percentile([r["ttft_ms"] for r in lat], 0.95) if lat else None,
            "full_p50": percentile([r["latency_ms"] for r in lat], 0.5) if lat else None,
            "full_p95": percentile([r["latency_ms"] for r in lat], 0.95) if lat else None,
            # free-tier 429 waits (generation retries) are an artefact of the eval's quota
            "throttled_items": sum(bool(r["latency"].get("throttled_ms")) for r in lat),
            "ttft_p95_unthrottled": percentile(
                [r["ttft_ms"] - (r["latency"].get("throttled_ms") or 0) for r in lat], 0.95)
            if lat else None,
        },
        "cost_inr": {"mean": _mean([r["cost_inr"] for r in rows]),
                     "p95": percentile([r["cost_inr"] for r in rows], 0.95) if rows else None},
        "tokens": {"in_mean": _mean([r["tokens_in"] for r in rows]),
                   "out_mean": _mean([r["tokens_out"] for r in rows])},
    }  # fmt: skip


# ------------------------------------------------------------------ run


def _save(path: Path, args: argparse.Namespace, rows: list[dict], tokens: dict, stopped: str | None,
          retry_after: float | None) -> dict:  # fmt: skip
    report = {"model": args.model, "judge": args.judge, "config": args.config,
              "answer_prompt": ANSWER_VERSION, "judge_prompt": JUDGE_VERSION,
              "stopped": stopped, "retry_after_s": retry_after, "tokens_used": tokens,
              "summary": summarize(rows), "items": rows}  # fmt: skip
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return report


def _resume_cmd(args: argparse.Namespace, path: Path) -> str:
    flags = " --all" if args.all else f" --n {args.n} --no-answer {args.no_answer}"
    return (f"python evals/answer_eval.py --model {args.model} --judge {args.judge} "
            f"--config {args.config}{flags} --resume {path.resolve().relative_to(ROOT).as_posix()}")  # fmt: skip


def _needs_judge(row: dict, model: str) -> bool:
    j = row.get("judge")
    return not row["abstained"] and (row.get("judge_model") != model or bool(j and "error" in j))


def run(
    args: argparse.Namespace, shared_retriever=None, shared_pacer: Pacer | None = None
) -> int:  # noqa: ANN001
    """One run or turn. A shared retriever stays open for the caller's next turn."""
    from backend.app.answer import Answerer, load_retriever

    if args.live:
        os.environ["SAHAYI_REWRITE_CACHE"] = "0"
        os.environ["SAHAYI_RERANK_CACHE"] = "0"
    items = load_items(args.n, args.seed, args.no_answer, not args.no_eligibility)
    if args.only:
        items = [i for i in items if i["id"] in set(args.only.split(","))]
    by_id = {i["id"]: i for i in items}
    done: dict[str, dict] = {}
    run_tokens: dict = {}  # spent in this invocation
    all_tokens: dict = {}  # spent over all invocations of this results file
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        out_path = Path(args.resume)
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        all_tokens = prev.get("tokens_used") or {}
        done = {r["id"]: r for r in prev["items"]
                if r["model"] == args.model and r.get("prompt") == ANSWER_VERSION
                and not failed(r)}  # fmt: skip
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_path = RESULTS / f"{stamp}_answers_{args.tag or args.model.split('/')[-1]}.json"
    rows = [done[i["id"]] for i in items if i["id"] in done]
    todo = [i for i in items if i["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    pending = [] if args.no_judge else [r for r in rows if _needs_judge(r, args.judge)]
    print(f"{len(items)} items, {len(rows)} done ({len(pending)} to judge), {len(todo)} to run; "
          f"model {args.model}, judge {args.judge}", flush=True)  # fmt: skip
    pacer = shared_pacer or Pacer()
    stopped, retry_after = None, None

    def spend(model: str, t_in: int, t_out: int) -> None:
        add_tokens(run_tokens, model, t_in, t_out)
        add_tokens(all_tokens, model, t_in, t_out)

    def judged(item: dict, result: dict, chunks: list[dict]) -> dict | None:
        spent: dict = {}
        out = judge(item, result, chunks, args.judge, pacer, not args.live, spent)
        for model, t in spent.items():
            spend(model, t["in"], t["out"])
        if out and out.get("error"):
            secs = retry_after_s(out["error"])
            if "429" in out["error"]:
                raise BudgetExhausted(f"judge {item['id']}: {out['error']}", secs)
        return out

    retriever = shared_retriever or load_retriever(args.config)
    try:
        for row in pending:
            if row["id"] not in by_id:
                continue
            chunks = [retriever.chunks[c] for c in row["context"]]
            new = judged(by_id[row["id"]], row, chunks)
            if (row.get("judge") or {}).get("statements"):  # kept for judge agreement
                row["judge_prev"] = {"model": row.get("judge_model") or "openai/gpt-oss-20b",
                                     "statements": row["judge"]["statements"]}  # fmt: skip
            row["judge"], row["judge_model"] = new, args.judge
            _save(out_path, args, rows, all_tokens, None, None)
        answerer = Answerer(retriever, model=args.model, keep_raw=True, entail_model=args.entail,
                            stream_factory=GroqStream if args.live else CachedStream,
                            before_retry=lambda: time.sleep(61))  # fmt: skip
        if todo:
            answerer.warm_up()
        for n, item in enumerate(todo, start=1):
            pacer.wait(args.model)
            pacer.wait("qwen/qwen3.8-27b", 900)  # the rewrite call
            # Groq appears to reserve max_completion_tokens against this limit: only a clear
            # window is safe, so this waits until the last call is a minute old
            pacer.wait(f"{args.model}:out", OTPM_BUDGET + 1, OTPM_BUDGET)
            result = answerer.ask(item["question"])
            flags = result["validation"]["flags"]
            problems = [f for f in flags if f.startswith(("generation_failed", "stream_failed"))]
            rw = retriever.last_rewrite
            if rw is not None and rw.error and rw.error != "rewrite deadline":
                problems.append(f"rewrite_failed: {rw.error}")
            if problems:  # a provider problem, not a result: the item is run again on resume
                raise BudgetExhausted(f"{item['id']}: {problems[0]}", retry_after_s(problems[0]))
            meta = result["meta"]
            gen_cached = meta["latency"].get("gen_first_delta_ms") is None and not args.live
            tr = retriever.last_trace
            pacer.record(args.model, meta["tokens_in"] + meta["tokens_out"])
            pacer.record(f"{args.model}:out", meta["tokens_out"])
            spend(
                tr.rewrite_model or "qwen/qwen3.8-27b", tr.rewrite_tokens_in, tr.rewrite_tokens_out
            )
            if not gen_cached:
                spend(meta["model"], meta["tokens_in"] - tr.rewrite_tokens_in,
                      meta["tokens_out"] - tr.rewrite_tokens_out)  # fmt: skip
            sources = answerer_context(answerer, retriever)
            gold = {(g["doc_id"], g["page"]) for g in item["gold"]}
            row = {
                "id": item["id"], "lang": item["lang"], "type": item["type"],
                "slice": item["slice"], "question": item["question"], "model": meta["model"],
                "abstained": result["abstained"], "abstain_reason": result["abstain_reason"],
                "answer": result["answer"], "schemes": result["schemes"],
                "follow_up_question": result["follow_up_question"],
                "citations": [c["chunk_id"] for c in result["citations"]],
                "context": [c["chunk_id"] for c in sources],
                "context_hit": bool(gold & {(c["doc_id"], c["page"]) for c in sources})
                if gold else None,
                "dropped": result["validation"]["dropped"], "flags": flags,
                "raw_generation": result.get("raw_generation"),
                "lang_match": answer_script(result["answer"])
                == ("en" if item["lang"] == "en" else "ml"),
                "prompt": ANSWER_VERSION,
                "ttfc_ms": meta["ttfc_ms"], "ttft_ms": meta["ttft_ms"],
                "latency_ms": meta["latency_ms"],
                "latency": meta["latency"], "live": args.live and not gen_cached,
                "tokens_in": meta["tokens_in"], "tokens_out": meta["tokens_out"],
                "cost_inr": meta["cost_inr"],
            }  # fmt: skip
            if item.get("expected"):
                row["eligibility"] = eligibility_correct(item, result)
            rows.append(row)
            _save(out_path, args, rows, all_tokens, None, None)  # the generation is never lost
            if not args.no_judge and not result["abstained"]:
                row["judge"], row["judge_model"] = judged(item, result, sources), args.judge
                _save(out_path, args, rows, all_tokens, None, None)
            j = row.get("judge") or {}
            sup = (f"{sum(s['supported'] for s in j['statements'])}/{len(j['statements'])}"
                   if j.get("statements") else "-")  # fmt: skip
            print(f"  [{n}/{len(todo)}] {item['id']} {item['lang']:8} abst={row['abstained']!s:5} "
                  f"ttfc={row['ttfc_ms']} ttft={row['ttft_ms']} full={row['latency_ms']} ms "
                  f"supported={sup} drops={len(row['dropped'])}", flush=True)  # fmt: skip
    except BudgetExhausted as exc:
        stopped = f"stopped (Groq quota / provider failure): {exc.args[0]}"
        retry_after = exc.args[1] if len(exc.args) > 1 else None
        print(stopped, flush=True)
    finally:
        if shared_retriever is None:
            retriever.close()
    report = _save(out_path, args, rows, all_tokens, stopped, retry_after)
    with TOKEN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "file": out_path.name, "items_total": len(rows),
                            "stopped": bool(stopped), "tokens": run_tokens}) + "\n")  # fmt: skip
    print(json.dumps(report["summary"], indent=1))
    print(f"tokens this run: {json.dumps(run_tokens)}")
    print(f"tokens this file: {json.dumps(all_tokens)}")
    print(f"wrote {out_path}")
    if stopped:
        print(f"RESUME: {_resume_cmd(args, out_path)}")
    return 2 if stopped else 0


def answerer_context(answerer, retriever) -> list[dict]:  # noqa: ANN001
    """The excerpts the generator saw for the last question (top k of the last search)."""
    return [retriever.chunks[cid] for cid in answerer.last_context]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--judge", default="openai/gpt-oss-120b")
    ap.add_argument("--all", action="store_true", help="every gold question + eligibility")
    ap.add_argument("--limit", type=int, help="run at most this many new items")
    ap.add_argument("--config", default="default")
    ap.add_argument("--n", type=int, default=24, help="answerable gold questions (stratified)")
    ap.add_argument("--no-answer", type=int, default=8)
    ap.add_argument("--no-eligibility", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--live", action="store_true", help="no caches, paced: real latency")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--entail", help="model for the optional entailment check")
    ap.add_argument("--resume", help="earlier run JSON: reuse its items for this model")
    ap.add_argument("--only", help="comma-separated item ids")
    ap.add_argument("--tag")
    args = ap.parse_args()
    if args.all:
        args.n, args.no_answer = 10_000, 10_000
    if args.judge == args.model:
        raise SystemExit("the judge must be a different model from the generator")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
