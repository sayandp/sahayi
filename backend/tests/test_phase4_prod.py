"""Production behaviour: generator fallback, card filter (h015), answer cache, degrade mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app import cache as cache_mod
from backend.app.answer import Answerer
from backend.app.cache import AnswerCache, is_personal, normalise
from backend.app.llm import client as client_mod
from backend.app.llm.client import AllGeneratorsBusy, FallbackStream, LLMError, chat_json
from backend.app.validate import Validator, filter_cards, is_rule_chunk
from backend.tests.test_phase3 import C1, C2, CHUNKS, GOOD, FakeRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "h015.json"
OTPM_429 = "HTTP 429: output tokens per minute (OTPM): Limit 1000"


class Scripted:
    def __init__(self, text: str = "", error: str | None = None, after: int = 0) -> None:
        self.text_out, self.error, self.after = text, error, after
        self.text, self.tokens_in, self.tokens_out = "", 11, 7
        self.ttft_ms, self.throttled_ms = 1.0, 0.0

    def __iter__(self):
        for i, ch in enumerate(self.text_out):
            if self.error and i == self.after:
                raise LLMError(self.error)
            self.text += ch
            yield ch
        if self.error and not self.text_out:
            raise LLMError(self.error)


def chain_of(*scripts):
    queue = list(scripts)
    factories = {
        "groq": lambda m, model, t: queue.pop(0),
        "gemini": lambda m, model, t: queue.pop(0),
    }
    chain = [("groq", "qwen"), ("groq", "oss20"), ("gemini", "flash")][: len(scripts)]
    return FallbackStream([], chain, factories=factories)


@pytest.mark.parametrize("error", [OTPM_429, "HTTP 429: tokens per day", "ReadTimeout: timed out",
                                   "HTTP 503: down"])  # fmt: skip
def test_fallback_switches_before_first_text_and_records_model(error):
    s = chain_of(Scripted(error=error), Scripted("hello"))
    assert "".join(s) == "hello"
    assert s.model == "oss20" and s.tokens_in == 11 and s.text == "hello"
    assert s.skipped[0].startswith("qwen: ")


def test_fallback_reaches_gemini_then_reports_all_busy():
    s = chain_of(Scripted(error=OTPM_429), Scripted(error=OTPM_429), Scripted("ok"))
    assert "".join(s) == "ok" and s.model == "flash" and len(s.skipped) == 2
    s = chain_of(Scripted(error=OTPM_429), Scripted(error="ReadTimeout"))
    with pytest.raises(AllGeneratorsBusy):
        list(s)


def test_no_switch_after_text_was_sent():
    s = chain_of(Scripted("abc", error="HTTP 500", after=2), Scripted("never"))
    with pytest.raises(LLMError) as exc:
        list(s)
    assert not isinstance(exc.value, AllGeneratorsBusy) and s.model == "qwen"


def test_gemini_stream_parses_sse(monkeypatch):
    lines = ["data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": t}]}}],
                                    "usageMetadata": {"promptTokenCount": 9,
                                                      "candidatesTokenCount": 2}})
             for t in ("he", "llo")]  # fmt: skip

    class Resp:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def iter_lines(self):
            yield from lines

    monkeypatch.setattr(client_mod.httpx, "stream", lambda *a, **k: Resp())
    monkeypatch.setattr(client_mod.get_settings(), "gemini_api_key", "k")
    s = client_mod.GeminiStream([{"role": "user", "content": "x"}], "gemini-3.6-flash")
    assert "".join(s) == "hello" and (s.tokens_in, s.tokens_out) == (9, 2)


def test_chat_json_never_uses_gemini_unless_asked(monkeypatch):
    """Eval judges must stop on a Groq quota error, not change provider silently."""
    settings = client_mod.get_settings()
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "gemini_api_key", "k")

    def fail(*a, **k):
        raise LLMError("HTTP 429: tokens per day")

    called = []
    monkeypatch.setattr(client_mod, "groq_chat", fail)
    monkeypatch.setattr(client_mod, "gemini_chat", lambda *a, **k: called.append(1) or fail())
    with pytest.raises(LLMError):
        chat_json([], groq_model="m")
    assert not called


def _answerer(stream, **kw):
    return Answerer(
        FakeRetriever([C1, C2]), model="qwen", stream_factory=lambda m, model: stream, **kw
    )


def test_model_used_is_in_meta():
    s = chain_of(Scripted(error=OTPM_429), Scripted(GOOD))
    meta = _answerer(s).ask("widow pension?")["meta"]
    assert meta["model"] == "oss20" and meta["fallbacks"][0].startswith("qwen")


# ------------------------------------------------------------------ h015 fixture


def test_h015_keeps_only_the_scheme_the_answer_rests_on():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chunks, data = fx["chunks"], json.loads(fx["raw_generation"])
    aliases = {f"C{i}": c for i, c in enumerate(fx["context"], start=1)}
    scheme_ids = {c: c.split("--")[0] for c in fx["context"]}
    v = Validator(aliases, chunks, fx["question"]).validate(data, scheme_ids)
    before = [s["scheme_id"] for s in v.schemes]
    assert {"kerala_pension_old_age", "kerala_pension_unmarried_women", "st_scholarships"} <= set(
        before
    )
    cited = list(dict.fromkeys(c for s in v.sentences for c in s["citations"]))
    kept, dropped = filter_cards(v.schemes, cited, fx["context"], chunks, scheme_ids)
    assert [s["scheme_id"] for s in kept] == ["sjd_parinayam"]
    assert set(dropped) == {
        "kerala_pension_old_age",
        "kerala_pension_unmarried_women",
        "st_scholarships",
    }


def test_card_filter_when_answer_cites_nothing_uses_top_three_schemes():
    ids = [f"s{i}--d:1:0" for i in range(4)]
    chunks = {c: {"text": "eligibility rule", "section": "eligibility"} for c in ids}
    scheme_ids = {c: c.split("--")[0] for c in ids}
    cards = [{"scheme_id": f"s{i}", "citations": [ids[i]]} for i in range(4)]
    kept, dropped = filter_cards(cards, [], ids, chunks, scheme_ids)
    assert [c["scheme_id"] for c in kept] == ["s0", "s1", "s2"] and dropped == ["s3"]


def test_rule_chunk_detection():
    assert is_rule_chunk({"section": "eligibility", "text": "x"})
    assert not is_rule_chunk({"section": "documents", "text": "income certificate"})
    assert is_rule_chunk({"section": None, "text": "വാർഷിക വരുമാനം ഒരു ലക്ഷം"})
    assert not is_rule_chunk({"section": None, "text": "office address"})


# ------------------------------------------------------------------ cache


RESULT = {"answer": "a", "schemes": [], "follow_up_question": None, "meta": {"latency_ms": 5}}


def test_normalise_and_personal():
    assert normalise("  What IS  PM-Kisan? ") == "what is pm-kisan"
    for q in ("I am a widow, pension?", "njan oru karshakan", "എനിക്ക് പെൻഷൻ കിട്ടുമോ?",
              "pension for age 65", "എന്റെ അമ്മ"):  # fmt: skip
        assert is_personal(q), q
    assert not is_personal("What documents does PM-KISAN need?")


def test_cache_roundtrip_ttl_and_keys(tmp_path, monkeypatch):
    c = AnswerCache(tmp_path, "corpus1", ttl_days=7)
    assert c.put("What is PMAY?", "en", {"sources": []}, RESULT)
    assert c.get("what is pmay", "en")["result"]["answer"] == "a"
    assert c.get("What is PMAY?", "ml") is None
    assert AnswerCache(tmp_path, "corpus2").get("What is PMAY?", "en") is None
    now = cache_mod.time.time()
    monkeypatch.setattr(cache_mod.time, "time", lambda: now + 8 * 86400)
    assert c.get("What is PMAY?", "en") is None and not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("question,result", [
    ("I am 65, pension?", RESULT),
    ("What is PMAY?", RESULT | {"degraded": True}),
    ("What is PMAY?", RESULT | {"abstain_reason": "error"}),
    ("What is PMAY?", RESULT | {"follow_up_question": "Are you a farmer?"}),
    ("What is PMAY?", RESULT | {"schemes": [{"eligible": True}]}),
])  # fmt: skip
def test_cache_stores_nothing_personal_or_broken(tmp_path, question, result):
    c = AnswerCache(tmp_path, "c")
    assert not c.put(question, "en", {}, result) and not list(tmp_path.iterdir())


def test_answerer_serves_second_ask_from_cache_without_generating(tmp_path):
    calls = []

    def factory(m, model):
        calls.append(1)
        return Scripted(GOOD.replace('"follow_up_question": "Do you have a ration card?"',
                                     '"follow_up_question": null'))  # fmt: skip

    a = Answerer(FakeRetriever([C1, C2]), model="qwen", stream_factory=factory,
                 cache=AnswerCache(tmp_path, "c"))  # fmt: skip
    first = a.ask("What pension do widows get?")
    assert not first["meta"]["cached"]
    events = list(a.stream("what pension do widows get"))
    assert [n for n, _ in events] == ["sources", "token", "result"]
    assert events[-1][1]["meta"]["cached"] and len(calls) == 1


# ------------------------------------------------------------------ degrade


def test_all_generators_busy_returns_sources_and_cards_with_notice():
    s = chain_of(Scripted(error=OTPM_429), Scripted(error=OTPM_429), Scripted(error="timeout"))
    events = list(_answerer(s, degrade=True).stream("widow pension?"))
    names = [n for n, _ in events]
    assert names == ["sources", "token", "result"]
    result = events[-1][1]
    assert result["degraded"] and not result["abstained"] and "busy" in result["answer"]
    assert {c["scheme_id"] for c in result["schemes"]} == {"pension", "pm_kisan"}
    card = next(c for c in result["schemes"] if c["scheme_id"] == "pm_kisan")
    assert card["benefit"].endswith(f"[{C2}]") and CHUNKS[C2]["text"][:20] in card["benefit"]
    assert [c["chunk_id"] for c in result["citations"]] == [C1, C2]


def test_without_degrade_the_eval_path_still_abstains():
    s = chain_of(Scripted(error=OTPM_429))
    result = _answerer(s).ask("widow pension?")
    assert result["abstained"] and result["abstain_reason"] == "error" and not result["degraded"]


class Cut(Scripted):
    """A completion stopped at the token limit."""

    finish_reason = "length"


def _twice(first, second, **kw):
    queue = [first, second]
    seen = []

    def factory(messages, model):
        seen.append(messages)
        return queue.pop(0)

    return Answerer(FakeRetriever([C1, C2]), model="qwen", stream_factory=factory, **kw), seen


def test_cut_off_output_retries_once_compact_and_never_parses_partial_json():
    waited = []
    a, seen = _twice(Cut(GOOD[:60]), Scripted(GOOD), before_retry=lambda: waited.append(1))
    result = a.ask("widow pension?")
    assert not result["abstained"] and "1,600" in result["answer"] and waited == [1]
    assert "compact_retry" in result["validation"]["flags"]
    assert "cut off" in seen[1][0]["content"] and "cut off" not in seen[0][0]["content"]
    assert result["meta"]["tokens_in"] == 22  # both attempts are counted


def test_even_a_complete_looking_cut_off_answer_is_not_parsed():
    a, _ = _twice(Cut(GOOD), Cut(GOOD))
    result = a.ask("widow pension?")
    assert result["abstained"] and result["abstain_reason"] == "truncated"
    assert "truncated_failed" in result["validation"]["flags"]
    assert result["citations"] == []


def test_production_shows_busy_mode_when_still_cut_off():
    a, _ = _twice(Cut(GOOD[:60]), Cut(GOOD[:60]), degrade=True)
    result = a.ask("widow pension?")
    assert result["degraded"] and "truncated_failed" in result["validation"]["flags"]


def test_groq_stream_reports_finish_reason(monkeypatch):
    from backend.tests.test_phase3 import _FakeResponse

    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}),
             "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]})]  # fmt: skip
    monkeypatch.setattr(client_mod.httpx, "stream", lambda *a, **k: _FakeResponse(200, lines))
    monkeypatch.setattr(client_mod.get_settings(), "groq_api_key", "k")
    s = client_mod.GroqStream([], "qwen/qwen3.8-27b")
    assert "".join(s) == "x" and s.finish_reason == "length" and s.max_tokens == 900


def test_production_chain_caps_groq_output_tokens(monkeypatch):
    seen = {}

    class Fake(Scripted):
        def __init__(self, messages, model, **kw):
            super().__init__("ok")
            seen.update(kw)

    monkeypatch.setattr(client_mod, "GroqStream", Fake)
    assert "".join(FallbackStream([], [("groq", "qwen")])) == "ok"
    assert seen["retries"] == 0 and "max_tokens" not in seen  # GroqStream's default: 900


def test_retrieval_only_answer_makes_no_llm_call_and_shows_sources_and_cards():
    from backend.app.production import NoGenerator

    a = Answerer(FakeRetriever([C1, C2]), model="none", degrade=True,
                 stream_factory=lambda m, model: NoGenerator())  # fmt: skip
    events = list(a.stream("What does PM-KISAN pay?"))
    assert [n for n, _ in events] == ["sources", "token", "result"]
    result = events[-1][1]
    assert result["degraded"] and {c["scheme_id"] for c in result["schemes"]} == {
        "pension",
        "pm_kisan",
    }
    assert result["meta"]["tokens_in"] == 0


def test_static_frontend_is_served_after_api_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main as main_mod

    (tmp_path / "index.html").write_text("<title>ui</title>", encoding="utf-8")
    monkeypatch.setattr(main_mod.get_settings(), "static_dir", str(tmp_path))
    client = TestClient(
        main_mod.create_app(answerer_factory=lambda: None, chunk_loader=lambda: CHUNKS)
    )
    assert "ui" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"
    assert client.get(f"/source/{C1}").json()["chunk_id"] == C1


def test_auto_resume_lock_is_exclusive_and_released(tmp_path):
    from evals.auto_resume import acquire_lock

    path = tmp_path / "x.lock"
    first = acquire_lock(path)
    assert first is not None and acquire_lock(path) is None
    first.close()
    second = acquire_lock(path)
    assert second is not None
    second.close()


def test_auto_resume_sees_a_driver_it_did_not_start():
    from evals.auto_resume import driver_running

    cmd = [["python.exe", "evals/run_until_done.py", "a.json"]]
    assert driver_running(cmd) and not driver_running([["python.exe", "tasks.py", "serve"]])


def test_shared_retriever_is_left_open_for_the_next_turn(tmp_path, monkeypatch):
    import json

    from evals import answer_eval, run_until_done

    class Retriever:
        chunks: dict = {}
        closed = False

        def close(self) -> None:
            self.closed = True

    path = tmp_path / "run.json"
    head = {"model": "m", "judge": "j", "config": "default", "tokens_used": {}, "items": []}
    path.write_text(json.dumps(head), encoding="utf-8")
    monkeypatch.setattr(answer_eval, "load_items", lambda *a: [])
    monkeypatch.setattr(answer_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(answer_eval, "TOKEN_LOG", tmp_path / "tokens.jsonl")
    retriever = Retriever()
    args = run_until_done.turn_args(head, path)
    assert answer_eval.run(args, retriever, answer_eval.Pacer()) == 0
    assert not retriever.closed


def test_memory_log_records_peak(tmp_path):
    import json

    from evals.run_until_done import log_memory

    path = tmp_path / "rss.jsonl"
    mem = log_memory("turn", path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["turn"] == "turn" and row["peak_rss_gb"] == mem["peak_rss_gb"] > 0


def test_auto_resume_treats_finish_steps_as_drivers():
    from evals.auto_resume import driver_running

    assert driver_running([["python.exe", "evals/gate2_finish.py", "a.json"]])


def test_gate2_report_says_interim_and_picks_no_primary(tmp_path):
    from evals.answer_eval import summarize
    from evals.gate2_report import report

    run = {"model": "openai/gpt-oss-20b", "judge": "j", "items": [], "summary": summarize([])}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    text = report([path])
    assert "INTERIM" in text and "Production primary: none" in text


def demo_client(monkeypatch, **settings):
    from fastapi.testclient import TestClient

    from backend.app import main as main_mod

    for key, value in {"demo_mode": True, "demo_rate_limit_per_min": 2, **settings}.items():
        monkeypatch.setattr(main_mod.get_settings(), key, value)

    def answerer():
        return Answerer(FakeRetriever([C1, C2]), stream_factory=lambda m, model: Scripted(GOOD))

    return TestClient(
        main_mod.create_app(answerer_factory=answerer, chunk_loader=lambda: CHUNKS, log_path=None)
    )


def test_demo_rate_limit_is_per_visitor_and_not_sse(monkeypatch):
    client = demo_client(monkeypatch)
    one, two = {"cf-connecting-ip": "1.1.1.1"}, {"cf-connecting-ip": "2.2.2.2"}
    body = {"question": "Who can apply?"}
    first = client.post("/ask", json=body, headers=one)
    assert first.headers["content-type"].startswith("text/plain")
    assert "event: result" in first.text
    assert client.post("/ask", json=body, headers=one).status_code == 200
    assert client.post("/ask", json=body, headers=one).status_code == 429
    assert client.post("/ask", json=body, headers=two).status_code == 200


def test_demo_hides_api_docs_and_key_status(monkeypatch):
    client = demo_client(monkeypatch)
    assert client.get("/docs").status_code == 404
    assert "groq_key" not in client.get("/health").json()


def test_normal_mode_still_streams_sse(monkeypatch):
    client = demo_client(monkeypatch, demo_mode=False, rate_limit_per_min=5)
    res = client.post("/ask", json={"question": "Who can apply?"})
    assert res.headers["content-type"].startswith("text/event-stream")


def test_daily_cap_survives_restart_and_resets_next_day(tmp_path):
    from backend.app.production import DailyCap

    path = tmp_path / "usage.json"
    assert DailyCap(2, path).take("2026-11-02") and DailyCap(2, path).take("2026-11-02")
    assert not DailyCap(2, path).take("2026-11-02")
    assert DailyCap(2, path).take("2026-11-03")
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"date", "count"}


def test_past_the_cap_makes_no_llm_call_and_cache_hit_is_free(tmp_path):
    from dataclasses import dataclass

    from backend.app.production import DailyCap, DemoAnswerer

    @dataclass
    class Config:
        rewrite: bool = True
        overlap_rewrite: bool = True

    retriever = FakeRetriever([C1, C2])
    retriever.config = Config()
    calls = []

    def factory(messages, model):
        calls.append(model)
        return Scripted(GOOD)

    class StoreAll:  # only the hit/miss path matters here, not what AnswerCache would store
        stored: dict = {}

        def get(self, question, lang):  # noqa: ANN001, ANN201
            return self.stored.get(question)

        def put(self, question, lang, sources, result):  # noqa: ANN001, ANN201
            self.stored[question] = {"sources": sources, "result": result}

    cache = StoreAll()
    answerer = DemoAnswerer(retriever, cap=DailyCap(1, tmp_path / "u.json"), stream_factory=factory,
                            degrade=True, cache=cache)  # fmt: skip
    first = answerer.ask("Who can apply for the scheme?")
    assert len(calls) == 1 and not first.get("degraded")
    assert answerer.ask("Who can apply for the scheme?")["meta"]["cached"]  # free
    busy = answerer.ask("What documents does the scheme need?")
    assert len(calls) == 1 and busy["degraded"] and retriever.config.rewrite is False
