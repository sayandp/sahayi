"""Phase 3: citation validation, answer streaming, abstain path, API. Groq is mocked."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from backend.app import answer as answer_mod
from backend.app.answer import Answerer, AnswerFieldStream, SentenceGate, parse_json
from backend.app.llm import client as client_mod
from backend.app.llm.client import GroqStream, LLMError, LLMResult
from backend.app.llm.prompts import ABSTAIN_TOKEN
from backend.app.retrieval import rerank as rerank_mod
from backend.app.retrieval.rerank import RerankSpec, RerankUnavailable
from backend.app.retrieval.search import SearchTrace
from backend.app.validate import (
    Validator,
    normalise_numbers,
    numbers_in,
    split_sentences,
    unsupported_numbers,
)

C1 = "pension--doc-aaa:2:0"
C2 = "pm_kisan--pmk-bbb:1:0"
CHUNKS = {
    C1: {"chunk_id": C1, "doc_id": "pension--doc-aaa", "scheme": "Widow Pension",
         "department": "Social Justice Department", "source_url": "https://example.gov.in/p.pdf",
         "page": 2, "lang": "ml", "downloaded_at": "2026-10-12", "section": None,
         "text": "60 വയസ്സിന് മുകളിലുള്ള വിധവകൾക്ക് പ്രതിമാസം 1,600 രൂപ പെൻഷൻ ലഭിക്കും. "
                 "കുടുംബ വാർഷിക വരുമാനം 1 lakh രൂപയിൽ താഴെയായിരിക്കണം."},
    C2: {"chunk_id": C2, "doc_id": "pm_kisan--pmk-bbb", "scheme": "PM-KISAN",
         "department": "Agriculture", "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
         "page": 1, "lang": "en", "downloaded_at": "2026-10-11", "section": "benefits",
         "text": "Rs 6000 per year is paid in three equal instalments of Rs 2000."},
}  # fmt: skip
ALIASES = {"C1": C1, "C2": C2}
SCHEME_IDS = {C1: "pension", C2: "pm_kisan"}


def validator(user_text: str = "") -> Validator:
    return Validator(ALIASES, CHUNKS, user_text)


# ------------------------------------------------------------- numbers / sentences


def test_malayalam_numerals_and_currency_normalised():
    assert "1600" in numbers_in("പ്രതിമാസം ൧൬൦൦ രൂപ")
    assert numbers_in("₹1,20,000 or Rs.6000/-") == {"120000", "6000"}
    assert {"1.5", "150000"} <= numbers_in("1.5 lakh")
    assert "6000" in normalise_numbers("Rs 6,000")


def test_number_supported_in_malayalam_digits_and_words():
    src = [CHUNKS[C2]["text"]]
    assert unsupported_numbers("You get ₹6,000 in 3 instalments [C2].", src) == []
    assert unsupported_numbers("You get ₹8,000 a year [C2].", src) == ["8000"]
    assert unsupported_numbers("പ്രതിമാസം ൧൬൦൦ രൂപ [C1]", [CHUNKS[C1]["text"]]) == []
    assert unsupported_numbers("income below 100000 [C1]", [CHUNKS[C1]["text"]]) == []


def test_split_sentences_keeps_tags_and_abbreviations():
    s = split_sentences(
        "It pays Rs. 6000 a year. [C2] Paid in parts [C2]. Are you 60?\n- Apply [C1]"
    )
    assert s == [
        "It pays Rs. 6000 a year. [C2]",
        "Paid in parts [C2].",
        "Are you 60?",
        "- Apply [C1]",
    ]


# ------------------------------------------------------------------- validation


def test_fake_ids_dropped_and_number_mismatch_dropped():
    v = validator().validate(
        {"answer": "Widows over 60 get ₹1,600 a month [C1]. They also get ₹5,000 [C1]. "
                   "PM-KISAN pays Rs 6000 [C9]. Something with no tag.",
         "schemes": []},
        SCHEME_IDS,
    )  # fmt: skip
    assert [s["citations"] for s in v.sentences] == [[C1]]
    reasons = {d.reason for d in v.dropped}
    assert reasons == {"number_mismatch", "unknown_citation", "no_citation"}
    assert v.unknown_citations == ["C9"]
    assert f"[{C1}]" in v.answer and "5,000" not in v.answer


def test_real_chunk_id_tag_accepted_and_user_numbers_allowed():
    v = validator("I am a 65 year old widow").validate(
        {"answer": f"At 65 you qualify for 1,600 a month [{C1}].", "schemes": []}, SCHEME_IDS
    )
    assert v.sentences and v.sentences[0]["citations"] == [C1]
    v2 = validator().validate({"answer": "At 65 you qualify [C1].", "schemes": []}, SCHEME_IDS)
    assert v2.empty and v2.dropped[0].reason == "number_mismatch"


def test_scheme_cards_validated():
    data = {"answer": "x [C1].", "schemes": [
        {"scheme_id": "pension", "name": "Widow Pension", "eligible": True,
         "reasons": ["Over 60 [C1]", "Owns 3 cars [C1]"], "benefit": "₹1,600 a month",
         "citations": ["C1"]},
        {"scheme_id": "made_up", "name": "Fake", "eligible": True, "citations": ["C7"]},
        {"scheme_id": "pm_kisan", "name": "PM-KISAN", "eligible": False,
         "reasons": ["Needs 9 acres [C2]"], "citations": ["C2"]},
    ]}  # fmt: skip
    v = validator().validate(data, SCHEME_IDS)
    by_id = {s["scheme_id"]: s for s in v.schemes}
    assert set(by_id) == {"pension", "pm_kisan"}
    assert by_id["pension"]["reasons"] == [f"Over 60 [{C1}]"]  # fields keep their own tags
    assert by_id["pension"]["eligible"] is True
    assert by_id["pension"]["benefit"] == f"₹1,600 a month [{C1}]"  # untagged: card's C1
    assert by_id["pm_kisan"]["eligible"] is None  # its only reason failed the number check
    assert any(f.startswith("scheme_dropped_unknown_id") for f in v.flags)


# ----------------------------------------------------------------- streaming


def test_answer_field_stream_decodes_split_escapes():
    raw = json.dumps({"answer": 'വിധവ "pension" ₹1,600\nok', "schemes": []}, ensure_ascii=True)
    s, out = AnswerFieldStream(), ""
    for i in range(0, len(raw), 3):
        out += s.feed(raw[i : i + 3])
    assert out == 'വിധവ "pension" ₹1,600\nok' and s.done


def test_sentence_gate_waits_for_tags_and_filters():
    gate = SentenceGate(validator())
    assert gate.feed("Widows get 1,600 a month.") == []  # tags may still follow
    assert gate.feed(" [C1") == []
    assert gate.feed("] They get 9,999 [C1]. Next") == [f"Widows get 1,600 a month. [{C1}]"]
    assert gate.flush() == []  # 9,999 fails the number check; "Next" has no tag


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"answer": "a"}\n```') == {"answer": "a"}
    assert parse_json("no json here") is None


# ------------------------------------------------------------------ answerer


@dataclass
class FakeHit:
    chunk_id: str
    score: float
    chunk: dict


class FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.chunks = CHUNKS
        self.last_trace = SearchTrace(rewrite_lang="en")
        self.last_rewrite = None

    def search(self, query: str) -> list[FakeHit]:
        return [FakeHit(i, 1.0, CHUNKS[i]) for i in self.ids]


class FakeStream:
    """Stands in for GroqStream: yields a canned completion in small deltas."""

    def __init__(self, text: str, fail: bool = False) -> None:
        self.full, self.fail = text, fail
        self.text, self.tokens_in, self.tokens_out = "", 1000, 100
        self.ttft_ms, self.throttled_ms = 5.0, 0.0

    def __iter__(self):
        if self.fail:
            raise LLMError("HTTP 503")
        for i in range(0, len(self.full), 7):
            self.text += self.full[i : i + 7]
            yield self.full[i : i + 7]


def make_answerer(text: str, ids=(C1, C2), fail=False, fallback=None) -> Answerer:
    def no_fallback(*a, **k):
        raise LLMError("no fallback in this test")

    return Answerer(FakeRetriever(list(ids)), model="openai/gpt-oss-20b",
                    stream_factory=lambda messages, model: FakeStream(text, fail),
                    json_fallback=fallback or no_fallback)  # fmt: skip


GOOD = json.dumps({
    "answer": "Widows over 60 get ₹1,600 a month [C1]. Do you have a ration card?",
    "abstained": False, "follow_up_question": "Do you have a ration card?",
    "schemes": [{"scheme_id": "pension", "name": "Widow Pension", "eligible": None,
                 "reasons": [], "benefit": "₹1,600 a month [C1]", "citations": ["C1"]}],
}, ensure_ascii=False)  # fmt: skip


def test_event_order_and_result_shape():
    events = list(make_answerer(GOOD).stream("I am a widow, any pension?"))
    names = [n for n, _ in events]
    assert names[0] == "sources" and names[-1] == "result"
    assert set(names[1:-1]) == {"token"} and len(names) >= 3
    result = events[-1][1]
    assert not result["abstained"] and result["citations"][0]["chunk_id"] == C1
    assert result["citations"][0]["link"].endswith("#page=2")  # PDF source
    assert result["as_of"] == "2026-10-12"
    assert result["meta"]["tokens_in"] == 1000 and result["meta"]["cost_inr"] > 0
    streamed = "".join(d["text"] for n, d in events if n == "token")
    assert "1,600" in streamed and C1 in streamed


def test_abstain_token_path_gives_department_contact(monkeypatch):
    monkeypatch.setattr(answer_mod, "contact_for", lambda sid, lang: f"CONTACT[{sid}]")
    text = json.dumps({"answer": ABSTAIN_TOKEN, "abstained": True, "schemes": []})
    events = list(make_answerer(text).stream("What is the capital of France?"))
    result = events[-1][1]
    assert result["abstained"] and result["abstain_reason"] == "model"
    assert "CONTACT[pension]" in result["answer"] and result["citations"] == []
    tokens = [d["text"] for n, d in events if n == "token"]
    assert tokens == [result["answer"]]  # the abstain token itself is never streamed


def test_all_claims_invalid_abstains():
    text = json.dumps({"answer": "You get ₹9,000 [C1]. Also [C8].", "schemes": []})
    result = make_answerer(text).ask("pension?")
    assert result["abstained"] and result["abstain_reason"] == "validation"
    assert {d["reason"] for d in result["validation"]["dropped"]} == {
        "number_mismatch",
        "unknown_citation",
    }


def test_stream_failure_falls_back_to_json_mode():
    calls = []

    def fallback(messages, groq_model, retries=3, max_tokens=None):
        calls.append(groq_model)
        return LLMResult(GOOD, "groq", groq_model, 10.0, 900, 90, data=json.loads(GOOD))

    result = make_answerer("", fail=True, fallback=fallback).ask("pension?")
    assert calls == ["openai/gpt-oss-20b"] and not result["abstained"]
    assert any(f.startswith("stream_failed") for f in result["validation"]["flags"])


def test_no_context_abstains_without_llm_call():
    result = make_answerer(GOOD, ids=()).ask("anything")
    assert result["abstained"] and "no_context" in result["validation"]["flags"]


# ------------------------------------------------------------------ Groq stream


class _FakeResponse:
    def __init__(self, status: int, lines: list[str], headers: dict | None = None) -> None:
        self.status_code, self.lines, self.headers = status, lines, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self.lines

    def read(self):
        return b"rate limited"


def test_groq_stream_parses_deltas_usage_and_retries(monkeypatch):
    chunk = lambda c: "data: " + json.dumps({"choices": [{"delta": {"content": c}}]})  # noqa: E731
    usage = "data: " + json.dumps({"choices": [], "x_groq": {"usage": {
        "prompt_tokens": 12, "completion_tokens": 3}}})  # fmt: skip
    responses = [
        _FakeResponse(429, [], {"retry-after": "0"}),
        _FakeResponse(200, [chunk('{"ans'), chunk('wer": 1}'), usage, "data: [DONE]"]),
    ]
    monkeypatch.setattr(client_mod.httpx, "stream", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(client_mod.get_settings(), "groq_api_key", "test-key")
    s = GroqStream([{"role": "user", "content": "hi"}], "openai/gpt-oss-20b")
    assert "".join(s) == '{"answer": 1}'
    assert (s.tokens_in, s.tokens_out) == (12, 3) and s.ttft_ms is not None


# ------------------------------------------------------------------ rerank fallback


class _Backend:
    def __init__(self, scores=None, exc=None, delay=0.0):
        self.scores, self.exc, self.delay = scores, exc, delay

    def score(self, query, texts):
        import time

        time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.scores[: len(texts)]


@pytest.mark.parametrize("primary", [
    _Backend(exc=RerankUnavailable("no key")), _Backend([9.0, 9.0], delay=0.5)])  # fmt: skip
def test_hosted_rerank_falls_back_to_local(monkeypatch, primary):
    local = RerankSpec("jina", "onnx", "hub:onnx/model_int8.onnx", 256)
    hosted = RerankSpec("jina", "api", None, 256, "jina", timeout_s=0.1, fallback=local)
    backends = {hosted: primary, local: _Backend([0.2, 0.8])}
    monkeypatch.setattr(rerank_mod, "load_backend", lambda spec: backends[spec])
    info: dict = {}
    out = rerank_mod.rerank("q", [("a", "x"), ("b", "y")], hosted, use_cache=False, info=info)
    assert [cid for cid, _ in out] == ["b", "a"]
    assert info["backend"] == "onnx" and info["fallback"]


# ------------------------------------------------------------------ API


@pytest.fixture
def api_client(tmp_path):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    app = create_app(answerer_factory=lambda: make_answerer(GOOD),
                     chunk_loader=lambda: CHUNKS, log_path=tmp_path / "q.jsonl")  # fmt: skip
    return TestClient(app), tmp_path / "q.jsonl"


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_sse_event_order_and_anonymised_log(api_client):
    client, log_file = api_client
    r = client.post("/ask", json={"question": "I am a widow aged 61, pension?"},
                    headers={"X-Request-ID": "abc123"})  # fmt: skip
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc123"
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert events[0][0] == "sources" and events[-1][0] == "result"
    assert all(n == "token" for n, _ in events[1:-1]) and len(events) >= 3
    assert events[-1][1]["meta"]["request_id"] == "abc123"
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert "widow" not in json.dumps(record) and len(record["q_hash"]) == 16
    assert record["cited"] == [C1]


def test_sync_endpoint_matches_schema(api_client):
    client, _ = api_client
    r = client.post("/ask/sync", json={"question": "widow pension?"})
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == "en" and body["schemes"][0]["scheme_id"] == "pension"


def test_source_lookup_and_404(api_client):
    client, _ = api_client
    r = client.get(f"/source/{C1}")
    assert r.status_code == 200 and r.json()["link"] == "https://example.gov.in/p.pdf#page=2"
    html = client.get(f"/source/{C2}").json()
    assert html["link"] == "https://www.myscheme.gov.in/schemes/pm-kisan"  # no #page for HTML
    assert client.get("/source/nope:1:0").status_code == 404


def test_rate_limit_and_health(api_client, monkeypatch):
    from backend.app.main import RateLimiter

    rl = RateLimiter(2)
    assert rl.check("ip", 0.0) is None and rl.check("ip", 1.0) is None
    assert rl.check("ip", 2.0) == pytest.approx(58.0)
    assert rl.check("ip", 61.0) is None
    client, _ = api_client
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/ask/sync", json={"question": "x"}).status_code == 422  # too short


def test_answer_released_before_scheme_cards_finish_streaming():
    streams = []

    def factory(messages, model):
        streams.append(FakeStream(GOOD))
        return streams[-1]

    a = Answerer(FakeRetriever([C1, C2]), model="openai/gpt-oss-20b", stream_factory=factory)
    for name, _ in a.stream("widow pension?"):
        if name == "token":
            # the last answer sentence is out while the scheme cards are still streaming
            assert len(streams[0].text) < len(GOOD)
            break


class SpecRetriever(FakeRetriever):
    """Calls the speculative hook mid-search, like Retriever in overlap mode."""

    def __init__(self, spec_ids: list[str], final_ids: list[str]) -> None:
        super().__init__(final_ids)
        self.spec_ids = spec_ids
        self.on_speculative = None

    def search(self, query: str) -> list[FakeHit]:
        if self.on_speculative:
            self.on_speculative(self.spec_ids)
        return super().search(query)


@pytest.mark.parametrize(("spec_ids", "hit"), [([C1, C2], True), ([C2], False)])
def test_speculative_generation_kept_only_when_context_matches(spec_ids, hit):
    streams = []

    def factory(messages, model):
        streams.append(FakeStream(GOOD))
        return streams[-1]

    a = Answerer(SpecRetriever(spec_ids, [C1, C2]), model="openai/gpt-oss-20b",
                 stream_factory=factory, speculative=True)  # fmt: skip
    result = a.ask("widow pension?")
    assert len(streams) == (1 if hit else 2)
    assert ("speculative_hit" if hit else "speculative_miss") in result["validation"]["flags"]
    assert not result["abstained"] and result["citations"][0]["chunk_id"] == C1


def test_answer_rebuilt_from_cards_when_verdict_sentence_untagged():
    text = json.dumps({"answer": "You are not eligible.", "schemes": [
        {"scheme_id": "pension", "name": "Widow Pension", "eligible": False,
         "reasons": ["Only widows over 60 qualify [C1]"], "citations": ["C1"]}]})  # fmt: skip
    result = make_answerer(text).ask("I am 50, a widow. Pension?")
    assert not result["abstained"]
    assert result["answer"].startswith("Widow Pension (you do not appear eligible): Only widows")
    assert "answer_from_scheme_cards" in result["validation"]["flags"]


def test_slow_rewrite_hits_deadline_and_search_continues(monkeypatch):
    import time as _time
    from dataclasses import replace

    from backend.tests.test_phase2b import _fake_retriever

    r, _ = _fake_retriever(monkeypatch, "en")
    r.config = replace(r.config, rewrite_deadline_s=0.2)

    def slow(q, trace):
        _time.sleep(1.0)
        raise AssertionError("must not be awaited")

    monkeypatch.setattr(r, "_rewrite", slow)
    t0 = _time.perf_counter()
    hits = r.search("orig")
    assert _time.perf_counter() - t0 < 0.8
    assert r.last_trace.rewrite_timeout and r.last_trace.variants == ["orig"]
    assert r.last_rewrite.error == "rewrite deadline" and hits


def test_query_kind_routes_only_manglish_and_mixed_to_rewrite():
    from backend.app.query import query_kind

    assert query_kind("പിഎം കിസാൻ പദ്ധതിയിൽ എത്ര രൂപ ലഭിക്കും?") == "ml"
    assert query_kind("PM-KISAN പദ്ധതിയിൽ പണം കിട്ടുമോ?") == "ml"
    assert query_kind("How much does PM-KISAN pay?") == "en"
    assert query_kind("enikku pension kittumo?") == "manglish"
    assert query_kind("what is the application fee for പെൻഷൻ scheme") == "mixed"


def test_skip_rewrite_makes_no_rewrite_call_for_clean_english(monkeypatch):
    from dataclasses import replace

    from backend.tests.test_phase2b import _fake_retriever

    r, _ = _fake_retriever(monkeypatch, "en")
    r.config = replace(r.config, skip_rewrite_kinds=("en",))

    def no_call(q, trace):
        raise AssertionError("rewrite must not be called")

    monkeypatch.setattr(r, "_rewrite", no_call)
    r.dense.lists["What is the pension?"] = [("a", 0.9), ("b", 0.8)]
    hits = r.search("What is the pension?")
    assert hits and r.last_trace.rewrite_skipped and r.last_rewrite.lang == "en"


def test_ttfc_recorded_and_not_after_ttft():
    result = make_answerer(GOOD).ask("widow pension?")
    assert 0 <= result["meta"]["ttfc_ms"] <= result["meta"]["ttft_ms"]


def test_answer_prompts_forbid_computed_totals_and_missing_fact_reasons():
    from backend.app.llm.prompts import ANSWER_SYSTEM_EN

    assert "compute totals or ranges" in ANSWER_SYSTEM_EN
    assert 'ONLY in "follow_up_question"' in ANSWER_SYSTEM_EN


def test_retry_after_parses_groq_message():
    from evals.answer_eval import retry_after_s

    assert retry_after_s("Please try again in 9m19.44s.") == pytest.approx(559.44)
    assert retry_after_s("Please try again in 1h2m") == pytest.approx(3720)
    assert retry_after_s("HTTP 500") is None
