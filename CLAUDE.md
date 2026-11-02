# CLAUDE.md — Sahayi (Malayalam Govt-Scheme RAG Assistant)

## What this is
Bilingual (Malayalam + English) RAG assistant that tells Kerala residents which government schemes they qualify for. **Every answer cites the source PDF + page. If no chunk supports a claim, the system abstains.** No GPU — LLM via Groq or Gemini API.

## Non-negotiables
1. **Cite or abstain.** Never let the LLM answer from its own knowledge. Every claim in a response maps to a retrieved chunk ID (doc, page). No support → return the abstain message + department contact.
2. **Evals gate changes.** Any change to chunking, embeddings, retrieval, reranking, or prompts must be run through `make eval` and the numbers recorded in `evals/results/`. Don't claim an improvement without a number.
3. **No PII stored.** No login. Don't persist raw user profiles. Logs keep only anonymised query traces (tokens, latency, cost, retrieved IDs).
4. **Freshness.** Every document carries `source_url` and `downloaded_at`; every answer shows an "as of" date.

## Stack
| Layer | Choice |
|---|---|
| Extraction | PyMuPDF; Tesseract (`mal+eng`) for scanned pages |
| Sparse | BM25 (`rank_bm25`), Malayalam-aware tokenisation |
| Dense | `BAAI/bge-m3` over contextual chunk headers ("scheme — department [section]" + text), Qdrant (local) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Query rewrite | `qwen/qwen3.8-27b` on Groq, ML + EN variants; **skipped for Malayalam-script questions** |
| Reranker | `jinaai/jina-reranker-v2-base-multilingual` int8 ONNX, local, top-6 at 256 tokens. Hosted Jina is the `deploy` config, not the default (slower and lower hit@5 here). `BAAI/bge-reranker-v2-m3` is too slow on CPU (~1 s/pair) |
| LLM | Production chain: `qwen/qwen3.8-27b` → `openai/gpt-oss-20b` (Groq) → `gemini-3.6-flash`. Evals use one generator, no fallback. Groq has no free-tier Llama. One key per provider; no key rotation |
| Backend | FastAPI (Python 3.11) |
| Frontend | React + Vite + Tailwind, ML/EN toggle |
| Evals | RAGAS + custom checks (hit@k, MRR, citation accuracy, abstention) |
| Run | Local: `python tasks.py up`, FastAPI on 8077 serving the built UI |
| Demo link | Cloudflare quick tunnel (`python tasks.py share`), `tools/cloudflared.exe` |
| Future | Cloud deploy (`deploy/`, `DEPLOY.md`): never built or run |

## Repo layout
```
sahayi/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, routes
│   │   ├── config.py          # pydantic-settings, env vars
│   │   ├── schemas.py         # request/response models
│   │   ├── profile.py         # extract structured user profile from free text
│   │   ├── retrieval/
│   │   │   ├── bm25.py
│   │   │   ├── dense.py
│   │   │   ├── fusion.py      # RRF
│   │   │   └── rerank.py
│   │   ├── llm/
│   │   │   ├── client.py      # Groq/Gemini with fallback + cost tracking
│   │   │   └── prompts.py     # cite-or-abstain prompts (ML + EN)
│   │   ├── answer.py          # orchestrates retrieve → rerank → generate → validate citations
│   │   └── logging.py         # per-query trace: tokens, latency, ₹ cost
│   └── tests/
├── ingest/
│   ├── sources.yaml           # scheme list: name, dept, url, lang
│   ├── download.py
│   ├── extract.py             # PyMuPDF + OCR fallback
│   ├── clean.py               # Malayalam Unicode normalisation
│   ├── chunk.py               # fixed | recursive | section-aware strategies
│   └── build_index.py         # BM25 + Qdrant
├── evals/
│   ├── gold/questions.jsonl   # ~150 Qs: ml / en / manglish + no-answer set
│   ├── run_eval.py
│   ├── metrics.py
│   ├── ablations.py           # chunking × retrieval × reranker grid
│   └── results/               # timestamped JSON + markdown tables
├── docs/demo_script.md
├── deploy/                    # future: cloud deploy files
├── tools/                     # gitignored: cloudflared
├── frontend/
│   └── src/ (components: ChatBox, SchemeCard, CitationChip, PdfViewer, LangToggle)
├── data/                      # gitignored: raw/, processed/, index/
├── Makefile
└── .env.example
```

## Commands
```bash
make setup          # pip install -r backend/requirements.txt && cd frontend && npm i
make ingest         # download → extract → clean → chunk → build_index
make serve          # uvicorn backend.app.main:app --reload --port 8077
make ui             # cd frontend && npm run dev
make test           # pytest backend/tests
make eval           # python evals/run_eval.py --config default
make ablate         # python evals/ablations.py  (writes comparison tables)
```
No `make` on Windows: use `python tasks.py <task>`. Also:
```bash
python tasks.py up     # build UI, start the app on 8077 with full answers, open the browser
python tasks.py down   # stop the app and any tunnel
python tasks.py share  # temporary public link until Ctrl+C
```

## Env vars (`.env.example`)
```
GROQ_API_KEY=
GEMINI_API_KEY=
LLM_PRIMARY=groq
LLM_MODEL_GROQ=openai/gpt-oss-120b
LLM_MODEL_GEMINI=gemini-3.6-flash
ANSWER_MODEL=qwen/qwen3.8-27b
ANSWER_MODEL_GEMINI=gemini-3.6-flash
API_PORT=8077
JINA_API_KEY=
EMBED_MODEL=BAAI/bge-m3
QDRANT_PATH=./data/index/qdrant
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
```
Check current model names on the Groq/Gemini docs before hardcoding — they change.

## Data model
Chunk metadata (required on every chunk):
```python
{
  "chunk_id": str,        # f"{doc_id}:{page}:{idx}"
  "doc_id": str,
  "scheme": str,
  "department": str,
  "source_url": str,
  "page": int,
  "lang": "ml" | "en",
  "downloaded_at": "YYYY-MM-DD",
  "text": str
}
```

API response:
```python
{
  "answer": str,
  "lang": "ml" | "en",
  "schemes": [{"name", "eligible": bool | None, "reasons": [str], "benefit", "how_to_apply", "citations": [chunk_id]}],
  "citations": [{"chunk_id", "source_url", "page", "snippet"}],
  "abstained": bool,
  "as_of": "YYYY-MM-DD",
  "meta": {"latency_ms", "tokens_in", "tokens_out", "cost_inr", "model"}
}
```

## Answer pipeline (`answer.py`)
1. Detect language; extract profile (age, district, occupation, income band, ration card, category).
2. Query rewrite: produce ML + EN variants for cross-lingual retrieval (English, Manglish and mixed-script questions; Malayalam-script questions skip it).
3. BM25 + dense (top 20 each) → RRF → Jina int8 rerank of the top 6 (overlapped with the rewrite call) → top 5.
4. Generate with prompt that requires `[chunk_id]` tags after each claim.
5. **Validate:** strip/flag any claim whose cited chunk_id wasn't in the context. If nothing valid remains → abstain.
6. Log trace.

Production only (`backend/app/production.py`; evals keep these off): generator fallback chain, scheme-card filter, 7-day answer cache without profile data, and a degrade mode that returns sources + cards with a "busy" notice when every generator fails.

Demo link only (`DEMO_MODE=1`, set by `tasks.py share`): 5 requests/min per visitor IP, a daily cap on generated answers (then busy mode, no LLM calls), no API docs, and `/ask` sent as `text/plain` with the same body, because quick tunnels do not support SSE.

The app reads its own index copy (`data/index/qdrant_serve`), since local Qdrant locks its folder and evals hold the main one. Gate 2 evals run in one process with one shared retriever (`evals/run_until_done.py`), logging memory to `data/logs/eval_rss.jsonl`.

## Eval targets
| Metric | Target |
|---|---|
| hit@5 | ≥ 0.85 (Gate 1) |
| MRR@10 | ≥ 0.70 |
| Faithfulness (RAGAS) | ≥ 0.90 (Gate 2) |
| Citation accuracy | ≥ 0.90 |
| Eligibility accuracy | ≥ 0.80 |
| Abstention on no-answer set | ≥ 0.90 |
| p95 latency | < 4 s |
| Cost/query | logged in ₹ |

CI runs `make eval` on push and fails if hit@5 or faithfulness drop below target.

## Ablations to report
- Chunking: fixed 512 / recursive 800+100 overlap / section-aware
- Retrieval: BM25 / dense / hybrid RRF
- Reranker: none / bge-reranker-v2-m3
- Embeddings: bge-m3 vs multilingual-e5
- LLM: Groq vs Gemini (faithfulness, latency, ₹)

Write results as markdown tables to `evals/results/` and copy the latest into README.

## Conventions
- Python: type hints, `ruff` + `black`, pydantic models at boundaries.
- Keep functions small and pure in `retrieval/` so ablations can swap them via config.
- All prompts live in `llm/prompts.py`, with ML and EN versions side by side.
- Malayalam text: NFC-normalise; strip zero-width joiners only where they break tokenisation.
- Never commit `data/` or API keys.
- Add a test for every bug in retrieval or citation validation.

## Status
Complete as a local app with an optional demo link. Gate 2's final numbers: `evals/results/gate2_final.md`.

## Build order (10 days)
1–2 Corpus + ingestion · 3–4 Retrieval + gold set (**Gate 1**) · 5–6 Answering API · 7–8 Evals + ablations (**Gate 2**) · 9–10 UI, deploy, real-user demo.
