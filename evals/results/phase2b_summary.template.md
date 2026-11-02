# Phase 2b: fast retrieval that passes Gate 1 (2026-10-20)

**Result.** The new `default` config passes Gate 1 at **hit@5 0.881** (MRR@10 0.785).
In a live run with the real Groq rewrite and no caches it measured **p95 1.95 s** on
this CPU (Ryzen 7 5800H, 8 cores), under the 2 s target. Peak RAM is about 3.1 GB.

- **Gold set:** 146 answerable questions (126 main + 20 held-out) and 25 no-answer.
  56 main questions were rephrased to remove lexical leakage, so these numbers are not
  comparable with the Phase 2 table.
- **Corpus:** 735 section-aware chunks.
- **Settings:** all model and browser caches are under `D:\rag\data\cache\`. The
  rewrite model is `qwen/qwen3.8-27b` on Groq. Groq no longer serves Llama chat
  models; on 2026-10-20 it listed gpt-oss-20b/120b, qwen3.8-27b and allam.

## 1. Reranker latency diagnosis (bge-reranker-v2-m3, one query, 20 candidates)

`evals/profile_rerank.py`. Settings: fp32, `torch.inference_mode`, torch threads 8
(8 physical / 16 logical cores). Pair length: median 356 subword tokens; 5 of 20
pairs exceed 512.

| setting | time |
|---|---|
| tokenisation (20 pairs) | 7 ms |
| forward, max_length 512, one batch of 20 | 27.5 s |
| forward, max_length 512, batches of 8 / one by one | 26.1 s / 27.5 s |
| forward, max_length 384, one batch / batches of 8 / one by one | 20.0 s / 19.3 s / 18.9 s |
| max_length 384, 16 threads | 28.6 s (slower than 8) |

**Diagnosis:** the forward pass is compute-bound, at about 1 s per pair for a
568M-parameter cross-encoder in fp32.

- Batching makes no difference, and tokenisation is negligible.
- max_length 384 saves 27%. Using all 16 logical threads hurts.
- The quick wins are applied: max_length 384 (256 for the default), physical-core
  threads, `inference_mode`, and one batched call per stage.
- The real levers are fewer FLOPs per pair (a smaller model) and fewer pairs (a
  smaller pool, or reranking only the new candidates).

## 2. Reranker options (`evals/bench_rerankers.py`, 6–10 candidates, cold)

| backend | s/pair | load | peak RAM | notes |
|---|---|---|---|---|
| bge-reranker-v2-m3 fp32 (sentence-transformers) | 0.96–1.1 | 42–49 s | 4.0–4.9 GB | reference |
| bge-reranker-v2-m3 int8 dynamic (our ONNX export) | 0.95 | 26 s | 4.9 GB | Spearman 0.96 vs fp32. Little gain in the micro-benchmark (no VNNI on Zen 3); in the full eval it was 23% faster (p95 8.6 s vs 11.2 s) at equal hit@5 |
| jina-reranker-v2-base-multilingual fp32 ONNX | 0.29 | 91 s | 5.6 GB | Jina's PyTorch remote code fails on transformers 5.x; the ONNX file works |
| jina-reranker-v2 int8 ONNX @384 | 0.23–0.28 | 5–15 s | 3.0–4.1 GB | about 3.5–4× faster than bge |
| jina-reranker-v2 int8 ONNX @256 | 0.14–0.15 | 3 s | 3.8–4.0 GB | used by the default |
| hosted API (Cohere `rerank-v3.5` / Jina) | – | – | – | implemented behind the same interface (`rerank_backend: api`). Disabled: no `COHERE_API_KEY` / `JINA_API_KEY`, so it falls back to no rerank |

**Selective rerank** skips reranking when the best dense list's top-1 minus top-2
cosine is at least 0.05 (fixed before running). It skipped **12–18%** of queries at
identical hit@5. It cuts the median but not the p95, because p95 is set by the
queries that do get reranked, so the default doesn't use it.

**Overlap.** The rewrite call (about 1 s of network time) runs in a thread while the
original query is retrieved and its top-6 is reranked speculatively. After the
rewrite returns, only candidates not yet scored are reranked. In live runs the wait
on the rewrite was 0–150 ms, so it is fully hidden behind local work.

## 3. Query rewrite (`backend/app/query.py`, prompt in `llm/prompts.py`)

- **What it does:** one Groq call returns `{lang, ml_query, en_query, profile_hints}`.
  Retrieval runs the original question plus the Malayalam and English variants and
  fuses them with RRF.
- **Fallback chain:** Groq → Gemini (if `GEMINI_API_KEY` is set) → no rewrite.
- **Cache:** keyed by a hash of the query. It stores `{lang, ml_query, en_query}` only,
  never the profile hints. It is used for evals; the API should keep it off or
  in-memory (no PII stored).
- **Live latency** (`evals/bench_rewrite.py`, paced, uncached): qwen3.8-27b **p50
  0.97 s, p95 1.16 s**; gpt-oss-20b p50 1.37 s, p95 1.74 s.
- **Throttling:** the free Groq tier rate-limits bulk eval runs to about 12 calls a
  minute. Those waits are recorded separately (`throttled_ms`) and excluded from
  latency.
- **Effect:** rewrite lifts Manglish hit@5 from 0.48 to 0.70–0.74 on its own, and to
  0.85–0.96 together with contextual headers.

## 4. Gold-set integrity

- **Leakage** (`evals/leakage.py`, `gold_leakage.md`): 56 of 126 answerable questions
  were ranked first by BM25 with over 40% word overlap with their gold chunk.
  - All 56 were rephrased in citizen style, with a second pass on 11 of them
    (`evals/gold/rephrase_2b.py`; the original text is kept as `question_orig`).
  - Mean overlap fell from 0.45 to 0.26, and leaky questions from 56 to 2. The two
    remaining ones overlap only through everyday words ("husband died", ചികിത്സ).
- **Held-out slice:** 20 questions by `openai/gpt-oss-120b`, a different model family
  from the original drafter (Claude) and from the rewriter (qwen); no Gemini key is
  set. The mix is 13 Manglish, 4 Malayalam questions about English documents and 3
  English questions about Malayalam documents.
  - Each comes with a verbatim evidence quote, checked programmatically.
  - 27 were generated. The LLM judge rejected 7, and those are kept in the YAML as
    `status: rejected` with the reason.
- **verify-gold** (`evals/verify_gold.py --sample N | --split | --ids`): an LLM judge
  (gpt-oss-120b) checks that each gold page answers its question. Result: held-out
  20/20 valid after rejections, and a random 20 of the main set 19/19 "yes" (one was
  rate-limited).

## 5. Abstain threshold (`evals/abstain.py`)

The positive class is no_answer (abstain). The threshold is chosen to maximise F1 on
the top-1 rerank score.

| config | score | threshold | F1 | abstain precision | abstain recall | answerable kept |
|---|---|---|---|---|---|---|
| **default** (Jina top-6 @256) | Jina logit | **-2.0084** (saved in `evals/configs.yaml`) | 0.53 | 0.60 | 0.48 | 94.5% |
| Jina top-10 @384 | Jina logit | -1.27 | 0.67 | 0.59 | 0.76 | 91.1% |
| dense, no rerank | cosine | 0.577 | 0.47 | 0.32 | 0.88 | 67.8% |

- **Jina's scores separate no-answer questions only moderately.** bge's did so well
  in Phase 2 (median 0.97 answerable vs 0.01 no-answer), but bge is too slow to run
  live.
- **So the threshold is a soft pre-filter, tuned to keep answer recall.** In Phase 3
  the cite-or-abstain prompt plus citation validation must do the real abstaining.

## 6. Ablation (gold: 146 answerable + 25 no-answer)

{{ABLATION}}

All rows use section-aware chunks, bge-m3 and BM25 (hybrid = RRF k=60). Rewrite means
qwen3.8-27b and three query variants; context means contextual chunk headers. Latency
from cached eval runs is not meaningful for rerank configs; the live runs are below.

### Live end-to-end retrieval latency (`evals/bench_e2e.py`: real rewrite, no caches)

{{E2E}}

### Latency breakdown (ms)

{{STAGES}}

### 10 worst misses (default; gold page not in the top 10)

{{WORST}}

Four of these retrieve the right scheme but a different page (q112, q116, q102,
h004). Page-level gold is stricter than what the answer step will need.

## Changes that moved the numbers

1. **Contextual chunk headers** (`index_context`): index and rerank
   "scheme — department [section]" plus the chunk text; the displayed and cited text
   is unchanged. hit@5 rose from 0.825 to 0.849 without rerank, and hit@10 from 0.89
   to 0.94. This fixed most Kerala-vs-central pension confusions.
2. **Query rewrite:** see section 3 for the Manglish gains.
3. **Jina int8 rerank on top-6 at 256 tokens:** hit@5 rose from 0.849 to 0.881. Top-10
   at 384 tokens gives 0.929 but costs about 2.5 s.
4. **Overlap of rewrite and speculative rerank:** live p95 dropped from about 2.4 s
   (sequential estimate) to 1.95 s.

## Deployment note: HF Spaces CPU basic

These estimates assume 2 vCPUs, which is what HF Spaces CPU basic provides.

- **Simulated latency** (2 threads on this laptop): p50 2.29 s, **p95 2.86 s**, over
  the 2 s target. Spaces vCPUs are usually slower than these cores, so treat this as
  a lower bound.
- **Options for Spaces:**
  - **Rerank fewer pairs:** top-4 (hit@5 then needs re-measuring), or selective
    rerank with a larger margin.
  - **Hosted rerank API:** Cohere or Jina, behind the existing config flag.
  - **Larger instance:** upgrade the Space's CPU.
- **Peak RAM** is about 3.1 GB in the live runs (bge-m3 plus Jina int8 plus the
  index), well within Spaces' 16 GB.
