# Phase 4b: deploy prep, 2-core retrieval simulation (2026-10-31)

**Not a container measurement.** Docker is not installed on this machine. These runs use the
exact Space folder (`data/deploy_space`) in a clean Windows venv with no torch and no keys for
Groq or Gemini, pinned to 2 logical cores on separate physical cores (Ryzen 7 5800H). Hugging Face
hardware is usually slower. Rewrites are cached, so they add no waiting; the rerank cache is off.

Gold set: 171 questions (146 answerable, 126 of them main).

| config | hit@5 main | hit@5 all | EN / ML / Manglish | p50 ms | p95 ms | peak RAM GB | fallbacks |
|---|---|---|---|---|---|---|---|
| **deploy_local** (run 1) | **0.897** | **0.884** | 0.93 / 0.863 / 0.842 | 1428 | **2110** | 1.9 | none |
| deploy_hosted | 0.889 | 0.877 | 0.912 / 0.863 / 0.842 | 1826 | 2944 | 1.91 | 54 HTTP 429, 2 timeouts |
| **deploy_local** (run 2) | **0.897** | **0.884** | 0.93 / 0.863 / 0.842 | 1411 | **2068** | 1.9 | none |

Decision: `deploy_local` stays the deploy default, because hosted did not win on p95.

- **Hosted is limited by the key:** a third of its queries hit HTTP 429 and fell back to local.
  A paid Jina tier might do better; that was not tested.
- **Query encoder:** int8 ONNX gives the same overall hit@5 as the eval's fp32 torch encoder
  (0.897 / 0.884).
- **By language, deploy_local run 2 (p50 / p95 ms):** EN 1552 / 2036, ML 1124 / 1151 (no rewrite),
  Manglish 1606 / 2271.

Cold start, 2 cores (OS file cache warm, so a Space's first start will be slower):
- Process start to `/health` responding: 9.9 s and 12.1 s.
- Loading models in the benchmark script: 8.4–16.7 s.

Assets (`deploy/build_bundle.py`): 894 MB in total, of which the bge-m3 query encoder is 586 MB, the
Jina reranker 297 MB, and the index and chunks 11 MB.

Image size has **not been measured**. Estimate: about 1.4 GB uncompressed (python:3.11-slim about
130 MB, dependencies about 350 MB on Windows, assets about 890 MB).

Playwright against the real server in retrieval-only mode, 360 px wide: 2 of 2 pass
(`frontend/e2e/deployed.spec.ts`). Mock-mode suite: 7 of 7 pass.
