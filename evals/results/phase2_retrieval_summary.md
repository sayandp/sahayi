# Phase 2 — retrieval results (2026-10-12)

Corpus: 735 section-aware chunks, 59 schemes. Gold: 126 answerable questions
(53 EN / 46 ML / 27 Manglish) + 25 no-answer, `evals/gold/questions.jsonl`; gold =
(doc_id, page). Embeddings `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3`, both on
CPU (8 threads). Targets: **Gate 1 hit@5 ≥ 0.85**, MRR@10 ≥ 0.70, p95 < 4 s.

| config | retrieval | rerank | hit@1 | hit@5 | hit@10 | MRR@10 | hit@5 EN | hit@5 ML | hit@5 Manglish | p95 latency (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | bm25 | no | 0.571 | **0.683** | 0.698 | 0.624 | 0.792 | 0.609 | 0.593 | ~0.0 s |
| bm25_nostem | bm25 | no | 0.563 | **0.683** | 0.714 | 0.622 | 0.792 | 0.609 | 0.593 | ~0.0 s |
| dense | dense | no | 0.714 | **0.825** | 0.881 | 0.766 | 0.868 | 0.978 | 0.481 | ~0.2 s |
| hybrid | hybrid | no | 0.587 | **0.794** | 0.857 | 0.675 | 0.811 | 0.87 | 0.63 | ~0.2 s |
| hybrid_w05 | hybrid | no | 0.603 | **0.802** | 0.873 | 0.695 | 0.811 | 0.913 | 0.593 | ~0.2 s |
| hybrid_w025 | hybrid | no | 0.619 | **0.817** | 0.873 | 0.707 | 0.849 | 0.913 | 0.593 | ~0.2 s |
| default_pool5 | hybrid | top-5 | 0.675 | **0.794** | 0.857 | 0.732 | 0.811 | 0.87 | 0.63 | ~5.5 s |
| default_pool10 | hybrid | top-10 | 0.698 | **0.841** | 0.857 | 0.757 | 0.849 | 0.957 | 0.63 | ~10.7 s |
| default | hybrid | top-20 | 0.73 | **0.865** | 0.889 | 0.791 | 0.887 | 0.957 | 0.667 | ~21.2 s |
| dense_rerank_pool10 | dense | top-10 | 0.706 | **0.849** | 0.881 | 0.771 | 0.887 | 0.935 | 0.63 | ~10.7 s |
| dense_rerank | dense | top-20 | 0.73 | **0.873** | 0.889 | 0.79 | 0.906 | 0.935 | 0.704 | ~21.2 s |

Latency: without reranking, the p95 is measured. With reranking, it is estimated as
0.2 s + ~1.05 s per reranked pair. That rate was measured uncached on this CPU
(20 pairs = 20.2 s), and the measured hybrid + rerank-20 run gave p95 25.2 s. Cached
runs reuse scores, so their measured latency is not meaningful.

## Findings

- **Gate 1 passes only with a 20-candidate cross-encoder rerank:** `default`
  (hybrid + rerank) at 0.865 and `dense_rerank` at 0.873, with MRR@10 about 0.79.
  Both cost about 20 s per query on this CPU, 5x over the 4 s p95 target.
- **Dense bge-m3 alone gives 0.825 hit@5 at 0.2 s.** It carries cross-lingual
  retrieval: Malayalam 0.98, English 0.87.
- **BM25 is weak** (0.68) because of the cross-script gap. Malayalam suffix stemming
  makes no difference. Fusing BM25 with dense lowers overall hit@5 (0.79–0.82) and
  only helps Manglish.
- **Manglish (romanised Malayalam) is the main gap:** 0.48–0.70 in every config.
  This is the job of pipeline step 2, LLM query rewrite into Malayalam and English
  variants (Phase 3). No LLM key is configured yet.
- **The reranker's top score separates no-answer questions:** median 0.97 on
  answerable versus 0.01 on no-answer. That's a strong abstention signal for Phase 3.

## Caveats

- The gold set was written by the same agent that built the retriever, from the
  ingested text. Have a person review it before treating Gate 1 as passed.
- The BM25 weights (0.5 / 0.25) and rerank pool sizes were fixed before running,
  not tuned. The configs are still compared on the same 126 questions, with no
  held-out split.
