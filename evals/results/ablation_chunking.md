# Ablation: chunking x retrieval x rerank (146 answerable gold questions)

No rewrite and no contextual headers, so these sit below the default's 0.897. Latency was measured while other work was running, so only hit rates compare fairly.

| chunking | retrieval | rerank | hit@1 | hit@5 | hit@10 | MRR@10 | EN | ML | Manglish | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | none | 0.278 | **0.524** | 0.603 | 0.378 | 0.623 | 0.391 | 0.556 | 2 |
| fixed | dense | none | 0.54 | **0.778** | 0.865 | 0.641 | 0.755 | 0.935 | 0.556 | 245 |
| fixed | hybrid | none | 0.405 | **0.73** | 0.833 | 0.55 | 0.736 | 0.826 | 0.556 | 198 |
| fixed | hybrid | Jina int8 top-6 | 0.563 | **0.762** | 0.833 | 0.653 | 0.755 | 0.87 | 0.593 | 1047 |
| recursive | bm25 | none | 0.286 | **0.5** | 0.587 | 0.385 | 0.566 | 0.391 | 0.556 | 2 |
| recursive | dense | none | 0.516 | **0.81** | 0.889 | 0.638 | 0.849 | 0.913 | 0.556 | 163 |
| recursive | hybrid | none | 0.437 | **0.73** | 0.833 | 0.566 | 0.755 | 0.783 | 0.593 | 178 |
| recursive | hybrid | Jina int8 top-6 | 0.563 | **0.738** | 0.833 | 0.65 | 0.755 | 0.826 | 0.556 | 988 |
| section_aware | bm25 | none | 0.198 | **0.492** | 0.563 | 0.336 | 0.547 | 0.391 | 0.556 | 3 |
| section_aware | dense | none | 0.595 | **0.778** | 0.849 | 0.672 | 0.811 | 0.913 | 0.481 | 189 |
| section_aware | hybrid | none | 0.429 | **0.746** | 0.825 | 0.562 | 0.755 | 0.804 | 0.63 | 222 |
| section_aware | hybrid | Jina int8 top-6 | 0.571 | **0.762** | 0.825 | 0.66 | 0.774 | 0.848 | 0.593 | 1024 |
