"""Retrieval components (BM25, dense, fusion, rerank) and the composed Retriever.

The torch query encoder and the onnxruntime reranker run back to back on the same CPU
cores. Busy-waiting idle threads from either library slow the other one down, so both
default to passive waiting (measured 2026-10-20 on 25 gold queries: retrieval p50
1878 -> 1387 ms, p95 2078 -> 1531 ms; scores are unchanged). OMP_WAIT_POLICY must be
set before torch is imported, hence here.
"""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
