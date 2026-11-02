# Gate 2 report

**INTERIM, does not count.** Progress: qwen/qwen3.8-27b 53/186, openai/gpt-oss-20b 81/186

**Production primary: none** (not picked: the runs have not finished).

## Side by side

| metric | qwen/qwen3.8-27b | openai/gpt-oss-20b |
|---|---|---|
| items run | 53 | 81 |
| answerable gold n (needs ≥ 120) | 53 | 81 |
| faithfulness (≥ 0.90) | 0.965 | 0.883 |
| citation accuracy (≥ 0.90) | 0.97 | 0.854 |
| Gate 2 | does not count yet | does not count yet |
| eligibility accuracy (≥ 0.80) | None (n=0) | None (n=0) |
| abstention recall (≥ 0.90) | – | – |
| abstention precision | 0.0 | 0.0 |
| false abstain rate | 0.151 | 0.247 |
| ₹ / query, mean | 0.532 | 0.03 |
| provider failures excluded | 0 | 0 |

## Truncation

| generator | compact retries | failed after retry | failure rate |
|---|---|---|---|
| qwen/qwen3.8-27b | 4 | 1 | 0.019 |
| openai/gpt-oss-20b | 0 | 0 | 0.0 |

## By slice and language

| slice | qwen3.8-27b n | qwen3.8-27b faith. | qwen3.8-27b cite | gpt-oss-20b n | gpt-oss-20b faith. | gpt-oss-20b cite |
|---|---|---|---|---|---|---|
| main | 33 | 0.966 | 0.978 | 61 | 0.876 | 0.848 |
| heldout | 20 | 0.964 | 0.957 | 20 | 0.907 | 0.875 |
| eligibility | 0 | – | – | 0 | – | – |
| no_answer | 0 | – | – | 0 | – | – |
| en | 15 | 0.96 | 0.971 | 25 | 0.948 | 0.775 |
| ml | 16 | 0.945 | 0.943 | 25 | 0.875 | 0.882 |
| manglish | 22 | 0.983 | 0.99 | 31 | 0.838 | 0.885 |

## Live latency (ms, p50 / p95)

| generator | n | first source (TTFC) | first answer token (TTFT) | full answer |
|---|---|---|---|---|
| qwen/qwen3.8-27b | pending | – | – | – |
| openai/gpt-oss-20b | pending | – | – | – |

Free-tier throttling is included.

## Judge agreement

| generator | judges | answers | statements | supported | cited ok |
|---|---|---|---|---|---|
| qwen/qwen3.8-27b | pending | – | – | – | – |
| openai/gpt-oss-20b | pending | – | – | – | – |

The main judge, gpt-oss-120b, comes from the same model family as gpt-oss-20b, so its scores for that generator may be less independent. The Gemini check is the guard, on a small sample.
