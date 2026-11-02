# Phase 3: answering + API (2026-10-20)

## Result

| goal | target | result |
|---|---|---|
| Faithfulness (Gate 2) | ≥ 0.90 | **0.914** (qwen3.8-27b, 19 judged answers) |
| Citation accuracy (Gate 2) | ≥ 0.90 | **0.926** |
| TTFT p95 | < 2.5 s | **not met**: 3.44 s measured live, before the thread fix. The fix took 0.55 s off retrieval p95, so about 2.9 s is expected; that has not been re-measured end to end |

- **Gate 2:** passes for the chosen generator, `qwen/qwen3.8-27b`, on a small sample: 25
  live items, of which 19 were answered and judged, plus 6 no-answer items. The judge is
  `openai/gpt-oss-120b`.
- **Unfinished items:** qwen's last 12 eligibility items hit Groq's free-tier cap of 200K
  tokens/day before they ran. To finish them once the quota frees up, run
  `python evals/answer_eval.py --model qwen/qwen3.8-27b --n 16 --no-answer 6 --live --patient --resume evals/results/2026-10-20_1445_answers_bench_qwen_full.json`.
- **Eligibility:** the 15-case set was run in full with gpt-oss-20b on the final
  pipeline. It scored **0.933 eligibility accuracy**, with faithfulness 0.915 and
  citation accuracy 0.904.
- **Retrieval:** unchanged and still passing Gate 1 (hit@5 0.881, MRR@10 0.785).
  Scheme-level hit@5 is **0.929**, and same scheme + same section hit@5 is **0.889**.

Caveats:
- **Small samples:** 19–22 judged answers per generator, so ±0.05 is within noise.
- **Judge family:** the judge is in the same model family as gpt-oss-20b (Groq offers no
  third family, and there is no Gemini key). It is a different family from qwen.
- **Strict judge:** it counts restated user facts ("your mother is BPL") as unsupported,
  because only excerpts count as support, so faithfulness is conservative.
- **Not RAGAS proper:** RAGAS is not installed; the metrics are re-implemented with one
  judge call per answer (`evals/answer_eval.py` docstring).

## 0. Hosted Jina reranker

- **Implementation:** `rerank_backend: api` with `rerank_fallback_onnx_file`, and the new
  `deploy` config. A call that errors, has no key, or takes more than 1.5 s is re-scored
  with local Jina int8. Jina's sigmoid scores are mapped back to logits so that
  `abstain_threshold` still applies.
- **Not benchmarked:** **`JINA_API_KEY` is not in `.env`**. I checked `.env`,
  `.env.example` and the user environment.
- **What was verified:** the `deploy` run fell back on 171/171 queries ("RerankUnavailable
  after 0 ms") and gave the same hit@5 (0.881). Latency there is local-rerank latency.
- **Pending:** a hosted vs local hit@5 and p50/p95 comparison needs the key. When it is
  added, run `python tasks.py eval default deploy`; the summary records
  `rerank_fallback_rate`.

| config | hit@5 | scheme hit@5 | scheme + section hit@5 | MRR@10 |
|---|---|---|---|---|
| default (local Jina int8) | 0.881 | 0.929 | 0.889 | 0.785 |
| deploy (hosted, fell back every time: no key) | 0.881 | 0.929 | 0.889 | 0.785 |

## 1. Generator: two candidates, both live on Groq

| | gpt-oss-20b (prompt v3) | **qwen3.8-27b** (prompt v4) |
|---|---|---|
| faithfulness / citation accuracy | 0.921 / 0.889 (fails) | **0.914 / 0.926** |
| first generated token, p50 / p95 | 0.94 / 1.18 s (reasoning cannot be turned off) | **0.70 / 0.88 s** (`reasoning_effort: none`) |
| TTFT p50 / p95 (live) | 2.97 / 3.62 s | **2.70 / 3.44 s** (22 items with no quota waits) |
| ₹ / query (including the rewrite) | 0.11 | 0.48 |
| language match / false-abstain rate | 0.955 / 0.29 | 1.0 / 0.0 |

- **Choice:** qwen is the default (`ANSWER_MODEL`). It is the faster of the two and the
  one that passes Gate 2.
- **Why 20b's eligibility was poor:** 20b abstained wrongly because it left eligibility
  verdict sentences untagged.
- **Fixes:** prompt v4 adds a tagged-verdict example, and the answer is now rebuilt from
  validated scheme cards when no sentence survives. With these, 20b scores 0.933
  eligibility accuracy and 0 abstentions on the 15 cases (last table row below).
- **Free-tier cost of qwen:** it also does the rewrite, so one model's 200K tokens/day
  covers about 45 answers. A paid tier is needed for real traffic.

## 2. Gate 2 and the other answer metrics by slice

Run files:
- qwen: `2026-10-20_1445_answers_bench_qwen_full.json`
- gpt-oss-20b bench: `2026-10-20_1311_answers_bench_gptoss20b.json`
- gpt-oss-20b eligibility after the fixes: `2026-10-20_1531_answers_elig_gptoss20b_v4.json`

How to read the tables:
- **Faithfulness:** the share of user-visible statements (answer sentences and card
  fields) that the judge finds supported by the excerpts.
- **Citation accuracy:** the statement's *own* citations support it, and its numbers
  match.
- **Relevancy:** the judge's 1–5 rating divided by 5.
- **Context hit@5:** whether the gold page was among the excerpts.

| generator | slice | n | answered | faithfulness | citation acc. | relevancy | context hit@5 | language match |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-27b | all | 25 | 19 | **0.914** | **0.926** | 0.958 | 0.947 | 1.0 |
| qwen/qwen3.8-27b | lang:en | 12 | 9 | 0.875 | 0.896 | 0.933 | 1.0 | 1.0 |
| qwen/qwen3.8-27b | lang:ml | 8 | 5 | 0.936 | 0.925 | 1.0 | 1.0 | 1.0 |
| qwen/qwen3.8-27b | lang:manglish | 5 | 5 | 0.964 | 0.976 | 0.96 | 0.8 | 1.0 |
| qwen/qwen3.8-27b | slice:main | 7 | 7 | 0.892 | 0.944 | 0.943 | 0.857 | 1.0 |
| qwen/qwen3.8-27b | slice:heldout | 9 | 9 | 0.915 | 0.918 | 0.956 | 1.0 | 1.0 |
| qwen/qwen3.8-27b | slice:eligibility | 3 | 3 | 0.963 | 0.905 | 1.0 | 1.0 | 1.0 |
| qwen/qwen3.8-27b | slice:no_answer | 6 | 0 | - | - | - | - | - |
| openai/gpt-oss-20b | all | 37 | 22 | **0.921** | **0.889** | 0.955 | 0.935 | 0.955 |
| openai/gpt-oss-20b | lang:en | 20 | 11 | 0.929 | 0.915 | 0.964 | 1.0 | 1.0 |
| openai/gpt-oss-20b | lang:ml | 10 | 6 | 0.881 | 0.8 | 1.0 | 0.857 | 1.0 |
| openai/gpt-oss-20b | lang:manglish | 7 | 5 | 0.95 | 0.957 | 0.88 | 0.857 | 0.8 |
| openai/gpt-oss-20b | slice:main | 7 | 7 | 0.924 | 0.917 | 0.943 | 0.857 | 1.0 |
| openai/gpt-oss-20b | slice:heldout | 9 | 7 | 0.923 | 0.903 | 0.914 | 1.0 | 0.857 |
| openai/gpt-oss-20b | slice:eligibility | 15 | 8 | 0.915 | 0.86 | 1.0 | 0.933 | 1.0 |
| openai/gpt-oss-20b | slice:no_answer | 6 | 0 | - | - | - | - | - |
| openai/gpt-oss-20b | all | 15 | 15 | **0.915** | **0.904** | 1.0 | 0.933 | 0.933 |
| openai/gpt-oss-20b | lang:en | 11 | 11 | 0.957 | 0.943 | 1.0 | 1.0 | 1.0 |
| openai/gpt-oss-20b | lang:ml | 2 | 2 | 0.974 | 0.913 | 1.0 | 0.5 | 1.0 |
| openai/gpt-oss-20b | lang:manglish | 2 | 2 | 0.625 | 0.571 | 1.0 | 1.0 | 0.5 |
| openai/gpt-oss-20b | slice:eligibility | 15 | 15 | 0.915 | 0.904 | 1.0 | 0.933 | 0.933 |

| generator | eligibility acc. (n) | card found | abstain recall (no-answer n) | abstain precision | false-abstain rate | TTFT p50 / p95 ms | full p50 / p95 ms | ₹ / query mean (p95) | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-27b | 1.0 (3) | 1.0 | 1.0 (6) | 1.0 | 0.0 | 2795 / 17040 (n=25) | 3445 / 17371 | 0.476 (0.6479) | 3694.96 / 493.4 |
| openai/gpt-oss-20b | 0.467 (15) | 0.533 | 1.0 (6) | 0.4 | 0.29 | 2973 / 3619 (n=37) | 2993 / 3620 | 0.106 (0.1371) | 2837.568 / 374.973 |
| openai/gpt-oss-20b | 0.933 (15) | 0.933 | - (0) | - | 0.0 | not live | - | 0.024 (0.0303) | 2258.867 / 264.667 |

The last 20b row was run with cached rewrites and rerank scores, so its latency is not
live and its ₹ excludes the rewrite.

## 3. Latency

### Measured live

The qwen run's p50 / p95 by stage, over 22 items with no free-tier quota waits:

| stage | p50 | p95 |
|---|---|---|
| retrieval (rewrite overlapped with local work) | 1.85 s | 2.32 s |
| ↳ rewrite call (qwen, overlapped) | 0.87 s | 1.00 s |
| ↳ query embedding (bge-m3, 2 calls) | 0.68 s | 1.02 s |
| ↳ speculative rerank (6 pairs) / rerank after the rewrite | 0.76 / 0.37 s | 0.81 / 0.63 s |
| generator first token | 0.70 s | 0.88 s |
| first validated sentence after that | ~0.1 s | ~0.37 s |
| **TTFT** | **2.70 s** | **3.44 s** |
| full answer | 3.41 s | 4.04 s |

### Fix found: thread spin contention

- **Cause:** the onnxruntime reranker and the torch query encoder run back to back on the
  same 8 cores, and each library's idle threads busy-wait.
- **Fix:** passive waiting is now the default (`session.intra_op.allow_spinning=0` and
  `OMP_WAIT_POLICY=PASSIVE`). It changes no scores, and Gate 1 was re-run unchanged.
- **Effect on 25 gold queries** (rewrite cached):

| | p50 | p95 |
|---|---|---|
| retrieval before | 1878 ms | 2078 ms |
| retrieval after | **1387 ms** | **1531 ms** |
| query embedding before | 734 ms | 925 ms |
| query embedding after | 391 ms | 451 ms |

Running the live qwen bench again after the fix was blocked: qwen's daily token window is
full, and the rewrite needs it too.

### Why TTFT p95 is still about 2.9 s

After the fix, the local work (embedding plus the 6-pair speculative rerank, ~0.9 s)
takes about as long as the rewrite call. That leaves this floor:

| step | approx. |
|---|---|
| rewrite p95 | 1.0–1.2 s |
| post-rewrite embedding + rerank | 0.4 s |
| qwen first token | 0.9 s |
| first sentence | 0.1–0.4 s |
| **total** | **about 2.6–2.9 s** |

So a hosted reranker alone no longer helps much on 8 cores, though it still matters on a
2-vCPU Space. Options to get under 2.5 s:
- **Paid Groq tier:** its queue times are lower; gpt-oss showed 0.2–0.4 s `queue_time`
  on the free tier.
- **Shorter rewrite deadline for non-Manglish questions:** retrieval then goes ahead
  without the variants. This needs a Gate 1 ablation first.
- **Shorter prompt:** fewer, shorter excerpts.

### Measured, not adopted

- **Speculative generation:** start generating on the pre-rewrite top-5 and keep the
  result only if the final top-5 matches (`Answerer(speculative=True)`,
  `evals/spec_match.py`). The final top-5 equals the speculative one for only **7.6%** of
  the gold questions, so it is off.
- **LLM entailment check** (`--entail`, `evals/entail_check.py`, gpt-oss-20b, on 117
  judged statements):
  - It caught 8 of 11 unsupported statements and wrongly flagged 9 of 106 good ones.
  - Micro-faithfulness went from 0.906 to 0.970.
  - It costs **+1.1 s p50 / +2.0 s p95** per answer, after streaming.
  - It is off by default. It suits `/ask/sync` if the latency is acceptable.

## 4. Cost

Prices were checked on 2026-10-20 at console.groq.com/docs/models, at ₹96.5 per USD:

| model | input, per 1M tokens | output, per 1M tokens |
|---|---|---|
| qwen3.8-27b | $0.80 | $4.00 |
| gpt-oss-20b | $0.075 | $0.30 |
| gpt-oss-120b | $0.15 | $0.60 |

Per query, including the rewrite call:
- **qwen:** **₹0.48 mean** (p95 ₹0.65), from about 3.7K input and 0.5K output tokens.
- **gpt-oss-20b:** ₹0.11.

## 5. Five example answers (qwen, live; citations shortened to S-tags)

**1. Malayalam, heldout (h013)**: judge 9/9 supported, TTFT 2887 ms

> **Q:** എന്റെ കുടുംബത്തിൽ NREGA ജോബ് കാർഡ് അപേക്ഷിക്കാൻ എന്തെല്ലാം രേഖകൾ വേണം?
>
> **A:** NREGA ജോബ് കാർഡ് അപേക്ഷിക്കാൻ അപേക്ഷകന്റെ ഫോട്ടോ, കുടുംബത്തിലെ എല്ലാ അപേക്ഷകരുടെയും പേര്, പ്രായം, ലിംഗം, ഗ്രാമത്തിന്റെയും ഗ്രാമ പഞ്ചായത്തിന്റെയും ബ്ലോക്കിന്റെയും പേര്, തിരിച്ചറിയൽ രേഖ (റേഷൻ കാർഡ്, വോട്ടർ ഐഡി, ആധാർ, PAN), SC/ST/IAY/LR ബെനീഫിഷ്യറി ആണോ എന്ന് വിവരങ്ങൾ, സാമ്പിൾ സിഗ്നേച്ചർ/തുമ്പ് ഇംപ്രഷൻ എന്നിവ ആവശ്യമാണ്. [S1]
>
> *Card:* Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA) (needs info)

- [S1] `mgnrega--mgnrega-608fc2:1:5`: Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA), page 1, https://www.myscheme.gov.in/schemes/mgnrega

**2. Malayalam, main (q086)**: judge 4/4 supported, TTFT 2882 ms

> **Q:** നെൽകർഷകർക്ക് പെൻഷൻ കൊടുക്കാൻ തുടങ്ങിയത് എന്ന് മുതലാണ്?
>
> **A:** നെൽകർഷകർക്ക് പെൻഷൻ നൽകുന്ന പദ്ധതിക്ക് 2008-09 വർഷം 31-01-2009-ലെ സർക്കാർ ഉത്തരവ് പ്രകാരം സർക്കാർ അനുമതി നൽകി. [S1] ഈ പദ്ധതിയുടെ പേര് 'കിസാൻ അഭിമാൻ' എന്നാണ്. [S1]
>
> *Card:* Kerala Farmers' Welfare Fund Board pension (Karshaka pension) (needs info)

- [S1] `kerala_farmers_welfare_fund--karshakapension-e99e34:1:0`: Kerala Farmers' Welfare Fund Board pension (Karshaka pension), page 1, https://keralaagriculture.gov.in/en/karshakapension/

**3. English, heldout (h020)**: judge 6/6 supported, TTFT 2698 ms

> **Q:** When do I need to email my complaint about the scheme?
>
> **A:** For the Advocate Grant scheme, you need to email your complaint to the concerned regional office of the Backward Classes Development Department by 12.06.2025. [S1] Complaints submitted after the specified date will not be considered for any reason. [S1]
>
> *Card:* Advocate Grant (financial assistance to young advocates) (needs info)

- [S1] `kerala_advocate_grant--7f13ea25ed7d4bb95a6ee0b1557c382d-cff460:2:0`: Advocate Grant (financial assistance to young advocates), page 2, https://egrantz.kerala.gov.in/egrantz_uploads/7f13ea25ed7d4bb95a6ee0b1557c382d.pdf

**4. English, heldout (h019)**: judge 9/9 supported, TTFT 2910 ms

> **Q:** If I planted ginger on 0.02 hectare and it dies after a month, how much will I be paid?
>
> **A:** For ginger planted on 0.02 hectare, the insurance period starts one month after planting and lasts until five months. [S1] If the crop dies after a month, it falls within the covered period. [S1] The compensation amount for 0.02 hectare of ginger is Rs 15. [S1]
>
> *Card:* State crop insurance and natural-calamity relief for farmers (eligible)

- [S1] `kerala_agri_crop_insurance--crop-insurance-schemes-a4bb5c:1:1`: State crop insurance and natural-calamity relief for farmers, page 1, https://keralaagriculture.gov.in/en/crop-insurance-schemes/
- [S2] `kerala_agri_crop_insurance--crop-insurance-schemes-a4bb5c:1:0`: State crop insurance and natural-calamity relief for farmers, page 1, https://keralaagriculture.gov.in/en/crop-insurance-schemes/

**5. Manglish, heldout (h026)**: judge 8/8 supported, TTFT 2795 ms

> **Q:** njan kerala resident aayi, disability 40% kooduthal ulla student aanu, family income 30k aanu, scholarship kittumo?
>
> **A:** നിങ്ങൾക്ക് കേരള സർക്കാരിന്റെ 'Scholarship for Differently Abled Students' പദ്ധതിയ്ക്കും, കേന്ദ്ര സർക്കാരിന്റെ 'Pre- and Post-Matric Scholarships for Students with Disabilities' പദ്ധതിയ്ക്കും അർഹതയുണ്ട്. [S1] [S2]
>
> *Card:* Scholarship for Differently Abled Students (Kerala) (eligible)
> *Card:* Pre- and Post-Matric Scholarships for Students with Disabilities (eligible)

- [S1] `sjd_disability_scholarship--sdas-6a3a34:1:2`: Scholarship for Differently Abled Students (Kerala), page 1, https://www.myscheme.gov.in/schemes/sdas
- [S2] `scholarship_disabilities--post-dis-9c5dfa:1:2`: Pre- and Post-Matric Scholarships for Students with Disabilities, page 1, https://www.myscheme.gov.in/schemes/post-dis
- [S3] `sjd_disability_scholarship--sdas-6a3a34:1:1`: Scholarship for Differently Abled Students (Kerala), page 1, https://www.myscheme.gov.in/schemes/sdas

## 6. Five worst faithfulness failures (qwen)

| id | lang | faithfulness | unsupported / mis-cited statement(s) |
|---|---|---|---|
| q118 | en | 0.60 | unsupported: The excerpts provided do not contain a complete list of backward classes in Kerala, so I cannot confirm if your specific caste is included. · unsupported: Employability Enhancement Programme and skill assistance for OBC students (BCDD): The user's caste is not stated, so it cannot be verified against the OBC or SE |
| h017 | en | 0.67 | unsupported: Kerala Social Security Pensions (Sevana Pension): The user did not provide personal facts to determine eligibility · unsupported: National Social Assistance Programme (NSAP): The user did not provide personal facts to determine eligibility |
| h015 | ml | 0.75 | unsupported: Parinayam I & II (marriage assistance for differently abled women / daughters): അപേക്ഷകൻ കേരളത്തിലെ സ്ഥിരവാസിയായിരിക്കണം എന്ന നിബന്ധന പരിശോധിക്കാൻ വിവരം ഇല്ല. · unsupported: Parinayam I & II (marriage assistance for differently abled women / daughters): വിവാഹം ചെയ്യാൻ ആഗ്രഹിക്കുന്ന പെൺകുട്ടി 18 വയസ്സ് പൂർത്തിയാക്കിയിരിക്കണം എന്ന നിബ |
| q078 | en | 0.80 | unsupported: Snehapoorvam (assistance to orphaned children's education): The user did not state if the child is an orphan, which is required for eligibility |
| h009 | manglish | 0.82 | unsupported: Pradhan Mantri Awas Yojana - Urban (PMAY-U): ഉപയോക്താവിന്റെ വരുമാനം 4 ലക്ഷം രൂപയാണ്. · unsupported: Pradhan Mantri Awaas Yojana - Gramin (PMAY-G): ഉപയോക്താവ് നഗരത്തിലാണോ ഗ്രാമത്തിലാണോ എന്ന് വ്യക്തമല്ല. |

Most of these are not invented scheme facts. They are card reasons about what the user did *not* say, which the judge counts as unsupported. Prompt v5 targets them (next steps).

## 7. Where validation dropped content

### qwen (live bench)

| id | lang | reason | detail | dropped text |
|---|---|---|---|---|
| q058 | manglish | number_mismatch | 60 | ഉപയോക്താവ് കർഷക തൊഴിലാളിയാണെന്ന് പറഞ്ഞിട്ടുണ്ട്, എന്നാൽ പെൻഷൻ ലഭിക്കാനുള്ള പ്രായം അല്ലെങ്കിൽ മറ്റ് നിബന്ധനകൾ (ഉദാ: 60 വയസ്സ്) excerpts-ൽ വ്യക്തമായി പറ |
| h017 | en | number_mismatch | 60 | For the National Social Assistance Programme (NSAP), the total timeline for the different processes, including verification, Gram Sabha/Ward, Gram Pan |
| e001 | en | no_citation | - | Yes, you can join Atal Pension Yojana because you meet the eligibility criteria. |

3 statements dropped in 3 of 25 items; reasons: {'no_citation': 1, 'number_mismatch': 2}

### gpt-oss-20b (live bench, prompt v3)

| id | lang | reason | detail | dropped text |
|---|---|---|---|---|
| h022 | manglish | no_citation | - | PM‑KMY-ൽ ചേർക്കാൻ നിങ്ങൾക്ക് യോഗ്യതയുണ്ടോ എന്ന് നിശ്ചയിക്കാൻ കൂടുതൽ വിവരങ്ങൾ ആവശ്യമാണ്. |
| h017 | en | no_citation | - | It takes 15 days for verification after you submit the application, 20 days for Gram Sabha/Ward, 15 days for Gram Panchayat/Municipality, and 10 days  |
| h017 | en | number_mismatch | 60 | In total, the pension can be approved within about 60 days from the date of submission.[C5] |
| q118 | en | no_citation | - | The list of backward castes in Kerala is published in the state’s OBC list and the SEBC list. |
| q118 | en | no_citation | - | The OBC list is annexed as part‑1 of the Kerala State and Subordinate Service Rule, 1958, and the SEBC list is published under various Government Orde |
| q118 | en | no_citation | - | (P) 208/66/Edn., G.O. |
| q118 | en | no_citation | - | (Ms) No. 95/08/SCSTDD, etc.). |
| q118 | en | no_citation | - | These lists include castes such as Chetty/Chetties, Dheevara, Ezhavas, Muslims, Latin Catholics, Anglo Indians, and many others as detailed in the exc |
| q118 | en | no_citation | - | To find out whether your specific caste is listed, you need to provide the name of your caste. |
| h009 | manglish | no_citation | - | നിങ്ങളുടെ കുടുംബ വരുമാനം ₹4,00,000 ആകയാല്‍ നിങ്ങള്‍ PMAY‑U യിലെ Low Income Group (LIG) വിഭാഗത്തില്‍പ്പെടുന്നു, അതിനാല്‍ നിങ്ങള്‍ക്ക് ഈ പദ്ധതിയില്‍ പങ് |
| h009 | manglish | no_citation | - | LIG വിഭാഗത്തിന് ലഭ്യമായ വീടുകളുടെ വലുപ്പം പദ്ധതിയുടെ നിര്‍ദ്ദേശങ്ങളില്‍ വ്യക്തമാക്കിയിട്ടില്ല, പക്ഷേ സാധാരണയായി LIG‑ക്കായി 30‑35 ചതുരശ്ര മീറ്റര്‍ വരെയ |
| h009 | manglish | number_mismatch | 30, 35 | LIG‑ക്കായി 30‑35 ചതുരശ്ര മീറ്റര്‍ വരെയുള്ള വീടുകള്‍ ലഭ്യമാണ് (പദ്ധതിയുടെ നിര്‍ദ്ദേശങ്ങള്‍ അനുസരിച്ച്) [C2] |

27 statements dropped in 16 of 37 items; reasons: {'no_citation': 22, 'number_mismatch': 5}

### gpt-oss-20b (eligibility, prompt v4)

| id | lang | reason | detail | dropped text |
|---|---|---|---|---|
| e003 | manglish | no_citation | - | നിങ്ങളുടെ അമ്മ 67 വയസ്സുള്ളതും ഇന്ത്യൻ പൗരത്വമുള്ളതും BPL കുടുംബത്തിൽ ഉള്ളതുമാണ്. |
| e003 | manglish | number_mismatch | 60, 80 | മാസം ₹200 (60-79 വയസ്സ്) അല്ലെങ്കിൽ ₹500 (80 വയസ്സ് മുതലുള്ളവർ) [C2] |
| e004 | en | no_citation | - | You are not eligible for the Indira Gandhi National Old Age Pension Scheme (IGNOAPS). |
| e005 | ml | number_mismatch | 40, 79 | നിങ്ങൾ 40‑79 വയസ്സുള്ള വിധവയാണ് [C2] |
| e007 | en | no_citation | - | You may be eligible for the Indira Gandhi National Widow Pension, but we need more information to confirm. |
| e008 | en | no_citation | - | Your brother is eligible for the Indira Gandhi National Disability Pension Scheme (IGNDPS). |
| e009 | manglish | no_citation | - | BPL Card |
| e009 | manglish | no_citation | - | Aadhaar card |
| e009 | manglish | no_citation | - | Age Proof |
| e009 | manglish | no_citation | - | Disability certificate (80% and more) |
| e009 | manglish | no_citation | - | Passport Size Photographs |
| e010 | en | no_citation | - | You are eligible to join PM‑KMY. |

16 statements dropped in 10 of 15 items; reasons: {'no_citation': 14, 'number_mismatch': 2}

The number check drops sums the model computed itself: "60 days in total" (h017) and "30–35 m²" (h009) are not in the source. It also drops numbers cited to the wrong chunk: e003's "₹200 (60-79)" cited C2, but the ages were in another chunk. Most no_citation drops are hedges or verdict sentences without a tag.

## 8. Changes that moved the numbers

1. **Prompt v2: worked tag example.** Without it, gpt-oss-20b cited only in the
   `citations` arrays, every sentence failed validation, and answers abstained.
2. **User numbers allowed.** The validator accepts numbers from the user's own question.
   Before this, "65 വയസ്സുള്ള…" was dropped as a number mismatch.
3. **Prompt v3: don't abstain when only a user fact is missing.** Instead return
   eligible=null with a follow-up question. This fixed e007.
4. **Per-field citation tags on scheme cards.** Card fields had been credited with the
   whole card's citations, which pulled measured citation accuracy down to 0.2 on the dry
   run.
5. **Prompt v4 plus rebuilding from validated cards.** gpt-oss-20b eligibility accuracy
   rose from 0.467 to 0.933 and false abstentions from 0.29 to 0.
6. **Passive thread waiting.** Retrieval p95 fell from 2078 to 1531 ms.
7. **Rewrite deadline (2 s) and bounded generation retries.** In a smoke test while
   qwen's quota was exhausted, one request had taken 83 s.

## 9. Next steps

- **Finish qwen's 12 eligibility items** and re-measure TTFT live once Groq's daily
  window frees up (the `--patient --resume` command above).
- **Prompt v5:** card reasons must state an excerpt rule, and missing facts go only in
  `follow_up_question`. Most of qwen's worst statements are reasons like "the user did
  not provide…".
- **Hosted Jina rerank:** add `JINA_API_KEY` and benchmark `deploy`.
- **Larger answer eval:** on a paid tier, or across several days with `--resume`.
