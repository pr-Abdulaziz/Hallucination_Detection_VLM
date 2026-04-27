# Strict Paper Comparison

## Evaluated Models

- `llava_1_5_13b_base` -> `/workspace/Hallucination_Detection_VLM/models/llava-v1.5-13b`
- `stage4_llava_gemini139` -> `/workspace/Hallucination_Detection_VLM/output/fghd/stage4_llava`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | skipped | Missing AMBER question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/amber/query_generative.jsonl |
| hss | skipped | HSS local judged evaluation is supplemental only and requires a local judge backend implementation |
| llava_bench_wild | skipped | local judged evaluation not implemented for this benchmark in the current stack |
| mmhal_bench | skipped | local judged evaluation not implemented for this benchmark in the current stack |
| object_halbench | skipped | Missing Object HalBench question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/object-halbench/questions.jsonl |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | none |  |  |  |  |  |  |  |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.