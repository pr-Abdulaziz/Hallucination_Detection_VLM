# Strict Paper Comparison

## Evaluated Models

- `llava15_base_13b` -> `models/llava-v1.5-13b`
- `ours_stage5_lora` -> `output/fghd/stage5_llava_margin`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | skipped | Missing AMBER question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/amber/query_generative.jsonl |
| hss | skipped | HSS local judged evaluation is supplemental only and requires a local judge backend implementation |
| llava_bench_wild | skipped | benchmark requires --openai-judge-model or OPENAI_JUDGE_MODEL |
| mfhallubench | skipped | Missing MFHaluBench predictions file: /workspace/Hallucination_Detection_VLM/playground/data/eval/mfhallubench/predictions.jsonl |
| mhalubench | skipped | Missing MHaluBench predictions file: /workspace/Hallucination_Detection_VLM/playground/data/eval/mhalubench/predictions.jsonl |
| mmhal_bench | skipped | benchmark requires --openai-judge-model or OPENAI_JUDGE_MODEL |
| object_halbench | skipped | Missing Object HalBench question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/object-halbench/questions.jsonl |
| pope_adv | skipped | Missing POPE question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/pope/llava_pope_test.jsonl |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | none |  |  |  |  |  |  |  |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.