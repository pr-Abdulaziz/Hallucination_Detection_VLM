# Strict Paper Comparison

## Evaluated Models

- `llava_1_5_13b_base` -> `/workspace/Hallucination_Detection_VLM/models/llava-v1.5-13b`
- `stage4_llava_gemini139` -> `/workspace/Hallucination_Detection_VLM/output/fghd/stage4_llava`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| mfhallubench | skipped | Missing MFHaluBench predictions file: /workspace/Hallucination_Detection_VLM/playground/data/eval/mfhallubench/predictions.jsonl |
| mhalubench | skipped | Missing MHaluBench predictions file: /workspace/Hallucination_Detection_VLM/playground/data/eval/mhalubench/predictions.jsonl |
| pope_adv | skipped | Missing POPE question file: /workspace/Hallucination_Detection_VLM/playground/data/eval/pope/llava_pope_test.jsonl |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | none |  |  |  |  |  |  |  |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.