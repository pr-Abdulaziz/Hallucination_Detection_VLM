# Supplemental Local Evaluation

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

## Supplemental Rows

| benchmark | metric | our value | paper reference | strictly comparable | note |
| --- | --- | --- | --- | --- | --- |
| none | none |  |  |  |  |

## Notes

- These rows are excluded from the strict paper comparison table.

- Reasons include proxy evaluators, unmatched protocols, or missing paper reference rows.