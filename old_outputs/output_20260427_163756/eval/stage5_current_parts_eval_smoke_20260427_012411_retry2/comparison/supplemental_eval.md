# Supplemental Local Evaluation

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | ok |  |
| object_halbench | ok |  |
| pope_adv | ok |  |

## Supplemental Rows

| benchmark | metric | our value | paper reference | strictly comparable | note |
| --- | --- | --- | --- | --- | --- |
| amber | amber_chair | 0.0 | 2.1 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cog | 0.0 | 1.2 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cover | 37.5 | 47.3 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_hal | 0.0 | 13.4 | False | local AMBER generative evaluation; automatic and no external judge API |
| object_halbench | chairi | 3.030303 | 3.2 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| object_halbench | chairs | 20.0 | 5.3 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| pope_adv | accuracy | 60.0 |  | False | no paper reference row |
| pope_adv | precision | 100.0 |  | False | no paper reference row |
| pope_adv | recall | 33.333333 |  | False | no paper reference row |

## Notes

- These rows are excluded from the strict paper comparison table.

- Reasons include proxy evaluators, unmatched protocols, or missing paper reference rows.
## Run Note

This evaluation compares LLaVA-1.5-7B base, direct paper-style HSA-DPO, Part A normal DPO, and Part B API-verified severity-margin HSA-DPO.
Existing older evaluation outputs were backed up to `output/old_output_results/eval_before_stage5_current_parts_20260427_012411_retry2` before this run.
