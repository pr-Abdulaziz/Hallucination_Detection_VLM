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
| amber | amber_chair | 5.6 | 2.1 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cog | 2.7 | 1.2 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cover | 50.4 | 47.3 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_hal | 26.9 | 13.4 | False | local AMBER generative evaluation; automatic and no external judge API |
| object_halbench | chairi | 12.808099 | 3.2 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| object_halbench | chairs | 40.333333 | 5.3 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| pope_adv | accuracy | 56.3 |  | False | no paper reference row |
| pope_adv | precision | 0.0 |  | False | no paper reference row |
| pope_adv | recall |  |  | False | no paper reference row |

## Notes

- These rows are excluded from the strict paper comparison table.

- Reasons include proxy evaluators, unmatched protocols, or missing paper reference rows.
## Run Note

This evaluation compares LLaVA-1.5-7B base against the two-shot API-verified severity-margin HSA-DPO model. Optional older adapters are included only when their weight files are present.
Existing older evaluation outputs were backed up to `output/old_output_results/eval_before_2shot_verified_margin_20260428_172728` before this run.
