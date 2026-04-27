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
| amber | amber_chair | 16.3 | 2.1 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cog | 0.0 | 1.2 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_cover | 40.6 | 47.3 | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | amber_hal | 60.0 | 13.4 | False | local AMBER generative evaluation; automatic and no external judge API |
| object_halbench | chairi | 10.810811 | 3.2 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| object_halbench | chairs | 60.0 | 5.3 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| pope_adv | accuracy | 80.0 |  | False | no paper reference row |
| pope_adv | precision | 100.0 |  | False | no paper reference row |
| pope_adv | recall | 66.666667 |  | False | no paper reference row |

## Notes

- These rows are excluded from the strict paper comparison table.

- Reasons include proxy evaluators, unmatched protocols, or missing paper reference rows.
## Evaluation Note

This evaluation uses automatic metrics only and does not use OpenAI API keys or external judge APIs.
