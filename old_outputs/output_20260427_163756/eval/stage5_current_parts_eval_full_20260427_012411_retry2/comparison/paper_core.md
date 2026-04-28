# Strict Paper Comparison

## Evaluated Models

- `llava15_base_7b` -> `models/llava-v1.5-7b`
- `direct_paper_hsa_b32_e1` -> `output/fghd/exp_direct_paper_hsa_b32_e1`
- `direct_normal_dpo_b32_e1` -> `output/fghd/exp_direct_normal_dpo_b32_e1`
- `api_verified_margin_hsa_b32_e1` -> `output/fghd/exp_api_verified_margin_hsa_b32_e1`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | ok |  |
| object_halbench | ok |  |
| pope_adv | ok |  |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope_adv | f1 | 84.184397 | 83.912576 | 84.9 | -0.271821 | -0.003229 | llava15_base_7b | HSA-DPO w/ LLaVA-1.5 |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.