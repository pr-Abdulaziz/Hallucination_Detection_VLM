# Strict Paper Comparison

## Evaluated Models

- `llava15_base_7b` -> `models/llava-v1.5-7b`
- `two_shot_verified_margin_hsa_b32_e1` -> `output/fghd/exp_2shot_verified_margin_hsa_b32_e1`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | ok |  |
| object_halbench | ok |  |
| pope_adv | ok |  |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope_adv | f1 | 84.5 |  | 84.9 |  |  | LLaVA-1.5 | HSA-DPO w/ LLaVA-1.5 |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.