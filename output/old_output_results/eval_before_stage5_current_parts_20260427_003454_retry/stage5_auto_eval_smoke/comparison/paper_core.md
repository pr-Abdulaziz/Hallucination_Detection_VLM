# Strict Paper Comparison

## Evaluated Models

- `llava15_base_13b` -> `models/llava-v1.5-13b`
- `ours_stage5_lora` -> `output/fghd/stage5_llava_margin`

## Benchmark Availability

| benchmark | status | note |
| --- | --- | --- |
| amber | ok |  |
| object_halbench | ok |  |
| pope_adv | ok |  |

## Strictly Comparable Rows

| benchmark | metric | baseline reproduced | our reproduced | paper reference | delta vs baseline | relative delta vs baseline | baseline row | paper row | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope_adv | f1 | 80.0 | 80.0 | 84.9 | 0.0 | 0.0 | llava15_base_13b | HSA-DPO w/ LLaVA-1.5 |  |

## Fairness Contract

- This table includes only rows marked strictly comparable to the referenced paper.

- Supplemental or proxy metrics are reported separately and do not appear in this delta table.