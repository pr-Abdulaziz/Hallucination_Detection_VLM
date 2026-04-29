# General Evaluation

## STAGE3 Summary

| metric | value |
| --- | --- |
| accepted_rows | 7132 |
| api_judge | gemini_openai |
| decision_rule | either |
| gemini_model | gemini-2.5-flash-lite |
| openai_model | gpt-4o-mini |
| prompt_mode | two_shot |
| rejected_rows | 1254 |
| stage3_dir | output\fghd\released_pref_stage3_2shot_experiment |
| total_rows | 8386 |
| workers | 3 |

## STAGE4 Summary

| metric | value |
| --- | --- |
| final_train_loss | 0.6542 |
| stage4_dir | output\fghd\exp_2shot_verified_margin_hsa_b32_e1 |

## Benchmark Summary

| benchmark | model | comparable_to_paper | note |
| --- | --- | --- | --- |
| amber | llava15_base_7b | False | local AMBER generative evaluation; automatic and no external judge API |
| object_halbench | llava15_base_7b | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| pope_adv | llava15_base_7b | True |  |
| amber | two_shot_verified_margin_hsa_b32_e1 | False | local AMBER generative evaluation; automatic and no external judge API |
| object_halbench | two_shot_verified_margin_hsa_b32_e1 | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| pope_adv | two_shot_verified_margin_hsa_b32_e1 | True |  |

## Runtime Summary

- General evaluation may include local proxy or supplemental metrics.