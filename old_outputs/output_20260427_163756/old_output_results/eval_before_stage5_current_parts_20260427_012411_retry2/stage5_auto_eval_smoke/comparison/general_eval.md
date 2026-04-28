# General Evaluation

## STAGE3 Summary

| metric | value |
| --- | --- |
| approvals_required | 2 |
| backend | gemini_openai_two_vote |
| dropped_rows | 7545 |
| preference_pairs_emitted | 98 |
| stage3_dir | /workspace/Hallucination_Detection_VLM/output/fghd/stage3 |
| total_input_rows | 7643 |
| vote_count | 2 |
| vote_rows_processed | 7643 |

## STAGE4 Summary

| metric | value |
| --- | --- |
| final_train_loss | 0.037 |
| stage4_dir | /workspace/Hallucination_Detection_VLM/output/fghd/stage5_llava_margin |

## Benchmark Summary

| benchmark | model | comparable_to_paper | note |
| --- | --- | --- | --- |
| pope_adv | llava15_base_13b | True |  |
| pope_adv | ours_stage5_lora | True |  |
| object_halbench | llava15_base_13b | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| object_halbench | ours_stage5_lora | False | rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction |
| amber | llava15_base_13b | False | local AMBER generative evaluation; automatic and no external judge API |
| amber | ours_stage5_lora | False | local AMBER generative evaluation; automatic and no external judge API |

## Runtime Summary

- General evaluation may include local proxy or supplemental metrics.