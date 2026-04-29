# Script Entry Points

This folder keeps only the runnable entry points needed for the reported pipeline.

## Data and Preference Construction

- `run_paper_stage1_faif.sh`: parse released fine-grained hallucination supervision.
- `run_paper_stage2_detector_dataset.sh`: build detector-style data from the parsed supervision.
- `run_paper_stage3_detect.sh`: optionally run a local/API hallucination detector and emit normalized critique records.
- `run_paper_stage4_rewrite.sh`: run critique-guided LLaVA rewrite and emit preference pairs from Stage 1 or Stage 3 critique records.
- `run_released_pref_stage3_validate.sh`: validate released preference pairs with Gemini/OpenAI judges. Set `SHOT_MODE=zero_shot` or `SHOT_MODE=two_shot`.
- `run_released_pref_stage4_repair.sh`: repair rejected pairs with local LLaVA.
- `run_released_pref_stage5_openai_verify.sh`: re-verify repaired pairs with `gpt-4.1-mini` and merge approved repairs.

## Training

- `run_direct_stage5_paper_hsa_batch32_epoch1.sh`: direct HSA-DPO baseline.
- `run_direct_stage5_normal_dpo_batch32_epoch1.sh`: matched standard DPO baseline.
- `run_2shot_verified_margin_hsa_batch32_epoch1.sh`: two-shot judged HSA-DPO-M experiment.
- `run_paper_stage5_train_hsa.sh`: configurable Stage 5 training wrapper used by the fixed training scripts.

## Evaluation

- `setup_stage5_eval_assets.sh`: prepare benchmark assets.
- `watch_stage5_eval_after_training.sh`: evaluate the base, direct HSA-DPO, standard DPO, and zero-shot judged HSA-DPO-M models.
- `watch_2shot_eval_after_training.sh`: evaluate the two-shot judged HSA-DPO-M model, with optional baselines when available.
