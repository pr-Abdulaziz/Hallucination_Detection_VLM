## FG Pipeline

This package is an extension layer on top of the original HSA-DPO repository.

Design rule:
- reuse the original `hsa_dpo/` code and datasets wherever possible
- avoid moving or renaming the original training/data paths
- make new outputs compatible with the original trainer before introducing new trainer logic

Current reuse points:
- detection bootstrap reads `hsa_dpo/data/hsa_dpo_detection.jsonl`
- stage 6 training reuses `hsa_dpo_train.sh`
- the adaptive trainer stub subclasses `hsa_dpo.trainer.llava_dpo_trainer.LlavaDPOTrainer`

Compatibility target for `output/fghd/D_pref_clean.jsonl`:
- `id`
- `question`
- `chosen`
- `rejected`
- `chosen_score`
- `rejected_score`

Those fields match what `hsa_dpo/models/llava-v1_5/train_dpo.py` already expects.
