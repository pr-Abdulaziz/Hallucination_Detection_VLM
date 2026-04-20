## FG Pipeline

This package is an extension layer on top of the original HSA-DPO repository.

Design rule:
- reuse the original `hsa_dpo/` code and datasets wherever possible
- avoid moving or renaming the original training/data paths
- make new outputs compatible with the original trainer before introducing new trainer logic

Current reuse points:
- detection bootstrap defaults to `fg_pipeline/data/hsa_dpo_detection.jsonl`
- `fg_pipeline/data/hsa_dpo_detection.jsonl` is a Stage-3-owned mirror of the original
  `hsa_dpo/data/hsa_dpo_detection.jsonl` so the baseline layout remains intact
- `fg_pipeline/paths.py` defines the extension-layer default paths used by Stage 3
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

Stage 6 adaptive extensions:
- prefer the carried `image` field over reconstructing paths from `id`
- consume `pair_confidence`, `severity_weight`, and `adaptive_weight` when present
