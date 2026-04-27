from __future__ import annotations

from pathlib import Path


FG_PIPELINE_ROOT = Path(__file__).resolve().parent
FG_DATA_DIR = FG_PIPELINE_ROOT / "data"
REPO_ROOT = FG_PIPELINE_ROOT.parent

# Stage-1-owned mirror of the released detection annotations (input).
DEFAULT_DETECTION_INPUT = FG_DATA_DIR / "hsa_dpo_detection.jsonl"
DEFAULT_SMOKE_DETECTION_INPUT = FG_DATA_DIR / "smoke_detection.jsonl"

# Stage 1 default output layout.
STAGE1_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage1"
DEFAULT_STAGE1_OUTPUT = STAGE1_OUTPUT_DIR / "detection_critiques.jsonl"
DEFAULT_STAGE1_STATS = STAGE1_OUTPUT_DIR / "stats.json"

# Stage 1 reads the same mirrored file by default.
DEFAULT_STAGE1_INPUT = DEFAULT_DETECTION_INPUT

# Stage 2 default output layout.
STAGE2_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage2"
DEFAULT_STAGE2_OUTPUT = STAGE2_OUTPUT_DIR / "rewrites.jsonl"
DEFAULT_STAGE2_STATS = STAGE2_OUTPUT_DIR / "stats.json"

# Stage 2 reads Stage 1 output by default.
DEFAULT_STAGE2_INPUT = DEFAULT_STAGE1_OUTPUT

# Stage 3 default output layout.
STAGE3_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage3"
DEFAULT_STAGE3_OUTPUT = STAGE3_OUTPUT_DIR / "vote_records.jsonl"
DEFAULT_STAGE3_PREFERENCES = STAGE3_OUTPUT_DIR / "preference_pairs.jsonl"
DEFAULT_STAGE3_STATS = STAGE3_OUTPUT_DIR / "stats.json"

# Stage 3 reads Stage 2 output by default.
DEFAULT_STAGE3_INPUT = DEFAULT_STAGE2_OUTPUT

# Stage 4 project pipeline defaults (wrapper over the existing baseline trainer).
STAGE4_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage4_llava"
DEFAULT_STAGE4_DATA = DEFAULT_STAGE3_PREFERENCES

# Stage 4 repair pass defaults.
STAGE4_REPAIR_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage4"
DEFAULT_STAGE4_REPAIR_OUTPUT = STAGE4_REPAIR_OUTPUT_DIR / "repair_records.jsonl"
DEFAULT_STAGE4_REPAIR_PREFERENCES = STAGE4_REPAIR_OUTPUT_DIR / "repair_preferences.jsonl"
DEFAULT_STAGE4_FINAL_PREFERENCES = STAGE4_REPAIR_OUTPUT_DIR / "final_preference_pairs.jsonl"
DEFAULT_STAGE4_REPAIR_STATS = STAGE4_REPAIR_OUTPUT_DIR / "stats.json"

# Stage 5 severity-margin DPO defaults.
STAGE5_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage5_llava_margin"
DEFAULT_STAGE5_DATA = DEFAULT_STAGE4_FINAL_PREFERENCES

# Paper-faithful additive pipeline defaults.
PAPER_STAGE1_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage1"
DEFAULT_PAPER_STAGE1_OUTPUT = PAPER_STAGE1_OUTPUT_DIR / "d_faif.jsonl"
DEFAULT_PAPER_STAGE1_STATS = PAPER_STAGE1_OUTPUT_DIR / "stats.json"

PAPER_STAGE2_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage2"
DEFAULT_PAPER_STAGE2_DETECTOR_DATA = PAPER_STAGE2_OUTPUT_DIR / "detector_train.json"
DEFAULT_PAPER_STAGE2_STATS = PAPER_STAGE2_OUTPUT_DIR / "detector_split_stats.json"

PAPER_STAGE3_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage3"
DEFAULT_PAPER_STAGE3_DETECTOR = PAPER_STAGE3_OUTPUT_DIR / "detector_lora"
DEFAULT_PAPER_STAGE3_DETECTIONS = PAPER_STAGE3_OUTPUT_DIR / "detections.jsonl"
DEFAULT_PAPER_STAGE3_STATS = PAPER_STAGE3_OUTPUT_DIR / "detection_stats.json"

PAPER_STAGE4_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage4"
DEFAULT_PAPER_STAGE4_REWRITES = PAPER_STAGE4_OUTPUT_DIR / "rewrite_records.jsonl"
DEFAULT_PAPER_STAGE4_PREFERENCES = PAPER_STAGE4_OUTPUT_DIR / "preference_pairs.jsonl"
DEFAULT_PAPER_STAGE4_STATS = PAPER_STAGE4_OUTPUT_DIR / "stats.json"

PAPER_STAGE5_DPO_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage5_dpo"
PAPER_STAGE5_HSA_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "paper_stage5_hsa_dpo"
