# Project Workflow

This file is the short team-facing map of the repo.
It explains:

- what belongs to the original baseline
- what belongs to our new pipeline
- what we are doing next

For the full research method, see [README.md](README.md).

## Last Update

- **Project method has been redesigned around five stages:** (1) critique
  detection / extraction, (2) critique-guided rewrite, (3) majority-vote
  preference validation, (4) LLaVA repair of Stage 3 rejects, (5)
  severity-margin DPO. The prior confidence-based
  Stages 3-5 remain removed; nothing about the new design brings back
  confidence scoring, temperature scaling, group-conditional thresholds,
  pair-confidence, or CRC / CV-CRC selection.
- **Stage 1 is implemented** as `fg_pipeline/stage1/`. The default
  backend (`ReleasedAnnotationBackend`) parses the released
  `hsa_dpo_detection.jsonl` supervision into a normalized `Stage1Record`.
  The local research path now also includes a `LlavaDetectorBackend`,
  detector dataset-prep, and benchmark-export entrypoints without changing
  the output schema.
- **Stage 2 is implemented** as `fg_pipeline/stage2/`. It consumes Stage
  1 JSONL and emits one corrected rewrite per hallucinated record;
  non-hallucinated rows are skipped. The default
  `TemplateRewriteBackend` is a deterministic smoke backend; the
  intended research backend is `LlavaRewriteBackend`, which uses the
  vendored LLaVA-v1.5 stack and is wired for real use. A
  `RewriteBackend` protocol is exposed so other backends can be plugged
  in without changing the Stage 2 output schema.
- **Stage 3 is implemented** as `fg_pipeline/stage3/`. It consumes Stage
  2 rewrites, runs verification votes per row, writes a Stage 3 audit
  artifact, and exports trainer-compatible preference pairs. The default
  `HeuristicVerificationBackend` is deterministic and smoke-oriented; research
  runs use `gemini_openai_two_vote`, `gemini_two_vote`, or `gemini_llava_two_vote`.
- **Stage 4 is implemented** as `fg_pipeline/stage4/`. It repairs Stage 3
  rejected rewrites with LLaVA, writes repair audit rows, and combines Stage 3
  approved pairs plus repaired pairs into `output/fghd/stage4/final_preference_pairs.jsonl`.
- **Stage 5 trains the final model** through `scripts/run_stage5_train.sh`.
  It uses the final Stage 4 preference file and the new `severity_margin` DPO
  objective. The older `scripts/run_stage4_train.sh` remains available only as
  a Stage 3-only HSA-DPO baseline wrapper.
- **The released HSA-DPO trainer remains the training stack.**
  `hsa_dpo_train.sh` still calls
  `hsa_dpo/models/llava-v1_5/train_dpo.py`, which uses
  `hsa_dpo.trainer.LlavaDPOTrainer` directly. A small, generic
  image-path resolver (prefer explicit `image`, fall back to
  `<image_folder>/<id>.jpg`) is kept inline as a runtime fix.
- **Vendored LLaVA-v1.5 compatibility fixes are preserved.** These are
  generic fixes, not tied to any removed approach.
- **Evaluation tooling is now split.** `fg_pipeline/eval/` provides
  strict paper comparison, supplemental local evaluation, and general
  runtime reporting.

## Current Architecture

The repo has two layers.

- `hsa_dpo/`
  Original HSA-DPO baseline. Still the active Stage 5 training stack.
- `fg_pipeline/`
  Extension layer. Hosts Stages 1-4, shared utilities, curated data
  fixtures, and the evaluation suite.

## Folder Layout

```text
.
├── hsa_dpo/
│   ├── data/
│   │   ├── hsa_dpo_preference_llava1dot5.jsonl   ← baseline preference data
│   │   ├── hsa_dpo_detection.jsonl               ← original released detection copy
│   │   └── images/                               ← baseline training images
│   ├── models/llava-v1_5/                        ← original training entrypoint
│   └── trainer/                                  ← original DPO trainer
│
├── fg_pipeline/
│   ├── stage1/                                   ← Stage 1 critique extraction
│   │   ├── schemas.py
│   │   ├── parser.py
│   │   ├── backends.py
│   │   ├── run_stage1.py                         ← parse / detector inference CLI
│   │   ├── run_stage1_detector_dataset.py        ← detector SFT data prep
│   │   └── run_stage1_export_benchmarks.py       ← detector benchmark export
│   ├── stage2/                                   ← Stage 2 critique-guided rewrite
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   ├── backends.py
│   │   └── run_stage2.py                         ← CLI entrypoint
│   ├── stage3/                                   ← Stage 3 majority-vote validation
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   ├── backends.py
│   │   └── run_stage3.py                         ← CLI entrypoint
│   ├── stage4/                                   ← Stage 4 LLaVA repair
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   └── run_stage4_repair.py                  ← CLI entrypoint
│   ├── data/
│   │   ├── hsa_dpo_detection.jsonl               ← Stage 1 input mirror
│   │   ├── smoke_detection.jsonl                 ← Stage 1 smoke fixture
│   │   └── (paper reference tables, etc.)
│   ├── eval/                                     ← paper/general evaluation layer
│   ├── paths.py                                  ← fg_pipeline-owned default paths
│   ├── schemas.py                                ← shared records (preference + Stage 1/3 re-exports)
│   └── io_utils.py                               ← JSONL helpers
│
├── scripts/
│   ├── run_stage1_critiques.sh                   ← Stage 1 parser / inference launcher
│   ├── run_stage1_detector_dataset.sh            ← Stage 1 detector SFT prep
│   ├── run_stage1_detector_train.sh              ← Stage 1 detector train wrapper
│   ├── run_stage1_export_benchmarks.sh           ← Stage 1 benchmark export
│   ├── run_stage2_rewrites.sh                    ← Stage 2 launcher
│   ├── run_stage3_validate.sh                    ← Stage 3 launcher
│   ├── run_stage4_rewrite.sh                     ← Stage 4 repair launcher
│   ├── run_stage5_train.sh                       ← Stage 5 training launcher
│   ├── run_stage4_train.sh                       ← legacy Stage 3-only training wrapper
│   ├── run_paper_eval.sh
│   ├── run_general_eval.sh
│   └── vastai/                                   ← environment bootstrap
│
├── tests/
│   ├── test_stage1_parser.py
│   ├── test_stage1_backend_cli.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   ├── test_stage4.py
│   ├── test_stage5_loss.py
│   ├── test_eval_pope_smoke.py
│   ├── test_eval_reference_tables.py
│   ├── test_eval_reporting.py
│   └── test_eval_schemas.py
│
├── hsa_dpo_train.sh                              ← shared DPO training entrypoint
└── vg/images/                                    ← Visual Genome images (Stage 1 supervision)
```

## Ownership Rules

- Do not remove or rename files under `hsa_dpo/` unless we explicitly decide to patch the baseline.
- `fg_pipeline/data/hsa_dpo_detection.jsonl` is a mirror of `hsa_dpo/data/hsa_dpo_detection.jsonl`.
- `vg/images/` is the image store for the redesigned Stage 1-5 pipeline.
  Stage 1 reads supervision rows that point into `vg/images/...`, and those
  image paths are carried forward through Stage 2, Stage 3, Stage 4 repair,
  and Stage 5 training.
- `hsa_dpo/data/images/` is only for baseline preference training.
- Do not mix the two image stores.

## Pipeline (current state)

- Stage 1 — Critique detection / critique extraction
  *(implemented; released-annotation parser is default, LLaVA detector is the local research path)*
- Stage 2 — Critique-guided rewrite
  *(implemented; template backend is smoke-only; LLaVA backend is the research path)*
- Stage 3 — Majority-vote preference validation
  *(implemented; heuristic backend is smoke-only, Gemini and Gemini+LLaVA are the research paths)*
- Stage 4 — LLaVA repair of Stage 3 rejects
  *(implemented via `scripts/run_stage4_rewrite.sh`)*
- Stage 5 — Severity-margin DPO
  *(implemented via `scripts/run_stage5_train.sh` -> `hsa_dpo_train.sh`)*

For the released Stage 1 data, the sentence after `Description to Assess:` is
the candidate response being evaluated. The normalized Stage 1 artifact stores
that sentence in `response_text`; the raw GPT supervision is preserved under
`metadata.raw_annotation_text`. Stage 2 passes `response_text` as
`original_response` and writes one corrected rewrite to `rewrite_response`.

## What Exists Today

- Stage 1 critique extraction (`fg_pipeline/stage1/` + `scripts/run_stage1_critiques.sh`).
- Stage 2 critique-guided rewrite (`fg_pipeline/stage2/` + `scripts/run_stage2_rewrites.sh`).
- Stage 3 majority-vote validation (`fg_pipeline/stage3/` + `scripts/run_stage3_validate.sh`).
- Stage 4 LLaVA repair (`fg_pipeline/stage4/` + `scripts/run_stage4_rewrite.sh`).
- Stage 5 severity-margin training wrapper (`scripts/run_stage5_train.sh`) over the HSA-DPO trainer path.
- Legacy Stage 3-only HSA-DPO wrapper (`scripts/run_stage4_train.sh`) for baseline comparison.
- Paper-core and general evaluation tooling under `fg_pipeline/eval/`.
- Shared JSONL / path / schema utilities under `fg_pipeline/`.
- Vast AI bootstrap scripts under `scripts/vastai/`.
- Per-instance Vast overrides through ignored `scripts/vastai/defaults.local.env`
  (copy from `scripts/vastai/defaults.env.example` when moving to a new box).

## What We Will Do Next

1. GPU-validate the Stage 1 detector path and Stage 2 rewrite path on Vast.
2. Run the full Stage 1 -> Stage 5 pipeline on a suitable GPU box,
   but inspect Stage 3 validation stats before Stage 4 repair and Stage 5 training.
3. Reproduce the paper baseline separately with the released preference file.
4. Keep strict paper comparison separate from supplemental local metrics.

## Execution Guide

### Stage 1 critique extraction (local / CPU-friendly)

Run from the repo root:

```bash
bash scripts/run_stage1_critiques.sh
```

Or directly via the module:

```bash
python -m fg_pipeline.stage1.run_stage1 \
  --input  fg_pipeline/data/hsa_dpo_detection.jsonl \
  --output output/fghd/stage1/detection_critiques.jsonl \
  --stats-out output/fghd/stage1/stats.json
```

Useful flags: `--backend released_annotations` (default), `--limit N`
for smoke runs, `--strict` to fail on malformed hallucinated rows.

Detector research path helpers:

```bash
bash scripts/run_stage1_detector_dataset.sh
bash scripts/run_stage1_detector_train.sh
bash scripts/run_stage1_export_benchmarks.sh
```

For a single-GPU Vast box, start with the parser-backed Stage 1 path above.
The detector helpers are research add-ons, not a prerequisite for the
Stage 1 -> 4 training pipeline.

### Stage 2 critique-guided rewrite (requires Stage 1 output)

Run Stage 1 first, then Stage 2:

```bash
bash scripts/run_stage1_critiques.sh
bash scripts/run_stage2_rewrites.sh
```

Or directly:

```bash
python -m fg_pipeline.stage2.run_stage2 \
  --input  output/fghd/stage1/detection_critiques.jsonl \
  --output output/fghd/stage2/rewrites.jsonl \
  --stats-out output/fghd/stage2/stats.json
```

Useful flags: `--backend template` (default, smoke-only) or `--backend llava`
(real, requires `--model-path models/llava-v1.5-13b`), `--limit N` for smoke
runs, `--strict` to fail on empty rewrites.

Stage 2 skips non-hallucinated rows. Output goes to
`output/fghd/stage2/rewrites.jsonl` with a compact `stats.json` alongside.

For real experiments, do not use the default `template` backend. Use:

```bash
BACKEND=llava MODEL_PATH=models/llava-v1.5-13b bash scripts/run_stage2_rewrites.sh
```

### Stage 3 majority-vote preference validation (requires Stage 2 output)

Run Stages 1-2 first, then Stage 3:

```bash
bash scripts/run_stage1_critiques.sh
bash scripts/run_stage2_rewrites.sh
bash scripts/run_stage3_validate.sh
```

Or directly:

```bash
python -m fg_pipeline.stage3.run_stage3 \
  --input  output/fghd/stage2/rewrites.jsonl \
  --output output/fghd/stage3/vote_records.jsonl \
  --preferences-out output/fghd/stage3/preference_pairs.jsonl \
  --stats-out output/fghd/stage3/stats.json
```

Useful flags: `--backend heuristic|gemini_openai_two_vote|gemini_two_vote|gemini_llava_two_vote`
(heuristic is smoke-only), `--limit N` for smoke runs, `--strict` to fail on
malformed Stage 2 rows, `--resume`, and `--checkpoint-every`.

For the recommended cross-vendor research backend:

```bash
BACKEND=gemini_openai_two_vote \
GEMINI_MODEL=gemini-2.5-flash-lite \
OPENAI_MODEL=gpt-4o-mini \
bash scripts/run_stage3_validate.sh
```

For a cross-family Gemini + LLaVA run:

```bash
BACKEND=gemini_llava_two_vote \
LLAVA_MODEL_PATH=models/llava-v1.5-13b \
LLAVA_DEVICE=cuda:0 \
bash scripts/run_stage3_validate.sh
```

Stage 3 writes:

- `output/fghd/stage3/vote_records.jsonl` — audit rows with verification votes per rewrite
- `output/fghd/stage3/preference_pairs.jsonl` — trainer-compatible preference pairs
- `output/fghd/stage3/stats.json` — compact validation counts

Rows approved by the selected backend flow directly into the final preference
set. Rejected rows are sent to Stage 4 repair.

Before launching Stage 4 repair on a long run, inspect:

- `output/fghd/stage3/stats.json`
- a small sample of `output/fghd/stage3/preference_pairs.jsonl`

The current smoke default is heuristic. With `GEMINI_API_KEY` or
`GOOGLE_API_KEY` plus `OPENAI_API_KEY` set, the launcher automatically selects
`gemini_openai_two_vote`.

### Stage 4 repair (LLaVA)

For the redesigned Stage 1-5 pipeline, run Stage 4 repair after Stage 3:

```bash
bash scripts/run_stage4_rewrite.sh
```

This repairs only Stage 3 rejected rewrites and writes:

- `output/fghd/stage4/repair_records.jsonl`
- `output/fghd/stage4/repair_preferences.jsonl`
- `output/fghd/stage4/final_preference_pairs.jsonl`

### Stage 5 training (severity-margin DPO)

Run Stage 5 after Stage 4 has written the final preference file:

```bash
bash scripts/run_stage5_train.sh
```

This points the trainer at `output/fghd/stage4/final_preference_pairs.jsonl`
and writes checkpoints under `output/fghd/stage5_llava_margin/`.
The default Stage 5 launcher follows the paper hyperparameters: 2 epochs,
LoRA rank 128, LoRA alpha 256, learning rate `2e-6`, total batch size 32,
DPO beta 0.1, and frozen projector. On a one-GPU box it reaches total batch
32 by using gradient accumulation.

The legacy `scripts/run_stage4_train.sh` wrapper still points the unchanged
trainer at `output/fghd/stage3/preference_pairs.jsonl`; use it only for the
older Stage 3-only HSA-DPO baseline path.

For a baseline-only reproduction run, keep using the released preference file
directly:

Run from the repo root on a multi-GPU Linux machine after the environment is bootstrapped and the base LLaVA model is at `models/llava-v1.5-13b`:

```bash
DATA_PATH=hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl \
IMAGE_FOLDER=hsa_dpo/data/images \
MODEL_PATH=models/llava-v1.5-13b \
OUTPUT_DIR=output/hsa_dpo_llava \
bash hsa_dpo_train.sh
```

Relevant knobs exposed by the script:

- `USE_CHOSEN_SCORE` (default `False`)
- `USE_REJECTED_SCORE` (default `True`; this is what reproduces the paper's severity-weighted rejected term)
- `DPO_LOSS_TYPE` (`hsa_weighted`, `severity_margin`, or `standard`)
- `SEVERITY_MARGIN_SCALE`, `SEVERITY_SCORE_NORMALIZER`
- `BATCH_SIZE`, `TOTAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `EPOCH`, `LEARNING_RATE`, `NUM_GPUS`

For a `1x RTX 6000 Ada (48 GB)` Vast instance, use the paper-fair Stage 5
wrapper after Stage 4 is complete:

```bash
bash scripts/run_stage5_train.sh
```

It defaults to `NUM_GPUS=1`, `BATCH_SIZE=1`, `TOTAL_BATCH_SIZE=32`, and
`EPOCH=2`. Use `bash scripts/vastai/run_pilot_train.sh` only for a short
environment sanity check, not for the final paper-comparison run.

Current data limitation:

- the released detection supervision does not expose the original user prompt
  separately
- the `question` field carried from Stage 1 through Stage 5 may therefore
  mirror the assessed candidate sentence rather than a distinct upstream prompt

## Evaluation Guide

The repo has three evaluation layers.

- `bash scripts/run_paper_eval.sh`
  Runs the strict paper-comparison wrapper and writes
  `output/eval/<run_name>/comparison/paper_core.{json,md}` plus `summary.csv`.
- `output/eval/<run_name>/comparison/supplemental_eval.{json,md}`
  Captures local or proxy rows that are intentionally excluded from the
  strict paper delta table.
- `bash scripts/run_general_eval.sh`
  Summarizes Stage 3 validation stats, Stage 5 trainer state (when present),
  and any selected public benchmark subset.

Evaluation is manifest-driven. The runner expects:

- one base `LLaVA-1.5-13B` row
- one local improved model row
- `model_base` when the model kind is `lora`

Strict paper comparison is local-only. It validates the manifest and requires:

- `temperature = 0.0`
- `num_beams = 1`
- `conv_mode = vicuna_v1`
- one shared `max_new_tokens` value across the manifest

The strict runner is only as fair as the benchmark adapter behind each row.
Keep strict paper comparison separate from supplemental rows, and do not treat
skipped local-judge benchmarks as missing paper deltas.
Run `bash scripts/vastai/install_eval_benchmarks.sh` on Vast to install the
evaluation extra and create the benchmark folder layout. Benchmark data itself
must still be placed under `playground/data/eval/` from the matching sources.
Supplemental judge benchmarks can use `OPENAI_JUDGE_MODEL` or
`--openai-judge-model`, but those rows remain outside the strict table unless
the adapter marks them paper-faithful.

## Vast AI Notes

- If `ssh` connects at the TCP level but hangs with `Connection timed out during
  banner exchange`, refresh the live SSH port with `vastai show instances` and
  retry. If the port is correct and the banner still never arrives, reboot the
  instance from Vast before assuming there is a repo-side failure.
- If you move to a different Vast template or GPU, do not edit the tracked
  launchers for one-off machine settings. Create
  `scripts/vastai/defaults.local.env` from
  `scripts/vastai/defaults.env.example` and keep the box-specific values there.
- On a fresh box, the safe bring-up order is:
  1. `bash scripts/vastai/bootstrap.sh`
  2. download `models/llava-v1.5-13b`
  3. `bash scripts/run_stage1_critiques.sh`
  4. `BACKEND=llava MODEL_PATH=models/llava-v1.5-13b bash scripts/run_stage2_rewrites.sh`
  5. `bash scripts/run_stage3_validate.sh`
  6. inspect `output/fghd/stage3/stats.json`
  7. `bash scripts/run_stage4_rewrite.sh`
  8. `bash scripts/vastai/run_pilot_train.sh` or `bash scripts/run_stage5_train.sh`
- Do not start a long Stage 5 run on a single 48 GB GPU before checking that
  Stage 2 rewrites, Stage 3 votes, and Stage 4 repairs are actually sane.

Supplemental rows are reported separately when a benchmark is proxy-only,
uses an unmatched local evaluator, or lacks a paper reference row.

## Immediate Team Focus

- Treat this local repo as the canonical development copy; Vast was only a run environment.
- Keep `hsa_dpo/` as the baseline layer.
- Keep Stage 3, Stage 4, and Stage 5 on top of the Stage 1/2 record contracts; do not reintroduce a confidence-based path.
