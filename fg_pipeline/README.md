# FG Pipeline

This package is an extension layer on top of the original HSA-DPO repository.

The project method has five stages:

1. Critique detection / extraction (Stage 1 — implemented)
2. Critique-guided rewrite (Stage 2 — implemented)
3. Majority-vote preference validation (Stage 3 — implemented)
4. LLaVA repair of rejected rewrites (Stage 4 — implemented)
5. Severity-margin DPO (Stage 5 — wrapper over `hsa_dpo/`)

Design rule:

- reuse the original `hsa_dpo/` code and datasets wherever possible
- avoid moving or renaming the original training/data paths
- keep Stage 1 and Stage 2 output schemas stable so downstream stages can
  consume them without coupling to a specific backend

## What lives in this package

- `fg_pipeline/stage1/` — Stage 1 critique detection / extraction
  - `schemas.py` — `Stage1Record`, `CritiqueItem`, severity mapping
  - `parser.py` — permissive parser for the released annotation format
  - `prompts.py` — local detector prompt template
  - `backends.py` — `CritiqueDetectorBackend` protocol plus
    `ReleasedAnnotationBackend` and `LlavaDetectorBackend`
  - `run_stage1.py` — parser / detector inference CLI
  - `run_stage1_detector_dataset.py` — detector SFT data prep
  - `run_stage1_export_benchmarks.py` — detector benchmark export
- `fg_pipeline/stage2/` — Stage 2 critique-guided rewrite
  - `schemas.py` — `Stage2Record`
  - `prompts.py` — rewrite prompt template + `PROMPT_VERSION` constant
  - `backends.py` — `RewriteBackend` protocol, `TemplateRewriteBackend`
    (smoke), `LlavaRewriteBackend` (real research backend)
  - `run_stage2.py` — CLI (`python -m fg_pipeline.stage2.run_stage2`)
- `fg_pipeline/stage3/` — Stage 3 majority-vote preference validation
  - `schemas.py` — `Stage3Record`, `VoteDecision`
  - `prompts.py` — local judge prompt template
  - `backends.py` — `VerificationBackend` protocol plus
    `HeuristicVerificationBackend`, `GeminiTwoVoteBackend`, and
    `GeminiLlavaTwoVoteBackend`
  - `run_stage3.py` — CLI (`python -m fg_pipeline.stage3.run_stage3`)
- `fg_pipeline/stage4/` — Stage 4 repair and final preference construction
  - `prompts.py` — repair prompt using Stage 1 critiques and Stage 3 vote feedback
  - `schemas.py` — `Stage4RepairRecord`
  - `run_stage4_repair.py` — CLI (`python -m fg_pipeline.stage4.run_stage4_repair`)
- `fg_pipeline/io_utils.py` — JSONL read/write helpers
- `fg_pipeline/paths.py` — extension-layer default paths (Stages 1-4)
- `fg_pipeline/schemas.py` — shared records re-exported at the package top level
- `fg_pipeline/eval/` — paper and general evaluation tooling (benchmarks,
  judges, reporting)
- `fg_pipeline/data/` — curated data fixtures (detection annotations,
  smoke fixture, paper reference tables)

## Running Stage 1

Stage 1 is CPU-friendly by default and does not require model inference.

```bash
bash scripts/run_stage1_critiques.sh
```

or directly:

```bash
python -m fg_pipeline.stage1.run_stage1 \
  --input  fg_pipeline/data/hsa_dpo_detection.jsonl \
  --output output/fghd/stage1/detection_critiques.jsonl \
  --stats-out output/fghd/stage1/stats.json
```

Flags: `--backend released_annotations` (default), `--limit N` (smoke runs),
`--strict` (fail on malformed rows).

Detector research helpers:

```bash
bash scripts/run_stage1_detector_dataset.sh
bash scripts/run_stage1_detector_train.sh
bash scripts/run_stage1_export_benchmarks.sh
```

Stage 1 output record (`Stage1Record`):

- `id`, `image`, `question`, `response_text`, `is_hallucinated`
- `critiques[]` — each with `index`, `hallucination_type`,
  `severity_label`, `severity_score` (`1|2|3` or `null`), `rationale`,
  `evidence_text`, `source_tag_text`, `source_score_text`
- `metadata` — `source`, `raw_annotation_text`, plus `parse_warnings`
  when recoverable parse issues were encountered

## Running Stage 2

Stage 2 consumes Stage 1 output. Run Stage 1 first, then Stage 2.

The default backend (`template`) is a deterministic smoke backend — it is
**not research-quality**. It removes identified evidence spans from the
original text and is useful for local testing and pipeline validation.
The intended research backend is `llava`, which uses the vendored
LLaVA-v1.5 stack and requires a local model path.

```bash
bash scripts/run_stage2_rewrites.sh
```

or directly:

```bash
python -m fg_pipeline.stage2.run_stage2 \
  --input  output/fghd/stage1/detection_critiques.jsonl \
  --output output/fghd/stage2/rewrites.jsonl \
  --stats-out output/fghd/stage2/stats.json
```

With the LLaVA backend (GPU required):

```bash
python -m fg_pipeline.stage2.run_stage2 \
  --backend llava \
  --model-path models/llava-v1.5-7b \
  --output output/fghd/stage2/rewrites_llava.jsonl \
  --stats-out output/fghd/stage2/stats_llava.json
```

Flags: `--backend template|llava`, `--limit N` (smoke runs), `--strict`
(fail on empty rewrites), `--model-path` (LLaVA), `--model-base` (LoRA),
`--image-root` (root for resolving relative image paths).

Stage 2 skips non-hallucinated rows. Output record (`Stage2Record`):

- `id`, `image`, `question`
- `original_response` — the original candidate sentence from Stage 1
- `rewrite_response` — the single corrected rewrite
- `critiques[]` — pass-through from Stage 1
- `metadata` — `source_stage`, `backend`, `prompt_version`

There are no confidence / calibration / threshold fields on Stage 1 or
Stage 2 output, by design.

## Running Stage 3

Stage 3 consumes Stage 2 output and decides whether the rewrite is good enough
to become a training pair.

The default backend (`heuristic`) is deterministic and smoke-only. Research
runs use `gemini_openai_two_vote` for cross-vendor hosted validation,
`gemini_two_vote` for the fastest Gemini-only path, or `gemini_llava_two_vote`
when you need one hosted Gemini vote plus one local LLaVA vote.

```bash
bash scripts/run_stage3_validate.sh
```

or directly:

```bash
python -m fg_pipeline.stage3.run_stage3 \
  --input  output/fghd/stage2/rewrites.jsonl \
  --output output/fghd/stage3/vote_records.jsonl \
  --preferences-out output/fghd/stage3/preference_pairs.jsonl \
  --stats-out output/fghd/stage3/stats.json
```

Flags: `--backend heuristic|gemini_openai_two_vote|gemini_two_vote|gemini_llava_two_vote`, `--limit N`
(smoke runs), `--strict` (fail on malformed Stage 2 rows or backend errors),
`--llava-model-path`, `--llava-model-base`, `--gemini-model`, `--openai-model`,
`--image-root`, `--resume`, and `--checkpoint-every`.

Stage 3 output:

- `vote_records.jsonl` — one audit row per hallucinated Stage 2 input,
  including the verification votes
- `preference_pairs.jsonl` — only pairs approved by the selected backend
- `stats.json` — compact counts and backend metadata

There are no confidence / calibration / threshold fields on Stages 1-3
output, by design.

## Running Stage 4

Stage 4 repairs only the rows that Stage 3 rejected. It then combines the
original Stage 3-approved pairs with the repaired pairs into the final training
dataset.

```bash
bash scripts/run_stage4_rewrite.sh
```

or directly:

```bash
python -m fg_pipeline.stage4.run_stage4_repair \
  --input output/fghd/stage3/vote_records.jsonl \
  --stage3-preferences output/fghd/stage3/preference_pairs.jsonl \
  --final-preferences-out output/fghd/stage4/final_preference_pairs.jsonl
```

For smoke tests use `--backend template`. For research runs use
`--backend llava --model-path models/llava-v1.5-7b`.

## Running Stage 5

Stage 5 trains on `output/fghd/stage4/final_preference_pairs.jsonl` with the
new `severity_margin` DPO loss:

```bash
bash scripts/run_stage5_train.sh
```

Use `scripts/run_stage4_train.sh` only for the older Stage 3-only HSA-DPO path.

## Downstream compatibility

Stage 5 still consumes the original HSA-DPO preference
format via `hsa_dpo/models/llava-v1_5/train_dpo.py`. The target schema
(`PreferenceCleanRecord`, re-exported from `fg_pipeline`) is:

- `id`
- `question`
- `chosen`
- `rejected`
- `chosen_score`
- `rejected_score`
- `image` (optional; explicit path preferred over the legacy
  `<image_folder>/<id>.jpg` fallback)

Stage 3 and Stage 4 both emit this schema. For the redesigned Stage 1-5 path,
use `bash scripts/run_stage5_train.sh`, which points the trainer at
`output/fghd/stage4/final_preference_pairs.jsonl`.

Current limitation: the released detection supervision does not expose the
original user prompt separately, so the `question` field carried into
`PreferenceCleanRecord` may mirror the assessed candidate sentence from
Stage 1 instead of a distinct upstream prompt.
