<!-- # magic-edit.github.io -->

<p align="center">
  <h2 align="center">[AAAI 2025] Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback
</h2>
  <p align="center">
    <a><strong>Wenyi Xiao<sup>1*</sup> , </strong></a>
    <a><strong>Ziwei Huang<sup>1*</sup> , </strong></a>
    <a><strong>Leilei Gan<sup>1†</sup> , </strong></a>
    <a><strong>Wanggui He<sup>2</sup>  </strong></a>
    <br>
    <a><strong>Haoyuan Li<sup>2</sup> ,  </strong></a>
    <a><strong>Zhelun Yu<sup>2</sup> , </strong></a>
    <a><strong>Fangxun Shu<sup>2</sup> ,  </strong></a>
    <a><strong>Hao Jiang<sup>2</sup> , </strong></a>
    <a><strong>Linchao Zhu<sup>1</sup>   </strong></a>
    <br>
    <sup>1</sup> Zhejiang University&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup> Alibaba Group&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp
    <br>
    <sup>*</sup>Equal contribution &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp <sup>†</sup>Corresponding author
    </br>
    </br>
        <a href="https://arxiv.org/pdf/2404.14233">
        <img src='https://img.shields.io/badge/Paper-Arxiv-orange' alt='Paper PDF'></a>
        <a href="https://huggingface.co/datasets/WenyiXiao/HSA-DPO">
        <img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow' alt='Dataset'></a>
        <a href="https://modelscope.cn/models/xiaowenyi/HSA-DPO">
        <img src='https://img.shields.io/badge/Model-ModelScope-blue' alt='Dataset'></a>

  </p>
</p>



## Overview

This repository contains the official implementation of the paper "Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback". For supplementary experiments and details, please refer to the [Appendix](asset/HSA_DPO_Appendix.pdf).

This working copy has two layers:

- the original `hsa_dpo/` package and training flow from the released paper
- an `fg_pipeline/` extension layer that hosts the new pipeline stages as they come online, along with shared utilities and the paper/general evaluation tooling

Project status:

- the project method has been redesigned around five stages: (1) critique detection / extraction, (2) critique-guided rewrite, (3) preference validation, (4) LLaVA repair of rejected rewrites, and (5) severity-margin DPO
- Stage 1 is implemented under `fg_pipeline/stage1/`; the default backend (`ReleasedAnnotationBackend`) parses the released `hsa_dpo_detection.jsonl` supervision into a normalized `Stage1Record` without any model inference, and the local research path now includes a `LlavaDetectorBackend` plus detector dataset-prep / train / benchmark-export entrypoints
- Stage 2 is implemented under `fg_pipeline/stage2/`; it consumes Stage 1 JSONL and emits one corrected rewrite per hallucinated record; the default `TemplateRewriteBackend` is smoke-only and deterministic; the intended research path is `LlavaRewriteBackend` using the vendored LLaVA-v1.5 stack
- Stage 3 is implemented under `fg_pipeline/stage3/`; it runs verification votes per rewrite, writes an audit JSONL, and exports initially approved preference pairs; the default `HeuristicVerificationBackend` is deterministic and smoke-oriented, while research runs use `gemini_openai_two_vote`, `gemini_two_vote`, or `gemini_llava_two_vote`
- Stage 4 is implemented under `fg_pipeline/stage4/`; it repairs Stage 3 rejected rewrites with LLaVA and writes the final combined preference dataset
- Stage 5 trains LLaVA with severity-margin DPO through `scripts/run_stage5_train.sh`; the legacy HSA-DPO wrapper remains available as `scripts/run_stage4_train.sh`
- the earlier confidence-based Stage 3-5 implementation remains fully removed and the new design does not reintroduce any confidence / calibration / threshold logic

Design rule for this project:

- keep the original HSA-DPO code and dataset paths intact
- reuse `hsa_dpo_train.sh` and `hsa_dpo/models/llava-v1_5/train_dpo.py` whenever possible
- add new logic around the old pipeline instead of replacing it

![model](asset/overview.png)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Vast AI](#vast-ai)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/Mr-Loevan/HSA-DPO.git
cd HSA-DPO

# Install HSA-DPO and dependencies
conda create -n hsa_dpo python==3.10
conda activate hsa_dpo
pip install -e .

# Linux GPU training stack
pip install -e ".[linux-train]"
```

## Vast AI

For a repo-specific Vast AI workflow, see [VAST_AI_SETUP.md](VAST_AI_SETUP.md).

## Dataset

### Download Dataset
```bash
pip install -U huggingface_hub

# Download all dataset files
hf download --repo-type dataset WenyiXiao/HSA-DPO --local-dir ./datasets
```

### Dataset Organization

**For hallucination detection:**
- Training data: `hsa_dpo_detection.jsonl`
- Images: from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html), stored under `./vg/images`

**For hallucination mitigation (HSA-DPO training):**
- Preference data: `hsa_dpo_preference_llava1dot5.jsonl`
- Images: extracted into `./hsa_dpo/data/images`

**Optional external data (not used by default):**
- `VLFeedback/` is not part of the default repo flow
- keep it outside the main pipeline unless you explicitly decide to use it as auxiliary preference data

### Prepare Data for Training

```bash
# 1. Create data directories
mkdir -p hsa_dpo/data
mkdir -p hsa_dpo/data/images
mkdir -p vg/images

# 2. Copy preference dataset
cp datasets/hsa_dpo_preference_llava1dot5.jsonl hsa_dpo/data/

# 3. Extract images
tar -xzf datasets/hsa_dpo_imgs.tar.gz -C hsa_dpo/data/images/

# 4. Verify data structure
ls hsa_dpo/data/
# Should show: hsa_dpo_preference_llava1dot5.jsonl and images/

ls hsa_dpo/data/images/ | head -5
# Should show: 0.jpg, 1.jpg, 2.jpg, 3.jpg, 4.jpg ...
```

**Note:** The images are named with sequential IDs (0.jpg, 1.jpg, ...) corresponding to the `id` field in the JSONL file.

**Important:** the repo uses two different image stores:

- `vg/images/` for the detection dataset
- `hsa_dpo/data/images/` for preference training

Do not merge them.

## Training

### Prerequisites

1. Install the HSA-DPO package:
```bash
pip install -e .
```

2. Prepare dataset following the instructions above (see [Dataset](#dataset) section)

3. Download the base LLaVA-v1.5 model:
```bash
# Download LLaVA-v1.5-13B model
hf download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b

# The CLIP vision encoder will be auto-downloaded during training
```

### Running Training

We provide a training script for HSA-DPO with LLaVA-v1.5:

```bash
# Configure paths in hsa_dpo_train.sh
vim hsa_dpo_train.sh

# Update these paths according to your setup:
# DATA_PATH="./hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl"
# IMAGE_FOLDER="./hsa_dpo/data/images"
# MODEL_PATH="path/to/llava-v1.5-13b"
# OUTPUT_DIR="./output/hsa_dpo_llava"

# Run training
bash hsa_dpo_train.sh
```

## Project Extension Pipeline

The project method is being rebuilt around five stages: (1) critique detection / extraction, (2) critique-guided rewrite, (3) preference validation, (4) LLaVA repair of rejected rewrites, (5) severity-margin DPO. The earlier confidence-based approach (confidence scoring, calibration, thresholding, and CRC / CV-CRC selection) remains removed and is not part of the new design.

What currently lives under `fg_pipeline/`:

- `fg_pipeline/stage1/` — Stage 1 critique detection / extraction (parser, local detector backend, detector data prep / export CLIs)
- `fg_pipeline/stage2/` — Stage 2 critique-guided rewrite (prompt template, smoke backend, LLaVA backend seam, CLI)
- `fg_pipeline/stage3/` — Stage 3 preference verification (vote schema, heuristic backend, Gemini/Gemini+LLaVA backends, CLI)
- `fg_pipeline/stage4/` — Stage 4 repair pass for Stage 3 rejected rewrites and final preference construction
- `fg_pipeline/io_utils.py`, `fg_pipeline/paths.py`, `fg_pipeline/schemas.py` — shared utilities
- `fg_pipeline/eval/` — strict paper-comparison and supplemental local evaluation tooling
- `fg_pipeline/data/` — curated data fixtures (Stage 1 supervision mirror, smoke fixture, paper reference tables)

Stage 1 launcher:

```bash
bash scripts/run_stage1_critiques.sh
# or:
python -m fg_pipeline.stage1.run_stage1 --help
```

Stage 1 is CPU-friendly and does not require model inference. Output goes to `output/fghd/stage1/detection_critiques.jsonl` with a compact `stats.json` alongside.

Detector research path helpers:

```bash
bash scripts/run_stage1_detector_dataset.sh
bash scripts/run_stage1_detector_train.sh
bash scripts/run_stage1_export_benchmarks.sh
```

In the released Stage 1 source rows, the sentence after `Description to Assess:`
is the assessed candidate response. The normalized Stage 1 output therefore
stores that sentence in `response_text`, while the raw GPT annotation payload
is preserved in `metadata.raw_annotation_text`.

Stage 2 launcher:

```bash
bash scripts/run_stage2_rewrites.sh
# or:
python -m fg_pipeline.stage2.run_stage2 --help
```

Stage 2 skips non-hallucinated Stage 1 rows and writes one rewrite per
hallucinated row to `output/fghd/stage2/rewrites.jsonl`.

Stage 3 launcher:

```bash
bash scripts/run_stage3_validate.sh
# or:
python -m fg_pipeline.stage3.run_stage3 --help
```

Stage 3 consumes Stage 2 rewrites, runs verification votes per row, writes an audit JSONL to
`output/fghd/stage3/vote_records.jsonl`, and writes Stage 4-compatible
preference pairs to `output/fghd/stage3/preference_pairs.jsonl`.

Research Stage 3 backend:

```bash
BACKEND=gemini_openai_two_vote \
GEMINI_MODEL=gemini-2.5-flash-lite \
OPENAI_MODEL=gpt-4o-mini \
bash scripts/run_stage3_validate.sh
```

With `GEMINI_API_KEY` or `GOOGLE_API_KEY` plus `OPENAI_API_KEY` set, the Stage 3 launcher
automatically switches from the smoke `heuristic` backend to `gemini_openai_two_vote`.
Use `BACKEND=gemini_llava_two_vote LLAVA_MODEL_PATH=models/llava-v1.5-13b` if
you want one hosted Gemini vote plus one local LLaVA vote.

Current limitation: the released detection data does not expose the original
user prompt separately, so the `question` field passed through Stages 1-3 may
mirror the assessed candidate sentence rather than an upstream prompt.

Evaluation launchers:

```bash
bash scripts/run_paper_eval.sh
bash scripts/run_general_eval.sh
python -m fg_pipeline.eval.run_eval --help
```

Stage 4 repair launcher:

```bash
bash scripts/run_stage4_rewrite.sh
# or:
python -m fg_pipeline.stage4.run_stage4_repair --help
```

Stage 4 consumes `output/fghd/stage3/vote_records.jsonl` plus
`output/fghd/stage3/preference_pairs.jsonl`, repairs rejected rows, and writes
the final training set to `output/fghd/stage4/final_preference_pairs.jsonl`.

### Training (Stage 5 — severity-margin DPO)

For the redesigned Stage 1-5 pipeline, use:

```bash
bash scripts/run_stage5_train.sh
```

This wrapper points `DATA_PATH` at
`output/fghd/stage4/final_preference_pairs.jsonl`, sets `OUTPUT_DIR` to
`output/fghd/stage5_llava_margin`, and trains with `DPO_LOSS_TYPE=severity_margin`.

### Legacy Training (Stage 3-only HSA-DPO)

The legacy wrapper keeps the released HSA-DPO trainer path available:
`hsa_dpo_train.sh` → `hsa_dpo/models/llava-v1_5/train_dpo.py` →
`hsa_dpo.trainer.LlavaDPOTrainer`.

For the older Stage 3-only training path, use:

```bash
bash scripts/run_stage4_train.sh
```

This wrapper points `DATA_PATH` at
`output/fghd/stage3/preference_pairs.jsonl`, sets `OUTPUT_DIR` to
`output/fghd/stage4_llava`, and then delegates to `hsa_dpo_train.sh`.

For a baseline-paper reproduction run, keep using the released preference file
directly:

```bash
DATA_PATH=hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl \
IMAGE_FOLDER=hsa_dpo/data/images \
MODEL_PATH=models/llava-v1.5-13b \
OUTPUT_DIR=output/hsa_dpo_llava \
bash hsa_dpo_train.sh
```

### Key Parameters

- `use_chosen_score`: Whether to use chosen scores in DPO loss (default: False)
- `use_rejected_score`: Whether to use rejected scores in DPO loss (default: True — reproduces the HSA-DPO severity-weighted rejected term)
- `dpo_loss_type`: `hsa_weighted`, `severity_margin`, or `standard`
- `severity_margin_scale`: Margin multiplier for Stage 5 severity-margin DPO
- `beta`: Temperature parameter for DPO loss (default: 0.1)
- `num_train_epochs`: Number of training epochs (default: 2)
- `per_device_train_batch_size`: Batch size per GPU (default: 8)
- `learning_rate`: Learning rate (default: 2e-6)

### Multi-GPU Training

The script supports multi-GPU training with DeepSpeed. Adjust `NUM_GPUS` in the script:

```bash
NUM_GPUS=2  # Use 2 GPUs
bash hsa_dpo_train.sh
```

## Evaluation

### Download Model Weights

```bash
pip install -U modelscope
modelscope download --model xiaowenyi/HSA-DPO --local-dir ./checkpoints
```

### Evaluation Suite

The repo now has a project-owned evaluation layer under `fg_pipeline/eval/`.

It now supports three report layers:

- **Strict paper comparison**: only rows that are honestly comparable to the referenced paper
- **Supplemental local evaluation**: local or proxy metrics that should not appear in the strict delta table
- **General eval**: Stage 3 / Stage 4 runtime summaries plus any selected public benchmark subset

The intended 3-way comparison is:

- base `LLaVA-1.5-13B`
- your local improved model
- the referenced paper’s reported numbers as a fixed overlay

OpenAI is out of scope for the default workflow in this repo. The strict
paper-comparison path is fully local-only. Supplemental local-judge benchmarks
remain separate from the strict comparison table.

### Model Manifest

The evaluation runner expects a JSON manifest:

```json
[
  {
    "model_id": "llava15_base_13b",
    "model_path": "models/llava-v1.5-13b",
    "model_base": null,
    "kind": "base",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": 512
  },
  {
    "model_id": "ours_stage4_lora",
    "model_path": "output/fghd/stage4_llava",
    "model_base": "models/llava-v1.5-13b",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": 512
  }
]
```

Supported model kinds:

- `base`
- `lora`
- `merged`

### Strict vs Supplemental Benchmarks

Strict paper-comparison defaults:

- `mhalubench`
- `mfhallubench`
- `pope_adv`

Supplemental local defaults:

- `object_halbench`
- `amber`
- `mmhal_bench`
- `llava_bench_wild`
- `hss`

Current implementation shape:

- `mhalubench` and `mfhallubench` are detection-side adapters that now evaluate per-model detector exports rather than hard-coded `stage3_detector` IDs
- `pope_adv` is included in the strict table only when decode settings match the fair-comparison contract
- `object_halbench`, `amber`, `mmhal_bench`, `llava_bench_wild`, and `hss` are rendered as supplemental unless they are made fully paper-faithful

### Dataset Prerequisites

Benchmark assets are not bundled in this repo. Prepare them separately.

Typical required assets:

- `POPE Adv.`: question file plus image directory
- `LLaVA-Bench-in-the-Wild`: `questions.jsonl`, `context.jsonl`, images, optional reference answers
- `MMHal-Bench`: question file plus images
- `Object HalBench`: normalized `questions.jsonl`, `annotations.jsonl`, images
- `AMBER`: normalized generative split plus images
- `MHaluBench`, `MFHaluBench`: prepared prediction/annotation files for the Stage 1 detector

If a dataset is missing:

- the runner fails clearly by default
- use `--skip-missing-datasets` to skip it and still render the report

### Run Strict Paper Comparison

```bash
MODEL_MANIFEST=path/to/models.eval.json \
bash scripts/run_paper_eval.sh
```

Direct CLI form:

```bash
python -m fg_pipeline.eval.run_eval \
  --run-name paper_core \
  --models-json path/to/models.eval.json \
  --benchmarks mhalubench,mfhallubench,pope_adv \
  --paper-core
```

Strict paper comparison validates the manifest before running:

- `temperature = 0.0`
- `num_beams = 1`
- `conv_mode = vicuna_v1`
- one shared `max_new_tokens` value across the manifest

### Run Supplemental / General Evaluation

```bash
MODEL_MANIFEST=path/to/models.eval.json \
bash scripts/run_general_eval.sh
```

The general runner summarizes:

- Stage 3 validation stats if available
- Stage 4 trainer state if available (under `output/fghd/stage4_llava/` or `output/hsa_dpo_llava/`)
- any selected public benchmark subset

### Output Layout

```text
output/eval/<run_name>/
├── models/<model_id>/predictions/
├── models/<model_id>/metrics/
├── models/<model_id>/judges/
└── comparison/
    ├── paper_core.json
    ├── paper_core.md
    ├── supplemental_eval.json
    ├── supplemental_eval.md
    ├── general_eval.json
    ├── general_eval.md
    └── summary.csv
```

The reports explicitly separate:

- paper reference values
- locally reproduced values
- local proxy values that are useful for research iteration but not yet strictly paper-comparable

### Run Inference

We provide a simple inference script to test the model:

```bash
# Run inference (LLaVA should already be installed from Installation step)
python inference/inference_example.py \
    --model-base path/to/llava-v1.5-13b \
    --lora-path ./output/hsa_dpo_llava \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

## Citation

If you find this work useful, we would appreciate it if you could cite our paper:

```bibtex
@article{xiao2025hsa_dpo,
  title     = {Detecting and Mitigating Hallucination in Large Vision Language Models
               via Fine-Grained AI Feedback},
  author    = {Xiao, Wenyi and Huang, Ziwei and Gan, Leilei and He, Wanggui and
               Li, Haoyuan and Yu, Zhelun and Shu, Fangxun and Jiang, Hao and
               Zhu, Linchao},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {24},
  pages     = {25543--25551},
  year      = {2025},
  month     = {Apr},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/34744},
  doi       = {10.1609/aaai.v39i24.34744}
}
```
