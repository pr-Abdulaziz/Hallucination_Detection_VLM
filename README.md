# Fine-Grained Hallucination Detection and Severity-Aware Mitigation in Vision-Language Models

![Fine-grained hallucination mitigation pipeline](asset/diagram_overview.png)

## Project Metadata

**Team:** Fine-Grained Hallucination Mitigation  
**Members:** Abdulaziz Alqahtani and Alwaleed Alharthi  
**Supervisor:** Dr. Muzammil Behzad  
**Affiliation:** King Fahd University of Petroleum and Minerals (KFUPM)

## Overview

Large vision-language models (LVLMs) can generate fluent image-grounded answers while still inventing unsupported objects, attributes, relations, or scene details. This project studies hallucination mitigation using fine-grained critique signals and severity-aware preference optimization.

The implementation starts from released fine-grained supervision and preference data, converts hallucination annotations into structured critiques, builds preference pairs through direct and judged paths, and trains LLaVA-style LVLMs with DPO and HSA-DPO variants. The project also includes automatic evaluation on hallucination-focused benchmarks and paper/report assets for presentation.

## Problem Definition

A model response can be partly correct and partly hallucinated. Treating the whole response as simply good or bad loses important information about where the hallucination occurs and how severe it is.

Given an image-question or image-instruction input `x` and an original response `y_hat`, the goal is to produce or select a better response `y` that is more visually grounded while preserving useful supported details. Severe hallucinations should receive stronger correction pressure than minor wording issues.

The project focuses on three practical questions:

1. Can released fine-grained hallucination supervision be normalized into usable critique records?
2. Can direct released preference pairs and judged preference pairs produce different alignment behavior?
3. Does severity-aware DPO improve hallucination behavior compared with normal DPO and the base LVLM?

## Methodology

The implemented pipeline has four main stages.

### 1. Fine-Grained Critique Extraction

Released hallucination annotations are parsed into structured records. Each row stores the original response, hallucination status, critique items, hallucination type, rationale, evidence, and severity information.

Output:

```text
output/fghd/paper_stage1/d_faif.jsonl
output/fghd/paper_stage1/stats.json
```

### 2. Critique-Guided Rewrite and Preference Construction

Hallucinated examples are rewritten with LLaVA using the extracted critiques. The original response becomes the rejected response, and the rewritten response becomes the chosen response. Severity scores are kept with the pair.

Preference row structure:

```text
{
  "image": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "severity": ...
}
```

### 3. Two Preference Paths

The project uses two preference-data paths:

**Direct preference path:** released preference pairs are used directly for DPO/HSA-DPO training.

**Judged preference path:** released preference pairs are checked using hosted LVLM/LLM judge models such as Gemini and GPT. The code supports two judge prompting modes: `zero_shot`, which is the paper baseline already used in the reported experiment, and `two_shot`, where two calibration examples are shown before each target pair. Accepted pairs continue to training. Rejected pairs can be repaired with LLaVA and merged back into a checked preference set.

The default judge and repair models are kept the same across zero-shot and two-shot runs for a fair comparison:

| Component | Default model |
| --- | --- |
| Gemini judge | `gemini-2.5-flash-lite` |
| OpenAI judge | `gpt-4o-mini` |
| Local repair model | `models/llava-v1.5-7b` |

Important output locations:

| Mode | Stage 3 output | Stage 4 output |
| --- | --- | --- |
| `zero_shot` | `output/fghd/released_pref_stage3/` | `output/fghd/released_pref_stage4/` |
| `two_shot` | `output/fghd/released_pref_stage3_2shot_experiment/` | `output/fghd/released_pref_stage4_2shot_experiment/` |

API keys are never stored in this repository. Put local credentials in `.env` or environment variables only.

### 4. Severity-Aware Alignment

The final training stage compares several preference-optimization variants:

| Variant | Description |
| --- | --- |
| Base | Unaligned LLaVA baseline |
| DPO | Standard direct preference optimization |
| DPO-HSA | HSA-DPO style severity-aware weighting |
| HSA-DPO-M | Judged-path severity-margin HSA-DPO variant |

Training uses LoRA over a LLaVA base model. The key severity signal is a response-level score derived from fine-grained hallucination severities.

Typical Stage 5 outputs:

```text
output/fghd/exp_direct_paper_hsa_b32_e1/
output/fghd/exp_direct_normal_dpo_b32_e1/
output/fghd/released_pref_stage5_hsa_dpo/
output/fghd/released_pref_stage5_2shot_experiment/
```

## Repository Structure

```text
asset/
  diagram_overview.png            # project methodology diagram
  paper/                          # LaTeX paper, references, paper figures
fg_pipeline/
  paper/                          # project pipeline modules
  eval/                           # evaluation and reporting utilities
hsa_dpo/
  models/                         # adapted HSA-DPO/LLaVA training code
  data/                           # local data layout, ignored if large
inference/
  inference_example.py            # local inference helper
notebooks/
  results_exploration.ipynb       # compact result visualization notebook
asset/result_figures/             # exported benchmark figures
scripts/
  run_paper_stage1_faif.sh
  run_paper_stage2_detector_dataset.sh
  run_released_pref_stage3_validate.sh
  run_released_pref_stage4_repair.sh
  run_paper_stage5_train_hsa.sh
  run_stage5_eval_auto.sh
tests/
  test_paper_pipeline.py
```

Large model files, raw image folders, checkpoints, and temporary training artifacts should not be committed.

## Setup

Create the Python environment:

```bash
conda create -n hsa_dpo python=3.10
conda activate hsa_dpo
pip install -e .
pip install -e ".[linux-train]"
```

Install Hugging Face tooling if datasets or models need to be downloaded:

```bash
pip install -U huggingface_hub
```

Expected local model and data folders:

```text
models/llava-v1.5-7b/
hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl
hsa_dpo/data/images/
vg/images/
```

The `vg/` folder is used only when running detection-side data paths that reference Visual Genome images. It is not required for every training or evaluation command.

## Running the Pipeline

### Stage 1: parse released fine-grained supervision

```bash
bash scripts/run_paper_stage1_faif.sh
```

### Stage 2: build detector-style data

```bash
bash scripts/run_paper_stage2_detector_dataset.sh
```

### Judged preference path

Run zero-shot released-preference validation:

```bash
SHOT_MODE=zero_shot API_JUDGE=gemini_openai bash scripts/run_released_pref_stage3_validate.sh
```

Run 2-shot released-preference validation:

```bash
SHOT_MODE=two_shot API_JUDGE=gemini_openai bash scripts/run_released_pref_stage3_validate.sh
```

Use `WORKERS=3` for the faster 2-shot run while keeping API pressure moderate:

```bash
SHOT_MODE=two_shot WORKERS=3 API_JUDGE=gemini_openai bash scripts/run_released_pref_stage3_validate.sh
```

This writes the new 2-shot preference files to:

```text
output/fghd/released_pref_stage3_2shot_experiment/
```

Repair rejected rows with LLaVA using the same mode:

```bash
SHOT_MODE=two_shot bash scripts/run_released_pref_stage4_repair.sh
```

Verify the repaired 2-shot rows with the same repair-verification model used in the main experiment:

```bash
SHOT_MODE=two_shot OPENAI_MODEL=gpt-4.1-mini bash scripts/run_released_pref_stage5_openai_verify.sh
```

The final 2-shot verified preference file is saved separately at:

```text
output/fghd/released_pref_stage5_openai_verify_2shot_experiment/final_verified_preference_pairs.jsonl
```

Run the full path by mode:

```bash
SHOT_MODE=zero_shot bash scripts/run_released_pref_pipeline.sh
SHOT_MODE=two_shot bash scripts/run_released_pref_pipeline.sh
```

Convenience wrappers are also available:

```bash
bash scripts/run_released_pref_zero_shot_pipeline.sh
bash scripts/run_released_pref_2shot_pipeline.sh
```

### Direct training path

Run direct HSA-DPO:

```bash
bash scripts/run_direct_stage5_paper_hsa_batch32_epoch1.sh
```

Run matched normal DPO:

```bash
bash scripts/run_direct_stage5_normal_dpo_batch32_epoch1.sh
```

Queue normal DPO after direct HSA-DPO:

```bash
bash scripts/queue_direct_normal_dpo_after_paper_b32_e1.sh
```

### Custom Stage 5 training

```bash
DATA_PATH=hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl \
IMAGE_FOLDER=hsa_dpo/data/images \
OUTPUT_DIR=output/fghd/custom_hsa_dpo \
DPO_LOSS_TYPE=hsa_weighted \
USE_REJECTED_SCORE=True \
USE_CHOSEN_SCORE=False \
bash scripts/run_paper_stage5_train_hsa.sh
```

## Evaluation

The automatic evaluation runner compares the base model and trained LoRA checkpoints on automatic hallucination benchmarks.

```bash
bash scripts/setup_stage5_eval_assets.sh
bash scripts/run_stage5_eval_auto.sh
```

The evaluation setup is automatic-metric based and does not require OpenAI or Gemini judge calls. It supports:

| Benchmark | Main reported metric |
| --- | --- |
| POPE adversarial | F1, accuracy, precision, recall |
| Object HalBench | CHAIRS, CHAIRI |
| AMBER | CHAIR, Cover, Hal, Cog |

Result figures are generated from:

```text
notebooks/results_exploration.ipynb
```

Exported figures are saved under:

```text
asset/result_figures/
```

## Current Result Snapshot

The latest summarized local results are:

| Model variant | POPE Adv. F1 higher is better | Object HalBench CHAIRS lower is better | AMBER CHAIR lower is better | AMBER Hal lower is better |
| --- | ---: | ---: | ---: | ---: |
| Base LLaVA | 84.18 | 53.00 | 7.70 | 35.90 |
| DPO-HSA | 83.91 | 38.00 | 5.50 | 25.10 |
| DPO | 83.88 | 37.67 | 5.20 | 25.20 |
| HSA-DPO-M | 83.87 | 36.00 | 5.30 | 25.50 |

The trained models improve strongly on Object HalBench and AMBER hallucination metrics, while POPE Adv. F1 remains slightly higher for the base model. Therefore, the results should be reported as benchmark-dependent rather than as a universal improvement claim.

## Paper and Presentation Assets

Important writing and presentation files:

```text
asset/paper/main.tex
asset/paper/main.pdf
asset/paper/references.bib
asset/paper/diagram_prompt_paperbanana.md
asset/diagram_overview.png
asset/result_figures/
asset/Fine-Grained Hallucination Detection and Severity-Aware Mitigation in Vision-Language.pptx
```

The paper includes methodology, prompts in the appendix, benchmark results, discussion, limitations, and conclusion. The presentation file provides the slide deck for the project overview, methodology, experiments, and results.

## Git and Storage Notes

The repository is intended to track code, reports, notebooks, diagrams, paper files, and compact result summaries.

Do not commit:

- local API keys or `.env`
- raw model directories under `models/`
- Visual Genome image folders under `vg/`
- large checkpoints
- temporary training caches

If full training artifacts are needed for backup, store them externally through cloud storage or a dedicated model artifact store rather than Git.

## Acknowledgments

This project builds on the released HSA-DPO method, LLaVA training ecosystem, Hugging Face tooling, Visual Genome data, and open-source Python deep learning libraries. The work was completed at KFUPM under the supervision of Dr. Muzammil Behzad.
