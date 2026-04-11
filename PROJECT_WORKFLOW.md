# Project Workflow

## 1. Overview

This project is based on the paper **Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback**.

Important distinction:

- The **paper** describes a full pipeline from hallucination generation to detector training to mitigation training and evaluation.
- The **repository** mainly implements the **mitigation training stage** and a small **inference path**.

## 2. What The Repo Actually Implements

Implemented here:

- package setup: `pyproject.toml`
- training launcher: `hsa_dpo_train.sh`
- mitigation training: `hsa_dpo/models/llava-v1_5/train_dpo.py`
- DPO/HSA-DPO trainer logic:
  - `hsa_dpo/trainer/base_dpo_trainer.py`
  - `hsa_dpo/trainer/llava_dpo_trainer.py`
- inference:
  - `inference/inference_example.py`
  - `inference/inference_example.ipynb`

Mostly reference or upstream support:

- `hsa_dpo/models/llava-v1_5/docs/`
- `hsa_dpo/models/llava-v1_5/scripts/`
- `hsa_dpo/models/llava-v1_5/llava/serve/`

Not clearly implemented here:

- GPT-4 / GPT-4V annotation generation
- detector training
- detect-then-rewrite preference construction
- full paper benchmark evaluation pipeline

## 3. Paper Workflow

The paper workflow is:

1. Generate hallucinatory responses.
   - DDG prompts from `Visual Genome`
   - VCR prompts from `Silkie`
   - DDG instructions from `RLHF-V`
2. Collect fine-grained AI feedback.
   - `GPT-4` for DDG
   - `GPT-4V` for VCR
3. Train a hallucination detector.
   - appendix says the detector backbone is `InternVL-Chat-Plus-v1.2-40B`
4. Build preference pairs with detect-then-rewrite.
   - rewriting model: `LLaVA`
5. Train the mitigation model with HSA-DPO.
   - main base model: `LLaVA-1.5-13B`
   - also reported: `Qwen-VL-Chat-7B`
6. Evaluate on detection and mitigation benchmarks.

## 4. Repository Workflow

The runnable local workflow is much smaller:

1. Install the package.
2. Download the mitigation dataset and images.
3. Download the base LLaVA model.
4. Edit `hsa_dpo_train.sh` paths and training settings.
5. Run `hsa_dpo_train.sh`.
6. Test the output with `inference/inference_example.py`.

## 5. Key Files

### Core files

- `README.md`
  - install, dataset, train, inference instructions
- `hsa_dpo_train.sh`
  - main training command
- `hsa_dpo/models/llava-v1_5/train_dpo.py`
  - loads preference data, images, model, and trainer
- `hsa_dpo/trainer/base_dpo_trainer.py`
  - score-aware DPO loss logic
- `hsa_dpo/trainer/llava_dpo_trainer.py`
  - LLaVA-specific chosen/rejected multimodal batching
- `inference/inference_example.py`
  - post-training inference test

### Reference files

- `asset/Referenced_Research_Paper.pdf`
- `asset/HSA_DPO_Appendix.pdf`
- `asset/overview.png`

## 6. Datasets

### Datasets in the paper

- `Visual Genome`
  - DDG prompts and annotation support
- `Silkie`
  - VCR prompts
- `RLHF-V instruction set`
  - DDG prompt instructions
- `D_faif`
  - fine-grained detector training data
- `D_pref`
  - preference data for HSA-DPO

### Datasets referenced by this repo

- `hsa_dpo_detection.jsonl`
  - mentioned in `README.md`
  - no detector training script found locally
- `hsa_dpo_preference_llava1dot5.jsonl`
  - used by `train_dpo.py`
- `hsa_dpo_imgs.tar.gz`
  - image archive for preference training

### Evaluation benchmarks in the paper

- `MHaluBench`
- `MFHaluBench`
- `Object HalBench`
- `AMBER`
- `MMHal-Bench`
- `POPE` Adversarial
- `LLaVA Bench in the wild`

## 7. Models And Baselines

Paper-side models:

- `GPT-4`
- `GPT-4V`
- `InternVL-Chat-Plus-v1.2-40B`
- `LLaVA-1.5-13B`
- `Qwen-VL-Chat-7B`

Paper-side baselines include:

- standard `DPO`
- `InstructBLIP`
- `LLaVA-1.5`
- `Qwen-VL-Chat`
- `GPT-4V`
- `LRV`
- `LLaVA-RLHF`
- `RLHF-V`
- `Silkie`
- `POVID`

## 8. Best Starting Baseline

Best baseline to start with:

- **standard DPO on the same base LLaVA-1.5 model and same preference dataset**

Why:

- easiest to reproduce in this repo
- fairest comparison against weighted HSA-DPO-style training
- requires the fewest code changes

Practical toggle:

- standard DPO:
  - `use_chosen_score=False`
  - `use_rejected_score=False`
- weighted run:
  - current launcher already uses `use_rejected_score=True`

## 9. What To Edit First

Edit first:

- `hsa_dpo_train.sh`
  - data path
  - image folder
  - model path
  - output dir
  - epochs / batch size / learning rate
  - score flags

Edit next if needed:

- `hsa_dpo/models/llava-v1_5/train_dpo.py`
  - dataset schema
  - image naming/path logic
  - training arguments

Edit carefully:

- `hsa_dpo/trainer/base_dpo_trainer.py`
  - changes the actual loss
- `hsa_dpo/trainer/llava_dpo_trainer.py`
  - changes multimodal batching and score passing

Do not edit early:

- `hsa_dpo/models/llava-v1_5/docs/`
- `hsa_dpo/models/llava-v1_5/scripts/v1_5/eval/`
- `hsa_dpo/models/llava-v1_5/llava/serve/`
- `hsa_dpo/models/llava-v1_5/llava/model/language_model/mpt/`

## 10. What Is Missing

Missing or incomplete for full paper reproduction:

- detector training code
- annotation-generation code
- detect-then-rewrite code
- self-contained benchmark evaluation code
- results / checkpoints / logs in the repo

Because of that, the safest interpretation is:

- this repo is a **mitigation-training starting point**
- not a complete end-to-end reproduction of the paper

## 11. Recommended Next Steps

1. Read:
   - `README.md`
   - `asset/Referenced_Research_Paper.pdf`
   - `asset/HSA_DPO_Appendix.pdf`
2. Download:
   - `hsa_dpo_preference_llava1dot5.jsonl`
   - `hsa_dpo_imgs.tar.gz`
   - base `LLaVA-v1.5` weights
3. Verify image extraction and `id -> .jpg` mapping.
4. Run a short smoke test with `hsa_dpo_train.sh`.
5. Run inference with `inference/inference_example.py`.
6. Reproduce:
   - standard DPO
   - weighted DPO / HSA-DPO-style run
7. Only after that, add your novelty or rebuild the missing paper stages.
