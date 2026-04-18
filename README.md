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

This working copy now has two layers:

- the original `hsa_dpo/` package and training flow from the released paper
- a lightweight `fg_pipeline/` extension layer for the new confidence-aware pipeline

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
huggingface-cli download --repo-type dataset WenyiXiao/HSA-DPO --local-dir ./datasets
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
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b

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

The new project work lives under `fg_pipeline/` and reuses the original HSA-DPO code.

Current stage launchers:

```bash
bash scripts/run_stage3_confidence.sh
bash scripts/run_stage4_rewrite.sh
bash scripts/run_stage5_verify.sh
bash scripts/run_stage6_train.sh
```

What these stages do today:

- Stage 3 bootstraps `D_det` from `hsa_dpo/data/hsa_dpo_detection.jsonl`
- Stage 4 is a scaffold rewrite stage and currently uses placeholder passthrough logic
- Stage 5 converts filtered rewrite outputs into an HSA-DPO-compatible preference format
- Stage 6 reuses the original `hsa_dpo_train.sh` with `DATA_PATH=output/fghd/D_pref_clean.jsonl`

Current status:

- the extension layer is ready as a project scaffold
- baseline HSA-DPO training works through the original code path
- a real rewrite model still needs to be added before the full new pipeline can produce useful clean preference pairs

### Key Parameters

- `use_chosen_score`: Whether to use chosen scores in DPO loss (default: False)
- `use_rejected_score`: Whether to use rejected scores in DPO loss (default: True)
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
