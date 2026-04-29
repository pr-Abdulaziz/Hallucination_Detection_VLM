# HSA-DPO Inference

This directory contains inference code and examples for comparing the base
LLaVA model against the trained HSA-DPO adapter.

## Setup

1. Make sure you've installed HSA-DPO with its dependencies:
```bash
pip install -e .
```

2. Download or prepare the model weights:
```bash
# Base model used by the current experiment
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir models/llava-v1.5-7b

# Trained adapter from the two-shot HSA-DPO-M experiment
output/fghd/exp_2shot_verified_margin_hsa_b32_e1
```

## Quick Start

### Cached Before/After Comparison

This reads the saved evaluation predictions and does not load the model:

```bash
python inference/inference_example.py \
    --mode cached \
    --eval-dir output/eval/2shot_verified_margin_eval_full_20260428_172728 \
    --benchmark object_halbench \
    --example-id 1903
```

### Live Before/After Inference

```bash
python inference/inference_example.py \
    --mode live \
    --base-model models/llava-v1.5-7b \
    --adapter-path output/fghd/exp_2shot_verified_margin_hsa_b32_e1 \
    --image hsa_dpo/models/llava-v1_5/llava/serve/examples/waterview.jpg \
    --prompt "Describe this image in detail. Mention only visible objects and avoid unsupported assumptions."
```

