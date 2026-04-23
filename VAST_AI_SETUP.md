# Vast AI Setup For This Repo

This file maps a generic Vast AI SSH setup onto this specific repository.

The key distinction is:

- some setup belongs on your local Windows machine,
- some setup belongs inside the repo,
- and some setup belongs on the remote Vast AI Linux instance.

## 1. What Goes Where

### Local machine only

These do **not** belong in the repo:

- your Vast API key
- your SSH private key
- your local SSH config
- your VS Code Remote SSH setup

Use these local paths on Windows:

- SSH private key: `C:\Users\ahrhq\.ssh\vast_key`
- SSH public key: `C:\Users\ahrhq\.ssh\vast_key.pub`
- SSH config: `C:\Users\ahrhq\.ssh\config`

### Repo files

These **do** belong in the repo:

- this guide: `VAST_AI_SETUP.md`
- remote bootstrap script: `scripts/vastai/bootstrap.sh`
- training launcher: `hsa_dpo_train.sh`

These should stay **local-only** and are ignored by Git:

- `VAST_AI_SETUP.local.md`
- `.vastai/`
- `scripts/vastai/defaults.local.env`
- `scripts/vastai/*.local.sh`
- `scripts/vastai/*.local.env`
- `scripts/vastai/*.local.py`
- `scripts/vastai/local/`
- `vast_sync.tar`

### Remote Vast instance

These happen after you SSH into the rented GPU:

- clone the repo
- create or activate a Python environment
- install Linux training dependencies
- download the base LLaVA model
- run `hsa_dpo_train.sh`

## 2. Step 1: Prepare Your Local Windows Machine

### Vast CLI

Optional, but useful:

```powershell
python -m pip install vastai
vastai set api-key YOUR_API_KEY
```

### SSH key

Create a dedicated key for Vast:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\vast_key"
```

### SSH config

Add this to `C:\Users\ahrhq\.ssh\config`:

```text
Host vastai
  HostName [INSTANCE_IP]
  User root
  Port [INSTANCE_PORT]
  IdentityFile C:\Users\ahrhq\.ssh\vast_key
  LocalForward 8080 localhost:8080
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

This is the correct place for the connection settings. Do not put this inside the repo.

## 3. Step 2: Rent The Vast AI Instance

Use a Linux CUDA image, not Windows.

For this repo, prefer:

- Ubuntu-based image
- NVIDIA GPU with enough VRAM for LLaVA-1.5-13B LoRA training
- CUDA-compatible PyTorch environment
- at least `2` GPUs if you want to keep `NUM_GPUS=2` in `hsa_dpo_train.sh`

After the instance starts:

1. copy the public IP
2. copy the SSH port
3. update `HostName` and `Port` in `C:\Users\ahrhq\.ssh\config`
4. connect with `ssh vastai` or VS Code Remote SSH

If `ssh vastai` times out, the instance is usually stopped/restarted and the IP or port changed.
Update `HostName` and `Port` from the currently active Vast instance panel and retry.

If OpenSSH reports `Bad owner or permissions on ...\\.ssh\\config` on Windows, reset ACLs:

```powershell
icacls "$env:USERPROFILE\\.ssh" /grant:r "$env:USERDOMAIN\\$env:USERNAME:(OI)(CI)F" "NT AUTHORITY\\SYSTEM:(OI)(CI)F" "BUILTIN\\Administrators:(OI)(CI)F" /t /c
```

## 4. Step 3: Clone The Repo On The Remote Machine

Run this on the Vast instance:

```bash
cd /workspace
git clone <YOUR_REPO_URL>
cd Fine-Grained-Hallucination-Detection-and-Severity-Aware-Mitigation-in-Vision-Language-Models
```

If you uploaded the repo another way, just `cd` into the project root on the remote machine.

## 5. Step 4: Bootstrap The Remote Python Environment

This repo now includes a remote setup script for Vast AI:

- `scripts/vastai/bootstrap.sh`

Run it on the remote machine from the repo root:

```bash
bash scripts/vastai/bootstrap.sh
```

What it does:

- creates `.venv` if needed
- upgrades `pip`, `setuptools`, and `wheel`
- installs this repo in editable mode with Linux training extras
- installs a `huggingface_hub` version compatible with this repo's `transformers` / LLaVA stack
- installs `modelscope`

What it does **not** do:

- it does not change your SSH config
- it does not inject API keys
- it does not download gated models automatically

### Optional: per-instance defaults for a new template / GPU

If you move to a different Vast template or GPU, do not keep re-editing the
tracked launchers. Instead, create an ignored override file on the remote box:

```bash
cp scripts/vastai/defaults.env.example scripts/vastai/defaults.local.env
```

Then edit only `scripts/vastai/defaults.local.env` for the new machine, for
example:

```bash
MODEL_PATH=models/llava-v1.5-13b
QWEN_MODEL_PATH=models/Qwen-VL-Chat
LLAVA_MODEL_PATH=models/llava-v1.5-13b
NUM_GPUS=1
BATCH_SIZE=1
EPOCH=1
```

The Stage 2, Stage 3, Stage 4, and pilot-train launchers now load this file
automatically when it exists.

## 6. Step 5: Download The Base Model

Run this on the Vast instance after the bootstrap script:

```bash
source .venv/bin/activate
hf download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b
```

The training script already points to:

- `MODEL_PATH="./models/llava-v1.5-13b"`

## 7. Step 6: Verify Repo Paths Before Running Training

The paths that matter for baseline Stage 4 training:

- preference dataset: `./hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl`
- training image folder: `./hsa_dpo/data/images`
- model path: `./models/llava-v1.5-13b`
- training output: `./output/hsa_dpo_llava`

Stage 1 (critique detection / extraction) runs off the supervision mirror
at `fg_pipeline/data/hsa_dpo_detection.jsonl`. It is CPU-friendly and
does not need the LLaVA base model:

```bash
bash scripts/run_stage1_critiques.sh
```

Output: `output/fghd/stage1/detection_critiques.jsonl` (plus
`stats.json`).

If you are running the redesigned project pipeline rather than a baseline-only
reproduction, the next steps are:

```bash
bash scripts/run_stage2_rewrites.sh
bash scripts/run_stage3_validate.sh
```

Use the real rewrite backend on a GPU box with:

```bash
BACKEND=llava MODEL_PATH=models/llava-v1.5-13b bash scripts/run_stage2_rewrites.sh
```

For the local research Stage 3 backend on a real GPU box:

```bash
QWEN_MODEL_PATH=models/Qwen-VL-Chat \
LLAVA_MODEL_PATH=models/llava-v1.5-13b \
bash scripts/run_stage3_validate.sh
```

## 8. Step 7: Run Stage 4 Training (HSA-DPO Baseline)

For the redesigned Stage 1-4 project pipeline, run Stage 4 through the wrapper
after Stage 3:

```bash
source .venv/bin/activate
bash scripts/run_stage4_train.sh
```

This uses `output/fghd/stage3/preference_pairs.jsonl` as the training data and
writes checkpoints under `output/fghd/stage4_llava`.

For a baseline-only reproduction run, use the released preference data
directly:

```bash
source .venv/bin/activate
DATA_PATH=hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl \
IMAGE_FOLDER=hsa_dpo/data/images \
MODEL_PATH=models/llava-v1.5-13b \
OUTPUT_DIR=output/hsa_dpo_llava \
bash hsa_dpo_train.sh
```

If the instance only has 1 GPU with 32 GB VRAM, this configuration is expected
to OOM. Use a larger box rather than forcing the same setup onto that machine.

For a smaller first validation run, use:

```bash
bash scripts/vastai/run_pilot_train.sh
```

If you disconnect often, run inside `tmux`:

```bash
tmux new -s hsa-dpo
bash hsa_dpo_train.sh
tmux attach -t hsa-dpo
```

## 9. Evaluation

On the remote instance, after the bootstrap script and model download:

```bash
source .venv/bin/activate
bash scripts/run_paper_eval.sh
bash scripts/run_general_eval.sh
```

`run_paper_eval.sh` is now the strict paper-comparison wrapper and is local
only by default. `run_general_eval.sh` writes general runtime summaries plus a
separate supplemental local-eval report. Proxy or judge-like benchmarks that
are not yet paper-faithful are intentionally kept out of the strict paper table.

## 10. Minimal Workflow Summary

### On Windows

1. set Vast API key
2. create SSH key
3. update `C:\Users\ahrhq\.ssh\config`
4. connect to `vastai`
5. keep any per-instance notes or one-off commands in `VAST_AI_SETUP.local.md`, not in tracked docs

### In this repo

1. keep `hsa_dpo_train.sh` as the baseline training entrypoint
2. use `scripts/vastai/bootstrap.sh`
3. use this document as the project-specific checklist
4. put machine-specific overrides in ignored `scripts/vastai/*.local.*` files rather than editing tracked setup docs for each instance

### On the Vast instance

1. clone repo
2. run `bash scripts/vastai/bootstrap.sh`
3. download LLaVA base model
4. optionally run `bash scripts/run_stage1_critiques.sh` to refresh the
   Stage 1 critique output (CPU-friendly; does not need the LLaVA model)
5. optionally run `bash scripts/run_stage1_detector_dataset.sh` and
   `bash scripts/run_stage1_detector_train.sh` if you want a local Stage 1
   detector model rather than parser-only extraction
5. optionally run `bash scripts/run_stage2_rewrites.sh` with
   `BACKEND=llava MODEL_PATH=models/llava-v1.5-13b` for the real rewrite
   backend (the default `template` backend works without a model)
6. optionally run `bash scripts/run_stage3_validate.sh` to build clean
   preference pairs from the Stage 2 rewrites; if `QWEN_MODEL_PATH` and
   `LLAVA_MODEL_PATH` are set, the launcher prefers the local Qwen+LLaVA
   ensemble backend automatically
7. run `bash scripts/run_stage4_train.sh` on a 2-GPU box for the redesigned
   pipeline, or `bash hsa_dpo_train.sh` for a baseline-only reproduction
8. optionally run `bash scripts/run_paper_eval.sh` / `bash scripts/run_general_eval.sh`

If you switch to a new template or GPU type, the first thing to copy over is:

```bash
cp scripts/vastai/defaults.env.example scripts/vastai/defaults.local.env
```

Then set the per-box values there instead of editing the tracked launchers.
