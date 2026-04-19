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
- installs `huggingface_hub` and `modelscope`

What it does **not** do:

- it does not change your SSH config
- it does not inject API keys
- it does not download gated models automatically

## 6. Step 5: Download The Base Model

Run this on the Vast instance after the bootstrap script:

```bash
source .venv/bin/activate
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b
```

The training script already points to:

- `MODEL_PATH="./models/llava-v1.5-13b"`

## 7. Step 6: Verify Repo Paths Before Training

This repo currently uses:

- preference data: `./hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl`
- images: `./hsa_dpo/data/images`
- model path: `./models/llava-v1.5-13b`

The local image directory is `images`, not `image`.

## 8. Step 7: Adjust The Training Script For The Remote GPU

Edit `hsa_dpo_train.sh` on the remote machine only if needed.

The settings you are most likely to change are:

- `NUM_GPUS`
- `BATCH_SIZE`
- `MODEL_PATH`
- `OUTPUT_DIR`

Examples:

- if the instance has 1 GPU, set `NUM_GPUS=1`
- if VRAM is tight, reduce `BATCH_SIZE`

## 9. Step 8: Start Training

Run this on the Vast instance:

```bash
source .venv/bin/activate
bash hsa_dpo_train.sh
```

For a smaller first validation run, use:

```bash
bash scripts/vastai/run_pilot_train.sh
```

If you disconnect often, run inside `tmux`.

Useful commands:

```bash
tmux new -s hsa-dpo
bash hsa_dpo_train.sh
tmux attach -t hsa-dpo
```

## 10. What You Should Add Next

If you want to make the Vast AI workflow cleaner, the next repo changes should be:

1. make `hsa_dpo_train.sh` fail fast when `deepspeed` is missing
2. add a single-GPU fallback script for non-DeepSpeed runs
3. add a small env file or shell file for per-instance overrides
4. add checkpoint sync instructions for downloading results back to your laptop

## 11. Minimal Workflow Summary

### On Windows

1. set Vast API key
2. create SSH key
3. update `C:\Users\ahrhq\.ssh\config`
4. connect to `vastai`

### In this repo

1. keep `hsa_dpo_train.sh`
2. use `scripts/vastai/bootstrap.sh`
3. use this document as the project-specific checklist

### On the Vast instance

1. clone repo
2. run `bash scripts/vastai/bootstrap.sh`
3. download LLaVA base model
4. run `bash hsa_dpo_train.sh`
