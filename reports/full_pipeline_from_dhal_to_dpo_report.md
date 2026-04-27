# Full Pipeline Report From Dhal To DPO

No API keys are included in this report.

## Executive Summary

Our work started from the released hallucination data and moved through two practical training directions:

- A direct preference-training path that uses the released preference pairs and trains LLaVA with HSA-DPO or normal DPO.
- A judgement-and-repair path that validates released preference pairs with Gemini/GPT-4o mini, sends rejected rows to LLaVA repair, and can then train on the repaired/merged preference data.

The final simplified comparison is:

- HSA-DPO on released preferences.
- Normal DPO on the same released preferences.

Both use the same base model, image folder, data source, LoRA configuration, and batch/epoch setting so the comparison isolates the DPO objective rather than changing multiple variables at once.

## Starting Dataset: `D_hal`

`D_hal` is the initial set of model responses that may contain hallucinations. In our repository this comes from:

`fg_pipeline/data/hsa_dpo_detection.jsonl`

The dataset contains 16,143 rows:

- 7,643 hallucinated rows.
- 8,500 non-hallucinated rows.

Each row contains an image reference, prompt/question or description context, and a response annotation. Many rows point to images under `vg/images/...`, which is why the `vg/` folder is still needed if we rerun the detector/rewrite reconstruction path.

## Stage 1: Fine-Grained Feedback Construction

Stage 1 converts the released annotation data into structured fine-grained AI feedback.

Input:

`fg_pipeline/data/hsa_dpo_detection.jsonl`

Implemented command:

```bash
bash scripts/run_paper_stage1_faif.sh
```

Implemented code:

- `fg_pipeline/paper/run_stage1_faif.py`
- `fg_pipeline/stage1/parser.py`
- `fg_pipeline/stage1/backends.py`
- `fg_pipeline/paper/prompts.py`

Output:

- `output/fghd/paper_stage1/d_faif.jsonl`
- `output/fghd/paper_stage1/stats.json`

What we did:

- Parsed released annotations into `D_FAIF`.
- Preserved hallucination type, rationale, evidence text, severity label, and severity score.
- Stored prompt/rubric metadata for reproducibility.
- Used the released fine-grained AI feedback rather than regenerating all annotations with external APIs.

Observed result:

- 16,143 `D_FAIF` records were produced.
- 7,643 hallucinated records.
- 8,500 non-hallucinated records.

Why VLMs matter here:

Fine-grained feedback must be visually grounded. The reference methodology used strong vision-language models to generate fine-grained feedback because text-only scoring cannot reliably verify whether an object, attribute, count, action, or relationship exists in the image.

## Stage 2: Detector Dataset Construction

Stage 2 turns `D_FAIF` into supervised training data for a local hallucination detector.

Input:

`output/fghd/paper_stage1/d_faif.jsonl`

Implemented command:

```bash
bash scripts/run_paper_stage2_detector_dataset.sh
```

Implemented code:

- `fg_pipeline/paper/run_stage2_detector_dataset.py`
- `fg_pipeline/paper/prompts.py`

Output:

- `output/fghd/paper_stage2/detector_train.json`
- `output/fghd/paper_stage2/detector_split_stats.json`

What we did:

- Built image/question/response detector examples.
- For hallucinated rows, the target is a structured tags/scores hallucination report.
- For clean rows, the target is `NO HALLUCINATION`.
- Used all available rows because the user chose not to reserve a validation split.

Observed result:

- 16,143 detector examples were produced.
- 7,643 hallucinated examples.
- 8,500 non-hallucinated examples.

Reference-method alignment:

The reference setup trains a local LVLM detector from fine-grained feedback. The detector is intended to replace repeated expensive API annotation when constructing many preference pairs.

## Stage 3: Validation, Detection, And Judgement Experiments

Stage 3 was the most experimental part of our work. We tried several LVLM/API approaches.

### Experiment A: Qwen-VL-Chat And LLaVA Voting

Models/backends:

- Qwen-VL-Chat.
- LLaVA family model.

Goal:

Validate Stage 2 rewrites using multiple model votes.

What happened:

- The initial setup was too slow.
- Qwen dependency warnings appeared for missing `tiktoken`, `matplotlib`, and `transformers_stream_generator`.
- Qwen checkpoint loading repeated and consumed significant time.
- The voting process was later optimized with early stopping, resume, incremental writes, and smaller token budgets.

Outcome:

This path was superseded because it was too slow and brittle for the one-box Vast workflow.

### Experiment B: Gemini + GPT-4o Mini Validation

Models/backends:

- Gemini 2.5 Flash-Lite.
- GPT-4o mini.

Implemented command:

```bash
bash scripts/run_stage3_validate.sh
```

Later released-preference validation command:

```bash
bash scripts/run_released_pref_stage3_validate.sh
```

Implemented code:

- `fg_pipeline/stage3/run_stage3.py`
- `fg_pipeline/stage3/backends.py`
- `fg_pipeline/paper/run_released_pref_stage3_validate.py`

Goal:

Use VLM API judges to decide whether a chosen rewrite is visually grounded and better than the rejected/original answer.

Important detail:

The API judges receive the image plus the question, rejected response, chosen response, and hallucination tags/severity. They output strict JSON:

```json
{"approved": true, "reason": "concise reason"}
```

Observed released-preference result:

- GPT-4o mini validation completed on 8,386 rows.
- 2,914 accepted.
- 5,472 rejected.

Lesson:

API judgement can be useful for audit/filtering, but strict filtering can reduce the training set too much. Therefore it should be reported as a separate judgement-and-repair path, not mixed with the direct HSA-DPO comparison.

### Experiment C: LLaVA Detector Training Attempt

Model/backend:

- LLaVA-1.5-13B detector LoRA path.

Goal:

Train a local detector from Stage 2 detector data.

What happened:

- The run failed because the bundled LLaVA import path was not correctly resolved in the environment.
- The error was `No module named llava`.

Outcome:

This path was abandoned because it required deeper environment repair and was heavier than the available time/instance constraints.

### Experiment D: Qwen2.5-VL-7B Detector Training And Inference

Model/backend:

- Qwen2.5-VL-7B-Instruct with LoRA.

Goal:

Train a local LVLM detector from the Stage 2 detector dataset.

What happened:

- Initial model download hit storage pressure.
- The first training attempt failed from a Transformers argument mismatch: `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`.
- A later attempt failed from an image token mismatch when the max length was too short.
- Training proceeded after switching to a longer sequence length.
- Detector inference produced 16,143 detection rows but predicted 0 hallucinated rows.

Outcome:

Stage 4 then produced 0 preference pairs. This means the detector path failed for downstream training.

Reason for failure:

The most likely reason is detector target/prompt/parser collapse or poor calibration, not proof that Qwen is inherently too weak. A detector that predicts all rows as non-hallucinated cannot drive a detect-then-rewrite pipeline because no rows are selected for rewriting.

### Experiment E: Qwen3-VL-Flash API Judge Idea

Model/backend:

- Qwen3-VL-Flash API was considered.

Outcome:

Cancelled before use. We removed Qwen API/model usage from the active plan to keep the final work to two pipelines.

## Stage 4: Rewrite And Repair

Stage 4 has two implemented variants.

### Variant 1: Detector-Based Rewrite

Implemented command:

```bash
bash scripts/run_paper_stage4_rewrite.sh
```

Implemented code:

- `fg_pipeline/paper/run_stage4_rewrite.py`
- `fg_pipeline/paper/prompts.py`
- `fg_pipeline/stage2/backends.py`

Input:

`output/fghd/paper_stage3/detections.jsonl`

Expected output:

- `output/fghd/paper_stage4/rewrite_records.jsonl`
- `output/fghd/paper_stage4/preference_pairs.jsonl`
- `output/fghd/paper_stage4/stats.json`

Purpose:

Use detected hallucination findings to rewrite the original hallucinated response into a corrected chosen response.

Why it failed in our Qwen detector run:

The Qwen detector predicted 0 hallucinated rows, so Stage 4 skipped all rows as non-hallucinated and produced 0 preference pairs.

### Variant 2: Released-Preference Judgement And LLaVA Repair

Implemented validation command:

```bash
bash scripts/run_released_pref_stage3_validate.sh
```

Implemented repair command:

```bash
bash scripts/run_released_pref_stage4_repair.sh
```

Implemented full judgement-repair-train command:

```bash
bash scripts/run_released_pref_pipeline.sh
```

Implemented code:

- `fg_pipeline/paper/run_released_pref_stage3_validate.py`
- `fg_pipeline/paper/run_released_pref_stage4_repair.py`

Process:

- Start from released preference pairs.
- Use Gemini and/or GPT-4o mini to judge whether the chosen response is acceptable.
- Pass accepted rows directly forward.
- Send rejected rows to LLaVA-7B repair.
- Merge accepted and repaired rows into a final preference file.
- Train HSA-DPO on the final preference file if desired.

Why this path exists:

It gives us a second pipeline that uses VLM judgement as an additional quality-control step before training. It should be reported separately from the direct training path because API judgement changes the training data.

## Stage 5: Preference Training

Stage 5 trains the final mitigation model.

Base model used in current experiments:

`models/llava-v1.5-7b`

Data used in current direct experiments:

`hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl`

Image folder:

`hsa_dpo/data/images`

Implemented script:

```bash
bash scripts/run_paper_stage5_train_hsa.sh
```

Training launcher:

`hsa_dpo_train.sh`

Training implementation:

- `hsa_dpo/models/llava-v1_5/train_dpo.py`
- `hsa_dpo/trainer/base_dpo_trainer.py`

Implemented DPO loss types:

- `hsa_weighted`: severity-aware HSA-DPO style weighting.
- `severity_margin`: margin-based severity objective.
- `standard`: normal DPO baseline.

Current final HSA-DPO command:

```bash
bash scripts/run_direct_stage5_paper_hsa_batch32_epoch1.sh
```

Current final normal DPO command:

```bash
bash scripts/run_direct_stage5_normal_dpo_batch32_epoch1.sh
```

Queued normal DPO command:

```bash
bash scripts/queue_direct_normal_dpo_after_paper_b32_e1.sh
```

Current final comparison:

- HSA-DPO, effective batch 32, 1 epoch.
- Normal DPO, effective batch 32, 1 epoch.
- Same base model.
- Same preference data.
- Same image folder.
- Same LoRA settings.
- Same learning rate and DPO beta.

Main hyperparameters:

- Base: LLaVA-1.5-7B.
- LoRA rank: 128.
- LoRA alpha: 256.
- Learning rate: `2e-6`.
- DPO beta: `0.1`.
- Effective total batch size: 32 for final runs.
- Epochs: 1 for final time-constrained comparison.
- Precision: bf16.
- DeepSpeed: ZeRO-3.
- Projector: frozen through `mm_projector_lr=0`.
- Model max length: 1024.

Why LLaVA was used:

The reference methodology uses LLaVA-family mitigation training and released preference data built for LLaVA-style responses. Using LLaVA keeps our training compatible with the released preference data and makes the HSA-DPO versus normal DPO comparison cleaner.

Why 7B instead of 13B:

The reference setup includes 13B-scale LLaVA training, but our available one-box 48GB workflow made 13B slower and more fragile. LLaVA-1.5-7B is a practical compromise that lets us finish controlled comparisons within the available time.

## Two Final Pipelines

### Pipeline 1: Direct Released-Preference Training

Flow:

```text
D_hal / released fine-grained feedback
-> released preference pairs
-> LLaVA-1.5-7B HSA-DPO
-> LLaVA-1.5-7B normal DPO baseline
-> evaluation
```

Purpose:

This is the cleanest training comparison because both runs use the same released preference data.

Status:

Implemented.

### Pipeline 2: Gemini/GPT Judgement + LLaVA Repair + Training

Flow:

```text
released preference pairs
-> Gemini/GPT-4o mini judgement
-> accepted rows pass
-> rejected rows repaired by LLaVA-7B
-> accepted + repaired rows merged
-> HSA-DPO training
-> evaluation
```

Purpose:

This tests whether additional VLM judgement and LLaVA repair can improve preference quality before training.

Status:

Implemented as scripts and code. GPT-4o mini validation completed with 2,914 accepted and 5,472 rejected out of 8,386 rows. Gemini validation was still active at last report.

## Evaluation Metrics

The automatic evaluation setup supports:

- POPE adversarial: accuracy, precision, recall, F1, yes ratio.
- Object HalBench: CHAIRs and CHAIRi.
- AMBER: CHAIR, Cover, Hal, Cog.

Judge-based evaluation benchmarks are separated because they require external judges:

- MMHal-Bench.
- LLaVA-Bench in the wild.
- HSS/severity scoring.

Reporting rule:

Only claim a direct comparison when the same model scale, benchmark assets, and official protocol are used. Otherwise label the result as a supplemental local comparison.

## Implementation Checklist

Implemented:

- Stage 1 released fine-grained feedback parser.
- Stage 2 detector dataset builder.
- Released-preference API validation with Gemini/OpenAI.
- Released-preference LLaVA repair.
- Direct HSA-DPO training.
- Direct normal DPO baseline training.
- Automatic evaluation runner.
- Experiment log and methodology notes.

Experimental or abandoned:

- Qwen/LLaVA local voting validation.
- LLaVA-13B local detector training.
- Qwen2.5-VL-7B detector training/inference.
- Qwen3-VL-Flash API judge.
- Severity-margin DPO ablation.

Not run as final:

- The full detector-based rewrite path, because the detector predicted 0 hallucinated rows.
- LLaVA-13B final training, because of resource/time constraints.

## What Needs To Be Reported To The Instructor

The report should say that we explored the full fine-grained feedback and preference-training pipeline, but the stable final experiment was narrowed to a controlled DPO comparison.

Key points:

- We used released fine-grained AI feedback as the supervision source.
- We built `D_FAIF` and detector training data.
- We tested multiple LVLM/API options for validation and detection.
- Qwen and LLaVA detector paths failed for practical implementation reasons in this environment.
- We therefore used the released preference data directly for final training.
- We split final work into direct training and API-judgement/repair training.
- The final controlled comparison is HSA-DPO versus normal DPO on the same LLaVA-1.5-7B base and same preference data.
