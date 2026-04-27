# Experiment Log For Instructor

No API keys are included. Proprietary APIs are named only by provider/model.

## Latest Known Status

- **Report purpose:** Instructor-facing experiment log for the hallucination mitigation project. It lists completed, failed, interrupted, active, and queued experiments discussed/executed in this chat.
- **Generated:** 2026-04-26 14:17
- **Main caution:** No API keys are included. Proprietary APIs are named only by provider/model.
- **Current main training:** HSA-DPO, LLaVA-1.5-7B, released HSA-DPO preferences, effective batch 32, 1 epoch, active at last check.
- **Current baseline queue:** Normal DPO, same LLaVA-1.5-7B/data/settings, queued to run after HSA-DPO finishes.
- **Best completed judgement result so far:** OpenAI GPT-4o mini judgement over released preferences: 8386 rows, 2914 accepted, 5472 rejected.
- **Qwen detector outcome:** Qwen2.5-VL-7B local detector training/inference path was attempted, but inference predicted 0 hallucinated rows; Stage 4 produced 0 preference pairs.
- **Full pipeline report:** `reports/full_pipeline_from_dhal_to_dpo_report.md`.

## VLM Fine-Grained Feedback And Prompt Alignment

- **Why VLMs were used:** The task is image-grounded hallucination mitigation, so hallucination feedback must verify visual claims such as objects, attributes, relations, counts, and actions against the image.
- **Referenced-paper alignment:** The paper uses GPT-4V/GPT-4 to produce fine-grained AI feedback, trains an open-source LVLM detector from that feedback, uses LLaVA for rewriting, and then trains with HSA-DPO.
- **Our supervision source:** We use the released HSA-DPO fine-grained annotations as the main `D_FAIF` source instead of regenerating all GPT-4/GPT-4V annotations.
- **Prompt requirement:** Paper-path prompts are stored in `fg_pipeline/paper/prompts.py` and should remain appendix-aligned for DDG annotation, VCR annotation, detector output, rewrite, and severity rubric.
- **Main reporting point:** API judges are audit/filter experiments; the current fair Stage 5 comparison is HSA-DPO versus normal DPO on the same released preference data and same LLaVA-1.5-7B base.
- **Detailed note:** See `reports/vlm_fine_grained_feedback_methodology.md`.

## Experiments

### E01 - Released fine-grained annotation parser
- **Stage:** Stage 1
- **Experiment:** Released fine-grained annotation parser
- **Models/Backends:** No model call; parses released HSA-DPO detection annotations
- **Data/Input:** fg_pipeline/data/hsa_dpo_detection.jsonl
- **Main settings:** Parser-only; preserve critiques/severity labels
- **Status:** Completed
- **Result/Observed output:** 16143 rows: 7643 hallucinated, 8500 non-hallucinated; about 7749 parsed critique items noted earlier
- **Failure/Issue:** None; this is released supervision, not regenerated GPT/Gemini annotation
- **Reason/Lesson:** Paper-faithful enough for our budget because released annotations already represent D_faif.
- **Artifacts:** output/fghd/stage1/* and later output/fghd/paper_stage1/d_faif.jsonl

### E02 - Original rewrite generation from critiques
- **Stage:** Stage 2
- **Experiment:** Original rewrite generation from critiques
- **Models/Backends:** LLaVA/template path depending on run; template only for smoke tests
- **Data/Input:** Stage 1 hallucinated rows/critiques
- **Main settings:** Rewrite hallucinated responses using critique text
- **Status:** Completed earlier
- **Result/Observed output:** Stage 2 outputs existed and were used by legacy Stage 3 validation
- **Failure/Issue:** Prompt was shorter than paper appendix rewrite prompt
- **Reason/Lesson:** Useful for smoke/current pipeline, but less paper-faithful than detect-then-rewrite with detector outputs.
- **Artifacts:** output/fghd/stage2/*

### E03 - Paper-faithful D_FAIF build from released annotations
- **Stage:** Paper Stage 1
- **Experiment:** Paper-faithful D_FAIF build from released annotations
- **Models/Backends:** No model call; released annotation parser plus appendix prompt templates for reproducibility
- **Data/Input:** fg_pipeline/data/hsa_dpo_detection.jsonl
- **Main settings:** Appendix-aligned metadata; preserve raw annotation text/source
- **Status:** Completed
- **Result/Observed output:** 16143 D_FAIF records written; 7643 hallucinated and 8500 non-hallucinated
- **Failure/Issue:** No failure
- **Reason/Lesson:** This is the clean paper-path source for detector/data experiments.
- **Artifacts:** output/fghd/paper_stage1/d_faif.jsonl; output/fghd/paper_stage1/stats.json

### E04 - Detector training dataset construction
- **Stage:** Paper Stage 2
- **Experiment:** Detector training dataset construction
- **Models/Backends:** No model call; dataset conversion
- **Data/Input:** output/fghd/paper_stage1/d_faif.jsonl
- **Main settings:** Hallucinated target: typed critique/severity report; clean target: NO HALLUCINATION; deterministic ordering/sampling
- **Status:** Completed
- **Result/Observed output:** 16143 detector examples written: 7643 hallucinated, 8500 non-hallucinated
- **Failure/Issue:** Dataset had fewer non-hallucinated rows than ideal 1:1.2 ratio target (9172); used all 8500
- **Reason/Lesson:** No validation split was used by user choice; use all data for detector dataset.
- **Artifacts:** output/fghd/paper_stage2/detector_train.json; detector_split_stats.json

### E05 - Qwen/LLaVA ensemble validation speedup
- **Stage:** Legacy Stage 3
- **Experiment:** Qwen/LLaVA ensemble validation speedup
- **Models/Backends:** Qwen-VL-Chat + LLaVA family ensemble
- **Data/Input:** Stage 2 rewrites over 7643 rows
- **Main settings:** Initially 3 votes Qwen->LLaVA->Qwen; later early stop, resume, incremental writes, max tokens adjusted
- **Status:** Partially implemented / superseded
- **Result/Observed output:** Speed/robustness code added; not kept as final research path
- **Failure/Issue:** Missing qwen-side packages caused warnings/import failures: tiktoken, matplotlib, transformers_stream_generator; Qwen checkpoint loading was slow and repeated; generation config needed fixing
- **Reason/Lesson:** Heavy local model voting was too slow and brittle for the one-box workflow; preflight checks and resume were necessary.
- **Artifacts:** scripts/run_stage3_validate.sh; output/fghd/stage3/* legacy outputs

### E06 - Gemini + GPT-4o mini strict validation over rewritten Stage 2 output
- **Stage:** Legacy Stage 3
- **Experiment:** Gemini + GPT-4o mini strict validation over rewritten Stage 2 output
- **Models/Backends:** Gemini 2.5 Flash-Lite + GPT-4o mini
- **Data/Input:** 7643 rewritten rows from current Stage 2/Stage 4-era pipeline
- **Main settings:** Two-vote judgement; pass only under strict rule used at that time
- **Status:** Completed but not enough data
- **Result/Observed output:** One run wrote 98 preference pairs and 7643 audit rows; an earlier logic issue showed 0 preference pairs when pass logic/path did not match Stage 4 input
- **Failure/Issue:** Too strict for training: only about 98 accepted pairs, far below thousands needed
- **Reason/Lesson:** API validation can audit quality, but using it as the only gate can destroy dataset size.
- **Artifacts:** output/fghd/stage3/vote_records.jsonl; output/fghd/stage3/preference_pairs.jsonl; stats.json

### E07 - Repair rows rejected by API validation
- **Stage:** Legacy Stage 4
- **Experiment:** Repair rows rejected by API validation
- **Models/Backends:** LLaVA-1.5-13B initially, later LLaVA-1.5-7B after storage/time tradeoff
- **Data/Input:** Rejected/failed rows from Stage 3 validation
- **Main settings:** LLaVA repair/rewrite, resumable; deterministic local generation
- **Status:** Completed in one run
- **Result/Observed output:** Stage 4 repair wrote 7545 repair pairs and 7643 final preference pairs in the current/legacy path
- **Failure/Issue:** Long runtime; LLaVA checkpoint loading and generation were slow; model-size/storage constraints pushed move from 13B to 7B
- **Reason/Lesson:** Repair can restore dataset volume after strict validation, but it is costly and not exactly paper detector->rewrite methodology.
- **Artifacts:** output/fghd/stage4/repair_records.jsonl; output/fghd/stage4/final_preference_pairs.jsonl; stats.json

### E08 - Severity-margin HSA-DPO on legacy Stage 4 pairs
- **Stage:** Legacy Stage 5
- **Experiment:** Severity-margin HSA-DPO on legacy Stage 4 pairs
- **Models/Backends:** LLaVA-1.5-13B initially, LoRA, DeepSpeed ZeRO-3
- **Data/Input:** output/fghd/stage4/final_preference_pairs.jsonl
- **Main settings:** 2 epochs; LoRA r=128 alpha=256; LR=2e-6; total batch 32; beta=0.1; freeze projector; severity_margin loss
- **Status:** Completed earlier
- **Result/Observed output:** Training completed at global_step=476; evaluation preflight confirmed trainer_state global_step=476
- **Failure/Issue:** Initial OOM on 48GB for 13B/large settings; ZeRO-3 no_sync incompatibility needed trainer patch; runtime long
- **Reason/Lesson:** Trainer compatibility patches were required; 13B is expensive for single 48GB GPU.
- **Artifacts:** output/fghd/stage5_llava_margin/checkpoint-476; trainer_state.json

### E09 - Automatic evaluation runner without external judge APIs
- **Stage:** Evaluation
- **Experiment:** Automatic evaluation runner without external judge APIs
- **Models/Backends:** LLaVA-1.5 base vs Stage 5 LoRA; automatic metrics only
- **Data/Input:** POPE adversarial, Object HalBench, AMBER assets
- **Main settings:** No OpenAI/Gemini judge; run names stage5_auto_eval_smoke/full; temperature=0, max_new_tokens=512
- **Status:** Set up / partial run
- **Result/Observed output:** Preflight initially failed because benchmark assets were missing; assets then installed; full run was started and later stopped so user could control run
- **Failure/Issue:** Missing benchmark files/images at first; transient Hugging Face DNS warning for CLIP config; evaluation outputs already existed unless CLEAN_EVAL=1
- **Reason/Lesson:** Evaluation must separate automatic metrics from judge-based benchmarks for fair comparison.
- **Artifacts:** scripts/run_stage5_eval_auto.sh; scripts/setup_stage5_eval_assets.sh; output/eval/stage5_auto_eval_*

### E10 - Move from 1x 48GB instance to 2x/other instances and cleanup
- **Stage:** Vast/storage
- **Experiment:** Move from 1x 48GB instance to 2x/other instances and cleanup
- **Models/Backends:** Infrastructure only
- **Data/Input:** Volume Local-35489770 and later no-volume instance attempt
- **Main settings:** Clean unneeded models/checkpoints; remove models/Qwen-VL-Chat; archive outputs; transfer attempt cancelled
- **Status:** Completed cleanup / cancelled transfer
- **Result/Observed output:** Qwen-VL-Chat removed; old checkpoint/output cleanup performed; later no-volume transfer attempt cancelled and temp transfer files removed
- **Failure/Issue:** No-volume instance required copying 20GB+ assets; user cancelled instance and transfer
- **Reason/Lesson:** Use same Vast volume for fast instance switching; otherwise copy only minimal assets.
- **Artifacts:** No final research artifact; infrastructure state only

### E11 - LLaVA detector training attempt
- **Stage:** Paper Stage 3
- **Experiment:** LLaVA detector training attempt
- **Models/Backends:** LLaVA-1.5-13B detector LoRA path
- **Data/Input:** output/fghd/paper_stage2/detector_train.json
- **Main settings:** 2 epochs; LoRA r=128 alpha=256; LR=1e-5; batch target 16; model_max_length initially 2048/other variants
- **Status:** Failed / abandoned
- **Result/Observed output:** Training failed at first due to import path issue: No module named llava
- **Failure/Issue:** Repo path/module import mismatch for LLaVA training script in this environment
- **Reason/Lesson:** LLaVA detector path required deeper environment fixes; we switched to Qwen2.5-VL detector attempt for implementation stability, then later abandoned local detector path.
- **Artifacts:** scripts/run_paper_stage3_detector_train.sh; output/fghd/paper_stage3/detector_lora partial/none

### E12 - Qwen2.5-VL-7B detector setup/download
- **Stage:** Paper Stage 3
- **Experiment:** Qwen2.5-VL-7B detector setup/download
- **Models/Backends:** Qwen2.5-VL-7B-Instruct
- **Data/Input:** output/fghd/paper_stage2/detector_train.json
- **Main settings:** LoRA detector training; output temporary /tmp/qwen_detector_lora_tmp; MODEL_MAX_LENGTH eventually 1024
- **Status:** Setup completed; training attempt proceeded
- **Result/Observed output:** Model downloaded to models/Qwen2.5-VL-7B-Instruct; initial no-space issue solved by removing old assets and using /tmp output
- **Failure/Issue:** No space left during model download; needed cleanup and external/tmp output location
- **Reason/Lesson:** Large model storage must be planned separately from project volume outputs.
- **Artifacts:** models/Qwen2.5-VL-7B-Instruct; /tmp/qwen_detector_lora_tmp during run

### E13 - Qwen2.5-VL detector training first errors
- **Stage:** Paper Stage 3
- **Experiment:** Qwen2.5-VL detector training first errors
- **Models/Backends:** Qwen2.5-VL-7B-Instruct + LoRA
- **Data/Input:** paper_stage2 detector_train.json
- **Main settings:** Initially used Trainer args including evaluation_strategy; later max length 512 then 1024
- **Status:** Failed then fixed
- **Result/Observed output:** First error: TrainingArguments got unexpected keyword argument evaluation_strategy. Second error: image features and image tokens do not match (tokens 497, features 999). Rerun with MODEL_MAX_LENGTH=1024 proceeded.
- **Failure/Issue:** Transformers version mismatch and image-token truncation mismatch at too-short max length
- **Reason/Lesson:** Version-compatible argument handling and enough sequence length are required for Qwen2.5-VL image tokens.
- **Artifacts:** fg_pipeline/paper/run_stage3_qwen_train.py; scripts/run_paper_stage3_qwen_train.sh before removal

### E14 - Qwen2.5-VL detector inference after training
- **Stage:** Paper Stage 3
- **Experiment:** Qwen2.5-VL detector inference after training
- **Models/Backends:** Qwen2.5-VL-7B detector LoRA
- **Data/Input:** 16143 paper_stage2 rows
- **Main settings:** BACKEND=qwen_detector; MODEL_BASE=models/Qwen2.5-VL-7B-Instruct; MODEL_PATH=paper_stage3/qwen_detector_lora
- **Status:** Failed as detector
- **Result/Observed output:** Stage 3 detect wrote 16143 detection rows but 0 predicted hallucinated; Stage 4 then wrote 0 preference pairs and 0 rewrite records
- **Failure/Issue:** Detector collapsed to non-hallucination or parser/target-format did not recover hallucination outputs; not necessarily because Qwen is too small, more likely prompt/label/parser/calibration mismatch plus no validation split
- **Reason/Lesson:** A detector must be evaluated before using it to filter Stage 4. 0 positives means the downstream preference pipeline fails completely.
- **Artifacts:** output/fghd/paper_stage3/detections.jsonl; detection_stats.json; output/fghd/paper_stage4/preference_pairs.jsonl empty

### E15 - Direct paper-style rewrite using released annotations as detector substitute
- **Stage:** Paper Stage 4
- **Experiment:** Direct paper-style rewrite using released annotations as detector substitute
- **Models/Backends:** LLaVA-1.5-7B rewriter
- **Data/Input:** Released D_FAIF/paper-stage labels instead of failed learned detector
- **Main settings:** Use released annotations as H_i so all hallucinated rows can be rewritten; no API gate
- **Status:** Run started/used as fallback concept
- **Result/Observed output:** Observed progress example: Paper Stage 4 rewrite around 855 rows at 6.46 it/s in one run; exact final result not recorded in chat before plan changed
- **Failure/Issue:** This is not a learned detector reproduction; it uses released annotations directly
- **Reason/Lesson:** Fastest way to produce thousands of pairs when detector training fails, but should be reported as annotation-driven preference construction.
- **Artifacts:** scripts/run_paper_stage4_rewrite.sh; output/fghd/paper_stage4/* when run

### E16 - Released preference judgement with Gemini only
- **Stage:** API Stage 3
- **Experiment:** Released preference judgement with Gemini only
- **Models/Backends:** Gemini 2.5 Flash-Lite
- **Data/Input:** hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl, 8386 rows
- **Main settings:** API_JUDGE=gemini; decision_rule=either in script context; image-root hsa_dpo/data/images
- **Status:** Active at last check
- **Result/Observed output:** Process still running; final output files not yet written at last check
- **Failure/Issue:** No final counts available yet; script appears to write outputs at end
- **Reason/Lesson:** Gemini judgement is useful as a separate audit/filter pipeline, but should be separated from paper-faithful direct training.
- **Artifacts:** output/fghd/released_pref_stage3/* target path

### E17 - Released preference judgement with OpenAI only
- **Stage:** API Stage 3
- **Experiment:** Released preference judgement with OpenAI only
- **Models/Backends:** GPT-4o mini
- **Data/Input:** hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl, 8386 rows
- **Main settings:** API_JUDGE=openai; openai_model=gpt-4o-mini; no API key stored in report
- **Status:** Completed
- **Result/Observed output:** 8386 rows validated: 2914 accepted, 5472 rejected; stats written
- **Failure/Issue:** Cost/time external dependency; results are model-judge dependent and not paper-faithful detector training
- **Reason/Lesson:** Good audit/filtering baseline, but should be reported separately from direct HSA-DPO training.
- **Artifacts:** output/fghd/released_pref_stage3_openai/validated_preferences.jsonl; rejected_for_repair.jsonl; judgement_records.jsonl; stats.json

### E18 - Qwen3-VL-Flash API judge idea
- **Stage:** Qwen API idea
- **Experiment:** Qwen3-VL-Flash API judge idea
- **Models/Backends:** Qwen3-VL-Flash API (planned)
- **Data/Input:** Released preferences / rewritten outputs
- **Main settings:** Would have used Qwen API key as an additional judge/filter
- **Status:** Cancelled before use
- **Result/Observed output:** User decided to remove Qwen API/model usage and stick with current plan
- **Failure/Issue:** Not executed; would add another external dependency and complicate the two-pipeline comparison
- **Reason/Lesson:** Keep final study simpler: direct training baseline plus optional Gemini/OpenAI judgement pipeline.
- **Artifacts:** Qwen API code/env removed from active path

### E19 - Direct paper HSA-DPO from released preference data, first batch16 attempt
- **Stage:** Stage 5 current
- **Experiment:** Direct paper HSA-DPO from released preference data, first batch16 attempt
- **Models/Backends:** LLaVA-1.5-7B + LoRA + DeepSpeed ZeRO-3
- **Data/Input:** hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl; images hsa_dpo/data/images
- **Main settings:** EPOCH=2; TOTAL_BATCH_SIZE=16; DPO_LOSS_TYPE=hsa_weighted; USE_REJECTED_SCORE=True
- **Status:** Interrupted by user choice
- **Result/Observed output:** Reached about 79/1048 steps earlier, then was cancelled to rerun faster settings
- **Failure/Issue:** Long runtime for available time; not a code failure
- **Reason/Lesson:** Batch16 2 epochs is more thorough, but too slow for current deadline.
- **Artifacts:** output/fghd/interrupted_stage5/stage5_paper_hsa_batch16_*.log; exp_direct_paper_hsa_* archive

### E20 - Direct paper HSA-DPO, batch32 epoch1
- **Stage:** Stage 5 current
- **Experiment:** Direct paper HSA-DPO, batch32 epoch1
- **Models/Backends:** LLaVA-1.5-7B + LoRA + DeepSpeed ZeRO-3
- **Data/Input:** hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl; images hsa_dpo/data/images
- **Main settings:** EPOCH=1; TOTAL_BATCH_SIZE=32; BATCH_SIZE=1; gradient_accumulation_steps=32; LoRA r=128 alpha=256; LR=2e-6; beta=0.1; DPO_LOSS_TYPE=hsa_weighted; USE_REJECTED_SCORE=True
- **Status:** Active at last check
- **Result/Observed output:** Latest observed progress: 33/262 steps (13%); latest logged loss at step 30: 22.025; rewards/accuracy 0.5906; ETA about 4h13m at that snapshot
- **Failure/Issue:** Warning: token sequence length 1083 > model_max_length 1024 for at least one row; training continued. This may truncate/affect a small number of long samples.
- **Reason/Lesson:** This is the current main paper-style HSA-DPO run, faster than batch16 2 epochs but fewer optimizer updates.
- **Artifacts:** output/fghd/exp_direct_paper_hsa_b32_e1; output/fghd/stage5_paper_hsa_b32_e1.log

### E21 - Normal DPO baseline, batch32 epoch1
- **Stage:** Stage 5 baseline
- **Experiment:** Normal DPO baseline, batch32 epoch1
- **Models/Backends:** LLaVA-1.5-7B + LoRA + DeepSpeed ZeRO-3
- **Data/Input:** Same released preference data/images as HSA-DPO
- **Main settings:** EPOCH=1; TOTAL_BATCH_SIZE=32; DPO_LOSS_TYPE=standard; USE_REJECTED_SCORE=False; USE_CHOSEN_SCORE=False
- **Status:** Queued
- **Result/Observed output:** Queue active and waiting for HSA-DPO batch32 epoch1 to finish
- **Failure/Issue:** Not started yet due single GPU memory; cannot run concurrently with HSA-DPO
- **Reason/Lesson:** This provides the fair baseline against HSA-DPO using the same model, data, epoch count, and effective batch size.
- **Artifacts:** scripts/run_direct_stage5_normal_dpo_batch32_epoch1.sh; output/fghd/exp_direct_normal_dpo_b32_e1 when complete

### E22 - Our severity-margin HSA-DPO on released preference data
- **Stage:** Stage 5 alternative
- **Experiment:** Our severity-margin HSA-DPO on released preference data
- **Models/Backends:** LLaVA-1.5-7B + severity_margin loss
- **Data/Input:** Released HSA-DPO preference data
- **Main settings:** DPO_LOSS_TYPE=severity_margin; SEVERITY_MARGIN_SCALE=0.5; normalizer=3.0
- **Status:** Planned then not run
- **Result/Observed output:** Script existed but was not queued after user narrowed plan to paper HSA-DPO plus normal DPO
- **Failure/Issue:** Excluded to keep comparison simple and fit time constraints
- **Reason/Lesson:** Useful future ablation, but not part of the current two-experiment setup.
- **Artifacts:** scripts/run_direct_stage5_margin_hsa_batch16.sh earlier; not active

