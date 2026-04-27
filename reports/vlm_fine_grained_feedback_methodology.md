# VLM Fine-Grained Feedback Methodology Note

No API keys are included in this document.

## Purpose

This note explains how vision-language models (VLMs) were used or considered in our hallucination mitigation experiments, why the paper-aligned prompts matter, and how the current experiments should be reported to the instructor.

## Paper-Aligned Fine-Grained Feedback

The referenced HSA-DPO method uses strong VLM/API models to create fine-grained AI feedback before mitigation training. In the paper workflow, the initial fine-grained annotation stage uses:

- GPT-4V for visual complex reasoning style examples, where the model must inspect the image-question-response content.
- GPT-4 with Visual Genome object/region annotations for detailed description generation examples, where structured scene annotations support more reliable hallucination identification.
- An open-source LVLM detector trained from the fine-grained AI feedback.
- LLaVA as the rewrite model that turns hallucinated responses into preferred corrected responses.
- HSA-DPO for final preference optimization, where the rejected response is penalized using hallucination severity.

Our reproducible budget path does not regenerate all GPT-4/GPT-4V annotations. Instead, Stage 1 uses the released HSA-DPO fine-grained annotation file as the main supervision source. This is important because those released annotations already encode the paper's fine-grained AI feedback: hallucination type, rationale, severity, and severity rationale.

## Prompt Alignment Requirement

For paper-faithful runs, the prompts should stay aligned with the referenced paper appendix rather than being replaced by ad-hoc prompts. The code stores prompt versions in `fg_pipeline/paper/prompts.py`:

- `paper_ddg_v1`: detailed description generation annotation prompt.
- `paper_vcr_v1`: visual complex reasoning annotation prompt.
- `paper_detector_v1`: local detector prompt/target format.
- `paper_rewrite_v1`: detect-then-rewrite prompt.
- `paper_severity_rubric_v1`: Minor/Moderate/Major severity rubric.
- `api_critic_feedback_v1` and `api_critic_revision_v1`: later experimental API-critic revision prompts.

For strict reproduction, these templates should be checked against the appendix wording before a final run. The main paper-style path should use the appendix-aligned detector/rewrite prompts and should record prompt versions in output metadata.

## Why Use VLMs

The task is image-grounded hallucination mitigation. A text-only model can judge fluency, but it cannot reliably verify whether an object, attribute, count, relation, action, or spatial claim is supported by the image. VLMs are therefore used for three roles:

- Fine-grained feedback: identify what visual claim is wrong, why it is wrong, and how severe it is.
- Rewriting: produce a corrected answer that preserves supported visual content and removes unsupported claims.
- Mitigation training: train an LVLM with preference pairs so the final model learns to prefer visually grounded responses.

This is why using the same model family as the referenced paper is valuable. LLaVA-based rewriting and LLaVA-based DPO training make the comparison closer to the paper's mitigation setup. In our actual runs, LLaVA-1.5-7B was used for the current direct HSA-DPO and normal DPO experiments because the 13B path was slower and more memory/storage constrained on the available Vast instance.

## Experiments And Lessons

The main experiments from this chat should be described as follows:

- Released fine-grained annotation parsing succeeded: 16,143 records were parsed, including 7,643 hallucinated and 8,500 non-hallucinated rows.
- Detector dataset construction succeeded: 16,143 detector examples were built from released fine-grained feedback.
- Qwen2.5-VL-7B local detector training was attempted, but inference predicted 0 hallucinated rows. This made Stage 4 produce 0 preference pairs. The likely issue is detector target/prompt/parser collapse or poor calibration for the trained detector path, not proof that Qwen is inherently unusable.
- LLaVA-13B training/evaluation paths were attempted but were too heavy for the available time/VRAM/storage constraints.
- Gemini and GPT-4o mini judgement were used as external audit/filter experiments, not as the final paper-faithful detector path.
- GPT-4o mini judgement over released preferences completed with 8,386 rows: 2,914 accepted and 5,472 rejected.
- Strict validation produced too few training pairs in earlier runs, which is why the final direct experiments use the released preference data directly.
- Current final comparison is simplified to two Stage 5 experiments on the same released preference data: HSA-DPO and normal DPO.

## Reporting Rule

The instructor report should separate three ideas:

- Paper-faithful supervision source: released fine-grained AI feedback from the referenced HSA-DPO dataset.
- Our failed/diagnostic experiments: Qwen detector, LLaVA-13B constraints, API judgement filtering, and evaluation setup issues.
- Current final training comparison: HSA-DPO versus normal DPO using the same released preference data, same LLaVA-1.5-7B base, same image folder, same batch/epoch settings, and different DPO objectives.

Do not claim direct superiority over the referenced paper unless the same benchmark assets, model size, and official evaluation protocol are used. If the protocol differs, label results as a supplemental local comparison.
