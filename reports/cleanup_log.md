# Cleanup Log

No project training outputs were deleted.

## 2026-04-26 Local Cleanup

Archived generated/local-only files to:

`old_outputs/cleanup_20260426_150211/`

Moved items:

- `.pytest_cache/`
- `vast_eval_train_patch.tar`
- `vast_stage5_zero3_fix.tar`
- `repo.bundle`
- `output/hsa_dpo_llava/`

Kept in place:

- `output/fghd/`, because it contains project experiment outputs referenced by the report.
- Active Vast training outputs, because training/judgement jobs were still running and should not be interrupted.

`.gitignore` now ignores `old_outputs/` so archived local files do not enter normal code changes.
