"""Stage 3 CLI — majority-vote preference validation.

Reads Stage 2 JSONL, runs three verification votes per hallucinated rewrite,
writes an audit JSONL, and exports a clean trainer-compatible preference JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paths import (
    DEFAULT_STAGE3_INPUT,
    DEFAULT_STAGE3_OUTPUT,
    DEFAULT_STAGE3_PREFERENCES,
    DEFAULT_STAGE3_STATS,
)
from fg_pipeline.schemas import PreferenceCleanRecord
from fg_pipeline.stage3.backends import (
    APPROVALS_REQUIRED,
    VOTE_COUNT,
    VerificationError,
    evaluate_votes,
    get_backend,
)
from fg_pipeline.stage3.schemas import Stage3Record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: majority-vote validation of Stage 2 rewrites.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STAGE3_INPUT,
        help="Path to Stage 2 JSONL (default: default Stage 2 output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE3_OUTPUT,
        help="Path to Stage 3 audit JSONL.",
    )
    parser.add_argument(
        "--preferences-out",
        type=Path,
        default=DEFAULT_STAGE3_PREFERENCES,
        help="Path to trainer-compatible preference JSONL.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=DEFAULT_STAGE3_STATS,
        help="Path to Stage 3 stats JSON.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="heuristic",
        help="Verification backend (default: heuristic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N Stage 2 rows.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed Stage 2 rows or verification errors.",
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default=None,
        help="Local Qwen-VL-Chat path for the qwen_llava_ensemble backend.",
    )
    parser.add_argument(
        "--llava-model-path",
        type=str,
        default=None,
        help="Local LLaVA path for the qwen_llava_ensemble backend.",
    )
    parser.add_argument(
        "--llava-model-base",
        type=str,
        default=None,
        help="Base model path when the LLaVA judge path is a LoRA adapter.",
    )
    parser.add_argument(
        "--llava-conv-mode",
        type=str,
        default="vicuna_v1",
        help="Conversation template for the LLaVA judge backend.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=".",
        help="Root directory used to resolve relative image paths for local judges.",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for each Qwen judge vote.",
    )
    parser.add_argument(
        "--llava-max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for each LLaVA judge vote.",
    )
    return parser


class _Stats:
    def __init__(self, backend_name: str, *, policy_version: str, approval_families_required: tuple[str, ...]) -> None:
        self.total_input_rows = 0
        self.vote_rows_processed = 0
        self.preference_pairs_emitted = 0
        self.dropped_rows = 0
        self.backend = backend_name
        self.vote_policy_version = policy_version
        self.approval_families_required = list(approval_families_required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_rows": self.total_input_rows,
            "vote_rows_processed": self.vote_rows_processed,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "dropped_rows": self.dropped_rows,
            "backend": self.backend,
            "vote_count": VOTE_COUNT,
            "approvals_required": APPROVALS_REQUIRED,
            "vote_policy_version": self.vote_policy_version,
            "approval_families_required": self.approval_families_required,
        }


def _aggregate_severity(critiques: list[dict[str, Any]]) -> float:
    scores = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        score = critique.get("severity_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return 1.0
    return float(mean(scores))


def _validation_warnings(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not (row.get("original_response") or "").strip():
        warnings.append("missing original_response")
    if not (row.get("rewrite_response") or "").strip():
        warnings.append("missing rewrite_response")
    critiques = row.get("critiques")
    if not isinstance(critiques, list):
        warnings.append("critiques is not a list")
    return warnings


def _process_rows(
    backend,
    rows: Iterable[dict[str, Any]],
    *,
    strict: bool,
    limit: int | None,
    total: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], _Stats]:
    stats = _Stats(
        backend_name=backend.name,
        policy_version=backend.policy_version,
        approval_families_required=getattr(backend, "approval_families_required", ()),
    )
    audit_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []

    progress = maybe_tqdm(rows, desc="Stage 3", total=total)
    for index, row in enumerate(progress):
        if limit is not None and index >= limit:
            break
        stats.total_input_rows += 1

        warnings = _validation_warnings(row)
        if warnings and strict:
            raise VerificationError(f"row id={row.get('id')!r} invalid for Stage 3: {warnings[0]}")

        critiques = list(row.get("critiques") or [])
        votes = []
        if not warnings:
            for vote_index in range(1, VOTE_COUNT + 1):
                votes.append(backend.vote(row, vote_index=vote_index, strict=strict))

        approvals = sum(1 for vote in votes if vote.approved)
        passed_majority, vote_policy = evaluate_votes(backend, votes)
        severity_score = _aggregate_severity(critiques)

        metadata: dict[str, Any] = {
            "source_stage": "stage3_verification",
            "backend": backend.name,
            "vote_count": VOTE_COUNT,
            "approvals_required": APPROVALS_REQUIRED,
            "vote_policy_version": backend.policy_version,
            "question_source": (
                "stage1_assessed_sentence"
                if row.get("question") == row.get("original_response")
                else "upstream_question"
            ),
        }
        metadata.update(vote_policy)
        if warnings:
            metadata["validation_warnings"] = warnings

        chosen = row.get("rewrite_response") if passed_majority else None
        rejected = row.get("original_response") if passed_majority else None

        audit_rows.append(
            Stage3Record(
                id=row.get("id"),
                image=row.get("image"),
                question=row.get("question", ""),
                original_response=row.get("original_response", ""),
                rewrite_response=row.get("rewrite_response", ""),
                critiques=critiques,
                votes=votes,
                approvals=approvals,
                rejections=VOTE_COUNT - approvals,
                passed_majority=passed_majority,
                response_severity_score=severity_score,
                chosen=chosen,
                rejected=rejected,
                metadata=metadata,
            ).to_dict()
        )
        stats.vote_rows_processed += 1

        if passed_majority:
            preference_rows.append(
                PreferenceCleanRecord(
                    id=row.get("id"),
                    question=row.get("question") or row.get("original_response", ""),
                    chosen=row.get("rewrite_response", ""),
                    rejected=row.get("original_response", ""),
                    chosen_score=1.0,
                    rejected_score=severity_score,
                    image=row.get("image"),
                    metadata={
                        "source_stage": "stage3_preference",
                        "verification_backend": backend.name,
                        "approvals": approvals,
                        "vote_count": VOTE_COUNT,
                        "approvals_required": APPROVALS_REQUIRED,
                        "response_severity_score": severity_score,
                        "vote_policy_version": backend.policy_version,
                        "approved_families": vote_policy.get("approved_families"),
                        "approval_families_required": vote_policy.get("approval_families_required"),
                        "question_source": metadata["question_source"],
                    },
                ).to_dict()
            )
            stats.preference_pairs_emitted += 1
        else:
            stats.dropped_rows += 1

    return audit_rows, preference_rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 3 input not found: {args.input}", file=sys.stderr)
        print("Run Stage 2 first:  bash scripts/run_stage2_rewrites.sh", file=sys.stderr)
        return 2

    try:
        backend = get_backend(
            args.backend,
            qwen_model_path=args.qwen_model_path,
            llava_model_path=args.llava_model_path,
            llava_model_base=args.llava_model_base,
            llava_conv_mode=args.llava_conv_mode,
            image_root=args.image_root,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            llava_max_new_tokens=args.llava_max_new_tokens,
        )
    except ValueError as exc:
        print(f"Stage 3 backend error: {exc}", file=sys.stderr)
        return 2

    try:
        total_rows = count_jsonl_rows(args.input)
        if args.limit is not None:
            total_rows = min(total_rows, args.limit)

        audit_rows, preference_rows, stats = _process_rows(
            backend,
            read_jsonl(args.input),
            strict=args.strict,
            limit=args.limit,
            total=total_rows,
        )
    except VerificationError as exc:
        print(f"Stage 3 verification failed: {exc}", file=sys.stderr)
        return 3

    write_jsonl(args.output, audit_rows)
    write_jsonl(args.preferences_out, preference_rows)

    stats_out = ensure_parent_dir(args.stats_out)
    with stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Stage 3 wrote {stats.preference_pairs_emitted} preference pair(s) "
        f"and {stats.vote_rows_processed} audit row(s) -> {args.output}"
    )
    print(f"Stage 3 preferences -> {args.preferences_out}")
    print(f"Stage 3 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
