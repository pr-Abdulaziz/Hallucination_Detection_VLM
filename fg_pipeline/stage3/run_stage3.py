"""Stage 3 CLI — preference validation.

Reads Stage 2 JSONL, runs verification votes per hallucinated rewrite, writes
an audit JSONL, and exports a clean trainer-compatible preference JSONL.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl
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
from fg_pipeline.stage3.schemas import Stage3Record, VoteDecision


CHECKPOINT_EVERY_DEFAULT = 50


def _backend_vote_count(backend: Any) -> int:
    return int(getattr(backend, "vote_count", VOTE_COUNT))


def _backend_approvals_required(backend: Any) -> int:
    return int(getattr(backend, "approvals_required", APPROVALS_REQUIRED))


def _backend_supports_row_parallelism(backend: Any) -> bool:
    return bool(getattr(backend, "supports_row_parallelism", False))


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
        "--resume",
        action="store_true",
        help="Resume from an existing Stage 3 audit file with matching metadata.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
        help="Write stats.json every N processed rows (default: 50).",
    )
    parser.add_argument(
        "--row-workers",
        type=int,
        default=1,
        help="Concurrent row workers for hosted/stateless Stage 3 backends.",
    )
    parser.add_argument(
        "--llava-model-path",
        type=str,
        default=None,
        help="Local LLaVA path for the gemini_llava_two_vote backend.",
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
        "--llava-max-new-tokens",
        type=int,
        default=64,
        help="Max tokens for each LLaVA judge vote.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for the gemini_llava_two_vote backend.",
    )
    parser.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=64,
        help="Max output tokens for each Gemini judge vote.",
    )
    parser.add_argument(
        "--llava-device",
        type=str,
        default=None,
        help="Optional device for LLaVA judge, e.g. cuda:0.",
    )
    return parser


class _Stats:
    def __init__(
        self,
        backend_name: str,
        *,
        policy_version: str,
        approval_families_required: tuple[str, ...],
        vote_count: int,
        approvals_required: int,
        input_path: str,
    ) -> None:
        self.total_input_rows = 0
        self.vote_rows_processed = 0
        self.preference_pairs_emitted = 0
        self.dropped_rows = 0
        self.backend = backend_name
        self.vote_count = vote_count
        self.approvals_required = approvals_required
        self.vote_policy_version = policy_version
        self.approval_families_required = list(approval_families_required)
        self.input_path = input_path

    def record_audit_row(self, audit_row: dict[str, Any]) -> None:
        self.total_input_rows += 1
        self.vote_rows_processed += 1
        if audit_row.get("passed_majority"):
            self.preference_pairs_emitted += 1
        else:
            self.dropped_rows += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_rows": self.total_input_rows,
            "vote_rows_processed": self.vote_rows_processed,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "dropped_rows": self.dropped_rows,
            "backend": self.backend,
            "vote_count": self.vote_count,
            "approvals_required": self.approvals_required,
            "vote_policy_version": self.vote_policy_version,
            "approval_families_required": self.approval_families_required,
            "input_path": self.input_path,
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


def _build_votes(backend: Any, row: dict[str, Any], *, strict: bool) -> tuple[list[VoteDecision], bool]:
    if getattr(backend, "name", "") in {"gemini_llava_two_vote", "gemini_two_vote"}:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                vote_index: executor.submit(backend.vote, row, vote_index=vote_index, strict=strict)
                for vote_index in (1, 2)
            }
            return [futures[index].result() for index in (1, 2)], False

    votes: list[VoteDecision] = []
    early_stop_applied = False
    for vote_index in range(1, _backend_vote_count(backend) + 1):
        votes.append(backend.vote(row, vote_index=vote_index, strict=strict))
    return votes, early_stop_applied


def _build_row_outputs(
    backend: Any,
    row: dict[str, Any],
    *,
    strict: bool,
    input_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    warnings = _validation_warnings(row)
    if warnings and strict:
        raise VerificationError(f"row id={row.get('id')!r} invalid for Stage 3: {warnings[0]}")

    critiques = list(row.get("critiques") or [])
    votes: list[VoteDecision] = []
    early_stop_applied = False
    if not warnings:
        votes, early_stop_applied = _build_votes(backend, row, strict=strict)

    approvals = sum(1 for vote in votes if vote.approved)
    rejections = sum(1 for vote in votes if not vote.approved)
    passed_majority, vote_policy = evaluate_votes(backend, votes)
    severity_score = _aggregate_severity(critiques)

    metadata: dict[str, Any] = {
        "source_stage": "stage3_verification",
        "backend": backend.name,
        "vote_count": _backend_vote_count(backend),
        "approvals_required": _backend_approvals_required(backend),
        "vote_policy_version": backend.policy_version,
        "executed_vote_count": len(votes),
        "early_stop_applied": early_stop_applied,
        "question_source": (
            "stage1_assessed_sentence"
            if row.get("question") == row.get("original_response")
            else "upstream_question"
        ),
    }
    if input_path is not None:
        metadata["input_path"] = input_path
    metadata.update(vote_policy)
    if warnings:
        metadata["validation_warnings"] = warnings

    chosen = row.get("rewrite_response") if passed_majority else None
    rejected = row.get("original_response") if passed_majority else None
    audit_row = Stage3Record(
        id=row.get("id"),
        image=row.get("image"),
        question=row.get("question", ""),
        original_response=row.get("original_response", ""),
        rewrite_response=row.get("rewrite_response", ""),
        critiques=critiques,
        votes=votes,
        approvals=approvals,
        rejections=rejections,
        passed_majority=passed_majority,
        response_severity_score=severity_score,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata,
    ).to_dict()

    if not passed_majority:
        return audit_row, None

    preference_row = PreferenceCleanRecord(
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
            "vote_count": _backend_vote_count(backend),
            "approvals_required": _backend_approvals_required(backend),
            "executed_vote_count": len(votes),
            "early_stop_applied": early_stop_applied,
            "response_severity_score": severity_score,
            "vote_policy_version": backend.policy_version,
            "approved_families": vote_policy.get("approved_families"),
            "approval_families_required": vote_policy.get("approval_families_required"),
            "question_source": metadata["question_source"],
        },
    ).to_dict()
    return audit_row, preference_row


def _process_rows(
    backend: Any,
    rows: Iterable[dict[str, Any]],
    *,
    strict: bool,
    limit: int | None,
    total: int | None = None,
    input_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], _Stats]:
    stats = _Stats(
        backend_name=backend.name,
        policy_version=backend.policy_version,
        approval_families_required=getattr(backend, "approval_families_required", ()),
        vote_count=_backend_vote_count(backend),
        approvals_required=_backend_approvals_required(backend),
        input_path=input_path or "<memory>",
    )
    audit_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []

    progress = maybe_tqdm(rows, desc="Stage 3", total=total)
    for index, row in enumerate(progress):
        if limit is not None and index >= limit:
            break
        audit_row, preference_row = _build_row_outputs(
            backend,
            row,
            strict=strict,
            input_path=input_path,
        )
        audit_rows.append(audit_row)
        if preference_row is not None:
            preference_rows.append(preference_row)
        stats.record_audit_row(audit_row)

    return audit_rows, preference_rows, stats


def _append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False))
        handle.write("\n")


def _reset_output_file(path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8", newline="\n"):
        pass


def _write_stats(path: str | Path, stats: _Stats) -> None:
    stats_out = ensure_parent_dir(path)
    with stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_existing_audit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _build_resume_state(
    *,
    audit_path: Path,
    preferences_path: Path,
    stats_path: Path,
    backend: Any,
    input_path: str,
) -> tuple[set[int | str], _Stats]:
    if not audit_path.exists():
        if preferences_path.exists() or stats_path.exists():
            raise VerificationError("resume requires an existing Stage 3 audit file")
        return set(), _Stats(
            backend_name=backend.name,
            policy_version=backend.policy_version,
            approval_families_required=getattr(backend, "approval_families_required", ()),
            vote_count=_backend_vote_count(backend),
            approvals_required=_backend_approvals_required(backend),
            input_path=input_path,
        )

    audit_rows = _load_existing_audit_rows(audit_path)
    processed_ids: set[int | str] = set()
    expected_preference_ids: set[int | str] = set()
    stats = _Stats(
        backend_name=backend.name,
        policy_version=backend.policy_version,
        approval_families_required=getattr(backend, "approval_families_required", ()),
        vote_count=_backend_vote_count(backend),
        approvals_required=_backend_approvals_required(backend),
        input_path=input_path,
    )

    for row in audit_rows:
        row_id = row.get("id")
        if row_id in processed_ids:
            raise VerificationError(f"resume found duplicate Stage 3 audit id={row_id!r}")
        processed_ids.add(row_id)

        metadata = row.get("metadata") or {}
        previous_backend = metadata.get("backend")
        previous_input_path = metadata.get("input_path")
        if previous_backend != backend.name:
            raise VerificationError(
                f"resume backend mismatch: existing={previous_backend!r}, current={backend.name!r}"
            )
        if previous_input_path != input_path:
            raise VerificationError(
                "resume input mismatch: "
                f"existing={previous_input_path!r}, current={input_path!r}"
            )

        if row.get("passed_majority"):
            expected_preference_ids.add(row_id)
        stats.record_audit_row(row)

    existing_preference_ids: set[int | str] = set()
    if preferences_path.exists():
        for row in read_jsonl(preferences_path):
            row_id = row.get("id")
            if row_id in existing_preference_ids:
                raise VerificationError(f"resume found duplicate Stage 3 preference id={row_id!r}")
            existing_preference_ids.add(row_id)

    if expected_preference_ids != existing_preference_ids:
        raise VerificationError("resume requires preference output to match existing audit output")

    return processed_ids, stats


def _iter_unprocessed_rows(
    rows: Iterable[dict[str, Any]],
    processed_ids: set[int | str],
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        if row.get("id") in processed_ids:
            continue
        yield row


def _process_one_row_task(args: tuple[Any, dict[str, Any], bool, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    backend, row, strict, input_path = args
    return _build_row_outputs(
        backend,
        row,
        strict=strict,
        input_path=input_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 3 input not found: {args.input}", file=sys.stderr)
        print("Run Stage 2 first:  bash scripts/run_stage2_rewrites.sh", file=sys.stderr)
        return 2
    if args.checkpoint_every < 1:
        print("Stage 3 backend error: --checkpoint-every must be >= 1", file=sys.stderr)
        return 2
    if args.row_workers < 1:
        print("Stage 3 backend error: --row-workers must be >= 1", file=sys.stderr)
        return 2

    resolved_input_path = str(args.input.resolve())
    try:
        backend = get_backend(
            args.backend,
            llava_model_path=args.llava_model_path,
            llava_model_base=args.llava_model_base,
            llava_conv_mode=args.llava_conv_mode,
            image_root=args.image_root,
            llava_max_new_tokens=args.llava_max_new_tokens,
            gemini_model=args.gemini_model,
            gemini_max_output_tokens=args.gemini_max_output_tokens,
            llava_device=args.llava_device,
        )
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"Stage 3 backend error: {exc}", file=sys.stderr)
        return 2
    if args.row_workers > 1 and not _backend_supports_row_parallelism(backend):
        print(
            f"Stage 3 backend error: --row-workers > 1 is not safe for backend {backend.name!r}",
            file=sys.stderr,
        )
        return 2

    try:
        if args.resume:
            processed_ids, stats = _build_resume_state(
                audit_path=args.output,
                preferences_path=args.preferences_out,
                stats_path=args.stats_out,
                backend=backend,
                input_path=resolved_input_path,
            )
        else:
            processed_ids = set()
            stats = _Stats(
                backend_name=backend.name,
                policy_version=backend.policy_version,
                approval_families_required=getattr(backend, "approval_families_required", ()),
                vote_count=_backend_vote_count(backend),
                approvals_required=_backend_approvals_required(backend),
                input_path=resolved_input_path,
            )
            _reset_output_file(args.output)
            _reset_output_file(args.preferences_out)
    except VerificationError as exc:
        print(f"Stage 3 verification failed: {exc}", file=sys.stderr)
        return 3

    try:
        total_rows = count_jsonl_rows(args.input)
        if args.limit is not None:
            total_rows = min(total_rows, args.limit)
        remaining_total = max(total_rows - len(processed_ids), 0)

        progress = maybe_tqdm(
            _iter_unprocessed_rows(read_jsonl(args.input), processed_ids, limit=args.limit),
            desc="Stage 3",
            total=remaining_total,
        )
        since_checkpoint = 0
        if args.row_workers == 1:
            for row in progress:
                audit_row, preference_row = _build_row_outputs(
                    backend,
                    row,
                    strict=args.strict,
                    input_path=resolved_input_path,
                )
                _append_jsonl_row(args.output, audit_row)
                if preference_row is not None:
                    _append_jsonl_row(args.preferences_out, preference_row)
                stats.record_audit_row(audit_row)
                since_checkpoint += 1
                if since_checkpoint >= args.checkpoint_every:
                    _write_stats(args.stats_out, stats)
                    since_checkpoint = 0
        else:
            task_args = (
                (backend, row, args.strict, resolved_input_path)
                for row in progress
            )
            with ThreadPoolExecutor(max_workers=args.row_workers) as executor:
                futures = [executor.submit(_process_one_row_task, item) for item in task_args]
                for future in maybe_tqdm(as_completed(futures), desc="Stage 3 write", total=len(futures)):
                    audit_row, preference_row = future.result()
                    _append_jsonl_row(args.output, audit_row)
                    if preference_row is not None:
                        _append_jsonl_row(args.preferences_out, preference_row)
                    stats.record_audit_row(audit_row)
                    since_checkpoint += 1
                    if since_checkpoint >= args.checkpoint_every:
                        _write_stats(args.stats_out, stats)
                        since_checkpoint = 0
    except VerificationError as exc:
        print(f"Stage 3 verification failed: {exc}", file=sys.stderr)
        return 3

    _write_stats(args.stats_out, stats)
    print(
        f"Stage 3 wrote {stats.preference_pairs_emitted} preference pair(s) "
        f"and {stats.vote_rows_processed} audit row(s) -> {args.output}"
    )
    print(f"Stage 3 preferences -> {args.preferences_out}")
    print(f"Stage 3 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
