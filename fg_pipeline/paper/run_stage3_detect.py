"""Paper Stage 3: run hallucination detector inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION, aggregate_severity
from fg_pipeline.paths import (
    DEFAULT_DETECTION_INPUT,
    DEFAULT_PAPER_STAGE3_DETECTIONS,
    DEFAULT_PAPER_STAGE3_STATS,
)
from fg_pipeline.stage1.backends import get_backend
from fg_pipeline.stage1.parser import ParseError
from fg_pipeline.stage1.prompts import PROMPT_VERSION as DETECTOR_PROMPT_VERSION


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper Stage 3: apply local hallucination detector and emit H_i.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DETECTION_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAPER_STAGE3_DETECTIONS)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_PAPER_STAGE3_STATS)
    parser.add_argument("--backend", type=str, default="llava_detector")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument(
        "--api-judge",
        type=str,
        default="gemini_openai",
        help="Judge family for --backend api_judge: gemini, openai, or gemini_openai.",
    )
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-max-output-tokens", type=int, default=None)
    parser.add_argument("--judge-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument(
        "--api-decision-rule",
        type=str,
        default="either",
        choices=("either", "both"),
        help="For multiple API judges, flag hallucination if either judge finds it or only if both do.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


class _Stats:
    def __init__(self, *, backend_name: str, input_path: str) -> None:
        self.backend = backend_name
        self.input_path = input_path
        self.total_rows = 0
        self.predicted_hallucinated_rows = 0
        self.predicted_non_hallucinated_rows = 0
        self.critique_items = 0

    def record(self, row: dict[str, Any]) -> None:
        self.total_rows += 1
        if row.get("is_hallucinated_pred"):
            self.predicted_hallucinated_rows += 1
        else:
            self.predicted_non_hallucinated_rows += 1
        self.critique_items += len(row.get("detected_critiques") or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "backend": self.backend,
            "input_path": self.input_path,
            "total_rows": self.total_rows,
            "predicted_hallucinated_rows": self.predicted_hallucinated_rows,
            "predicted_non_hallucinated_rows": self.predicted_non_hallucinated_rows,
            "critique_items": self.critique_items,
            "detector_prompt_version": DETECTOR_PROMPT_VERSION,
        }


def _row_from_detection(record: Any, *, backend_name: str, model_path: str | None) -> dict[str, Any]:
    rec = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    critiques = list(rec.get("critiques") or [])
    metadata = dict(rec.get("metadata") or {})
    metadata.update(
        {
            "source_stage": "paper_stage3_detection",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "detector_backend": backend_name,
            "detector_model": model_path or backend_name,
            "detector_prompt_version": DETECTOR_PROMPT_VERSION,
        }
    )
    return {
        "id": rec.get("id"),
        "image": rec.get("image"),
        "question": rec.get("question", ""),
        "original_response": rec.get("response_text", ""),
        "is_hallucinated_pred": bool(rec.get("is_hallucinated")),
        "detected_critiques": critiques,
        "response_severity_score": aggregate_severity(critiques),
        "metadata": metadata,
    }


def _iter_detections(
    backend: Any,
    rows: Iterable[dict[str, Any]],
    *,
    strict: bool,
    limit: int | None,
    total: int | None,
    model_path: str | None,
    stats: _Stats,
) -> Iterable[dict[str, Any]]:
    progress = maybe_tqdm(rows, desc="Paper Stage 3 detect", total=total)
    for index, row in enumerate(progress):
        if limit is not None and index >= limit:
            break
        result = backend.detect(row, strict=strict)
        output_row = _row_from_detection(result.record, backend_name=backend.name, model_path=model_path)
        stats.record(output_row)
        yield output_row


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Paper Stage 3 input not found: {args.input}", file=sys.stderr)
        return 2

    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "conv_mode": args.conv_mode,
        "image_root": args.image_root,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "api_judge": args.api_judge,
        "gemini_model": args.gemini_model,
        "openai_model": args.openai_model,
        "judge_max_output_tokens": args.judge_max_output_tokens,
        "judge_timeout_seconds": args.judge_timeout_seconds,
        "judge_retries": args.judge_retries,
        "api_decision_rule": args.api_decision_rule,
    }
    try:
        backend = get_backend(args.backend, **kwargs)
    except (ValueError, ImportError, FileNotFoundError, OSError) as exc:
        print(f"Paper Stage 3 backend error: {exc}", file=sys.stderr)
        return 2

    total = count_jsonl_rows(args.input)
    if args.limit is not None:
        total = min(total, args.limit)
    stats = _Stats(backend_name=backend.name, input_path=str(args.input))
    try:
        write_jsonl(
            ensure_parent_dir(args.output),
            _iter_detections(
                backend,
                read_jsonl(args.input),
                strict=args.strict,
                limit=args.limit,
                total=total,
                model_path=args.model_path,
                stats=stats,
            ),
        )
    except ParseError as exc:
        print(f"Paper Stage 3 strict detection parse failed: {exc}", file=sys.stderr)
        return 3

    payload = stats.to_dict()
    payload["output_path"] = str(args.output)
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Paper Stage 3 wrote {payload['total_rows']} detection row(s) "
        f"({payload['predicted_hallucinated_rows']} predicted hallucinated) -> {args.output}"
    )
    print(f"Paper Stage 3 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
