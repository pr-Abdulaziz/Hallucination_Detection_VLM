"""Stage 1 CLI — extract critiques from released detection supervision.

Usage::

    python -m fg_pipeline.stage1.run_stage1 \
        --input fg_pipeline/data/hsa_dpo_detection.jsonl \
        --output output/fghd/stage1/detection_critiques.jsonl \
        --stats-out output/fghd/stage1/stats.json

The CLI writes a normalized Stage 1 JSONL plus a compact stats JSON. It never
calls a hosted service. It can either parse released supervision via
:class:`ReleasedAnnotationBackend` or run a local LLaVA detector backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paths import (
    DEFAULT_STAGE1_INPUT,
    DEFAULT_STAGE1_OUTPUT,
    DEFAULT_STAGE1_STATS,
)
from fg_pipeline.stage1.backends import get_backend
from fg_pipeline.stage1.parser import ParseError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: extract critiques from released fine-grained supervision.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STAGE1_INPUT,
        help="Path to input detection JSONL (default: mirrored released file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE1_OUTPUT,
        help="Path to output Stage 1 JSONL.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=DEFAULT_STAGE1_STATS,
        help="Path to output Stage 1 stats JSON.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="released_annotations",
        help="Stage 1 backend to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows (smoke runs).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed hallucinated rows instead of recording warnings.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path used by llava_detector.",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Base model path when --model-path is a LoRA adapter.",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="vicuna_v1",
        help="Conversation template for llava_detector.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=".",
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Max tokens to generate for llava_detector annotations.",
    )
    return parser


def _iter_records(
    backend,
    rows: Iterable[dict[str, Any]],
    *,
    strict: bool,
    limit: int | None,
    total: int | None,
) -> Iterable[dict[str, Any]]:
    progress = maybe_tqdm(rows, desc="Stage 1", total=total)
    for i, row in enumerate(progress):
        if limit is not None and i >= limit:
            break
        result = backend.detect(row, strict=strict)
        yield result.record.to_dict()


def _compute_stats(output_path: Path) -> dict[str, Any]:
    total = 0
    hallucinated = 0
    non_hallucinated = 0
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("is_hallucinated"):
                hallucinated += 1
            else:
                non_hallucinated += 1
            for critique in rec.get("critiques") or []:
                by_type[critique.get("hallucination_type", "unknown")] += 1
                by_severity[critique.get("severity_label", "unknown")] += 1

    return {
        "total_rows": total,
        "hallucinated_rows": hallucinated,
        "non_hallucinated_rows": non_hallucinated,
        "critique_count_by_type": dict(by_type),
        "critique_count_by_severity": dict(by_severity),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 1 input not found: {args.input}", file=sys.stderr)
        return 2

    try:
        backend = get_backend(
            args.backend,
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            image_root=args.image_root,
            max_new_tokens=args.max_new_tokens,
        )
    except ValueError as exc:
        print(f"Stage 1 backend error: {exc}", file=sys.stderr)
        return 2

    total_rows = count_jsonl_rows(args.input)
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    rows_iter = read_jsonl(args.input)
    ensure_parent_dir(args.output)
    try:
        write_jsonl(
            args.output,
            _iter_records(
                backend,
                rows_iter,
                strict=args.strict,
                limit=args.limit,
                total=total_rows,
            ),
        )
    except ParseError as exc:
        print(f"Stage 1 strict parse failed: {exc}", file=sys.stderr)
        return 3

    stats = _compute_stats(args.output)
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Stage 1 wrote {stats['total_rows']} records "
        f"({stats['hallucinated_rows']} hallucinated, "
        f"{stats['non_hallucinated_rows']} non-hallucinated) -> {args.output}"
    )
    print(f"Stage 1 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
