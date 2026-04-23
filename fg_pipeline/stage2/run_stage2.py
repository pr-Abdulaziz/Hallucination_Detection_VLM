"""Stage 2 CLI — critique-guided rewrite of hallucinated responses.

Usage::

    python -m fg_pipeline.stage2.run_stage2 \\
        --input  output/fghd/stage1/detection_critiques.jsonl \\
        --output output/fghd/stage2/rewrites.jsonl \\
        --stats-out output/fghd/stage2/stats.json

    # smoke run on a small input with the default template backend
    python -m fg_pipeline.stage2.run_stage2 --limit 20

    # run with the LLaVA backend (requires GPU and local model)
    python -m fg_pipeline.stage2.run_stage2 \\
        --backend llava \\
        --model-path models/llava-v1.5-13b

The CLI reads Stage 1 JSONL, skips non-hallucinated rows, calls the
selected rewrite backend for hallucinated rows, and writes normalized
Stage 2 JSONL plus compact stats JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paths import (
    DEFAULT_STAGE2_INPUT,
    DEFAULT_STAGE2_OUTPUT,
    DEFAULT_STAGE2_STATS,
)
from fg_pipeline.stage2.backends import RewriteError, get_backend
from fg_pipeline.stage2.prompts import PROMPT_VERSION
from fg_pipeline.stage2.schemas import Stage2Record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: critique-guided rewrite of hallucinated responses.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STAGE2_INPUT,
        help="Path to Stage 1 JSONL (default: default Stage 1 output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE2_OUTPUT,
        help="Path to output Stage 2 JSONL.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=DEFAULT_STAGE2_STATS,
        help="Path to output Stage 2 stats JSON.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="template",
        help="Rewrite backend: 'template' (smoke) or 'llava' (real).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N input rows (smoke runs).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on backend errors instead of recording warnings.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to LLaVA model weights (required for --backend llava).",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Base model path when model-path is a LoRA adapter.",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="vicuna_v1",
        help="LLaVA conversation template name.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for LLaVA generation.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for resolving relative image paths (LLaVA backend).",
    )
    return parser


class _Stats:
    def __init__(self, backend_name: str) -> None:
        self.total_input = 0
        self.hallucinated = 0
        self.non_hallucinated_skipped = 0
        self.rewrites_emitted = 0
        self.backend = backend_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_rows": self.total_input,
            "hallucinated_rows": self.hallucinated,
            "non_hallucinated_skipped": self.non_hallucinated_skipped,
            "rewrites_emitted": self.rewrites_emitted,
            "backend": self.backend,
        }


def _run_pipeline(
    backend,
    rows: Iterable[dict[str, Any]],
    stats: _Stats,
    *,
    strict: bool,
    limit: int | None,
    total: int | None = None,
) -> Iterator[dict[str, Any]]:
    progress = maybe_tqdm(rows, desc="Stage 2", total=total)
    for i, row in enumerate(progress):
        if limit is not None and i >= limit:
            break
        stats.total_input += 1

        is_hallucinated = row.get("is_hallucinated", False)
        if not is_hallucinated:
            stats.non_hallucinated_skipped += 1
            continue

        stats.hallucinated += 1
        row_id = row.get("id")
        metadata: dict[str, Any] = {
            "source_stage": "stage2_rewrite",
            "backend": backend.name,
            "prompt_version": PROMPT_VERSION,
        }

        try:
            rewrite_text = backend.rewrite(row, strict=strict)
        except RewriteError as exc:
            if strict:
                raise
            metadata["rewrite_warning"] = str(exc)
            rewrite_text = ""

        if not rewrite_text or not rewrite_text.strip():
            if strict:
                raise RewriteError(
                    f"row id={row_id!r} backend returned empty rewrite"
                )
            metadata["rewrite_warning"] = metadata.get(
                "rewrite_warning", "backend returned empty rewrite"
            )
            rewrite_text = row.get("response_text", "") or ""

        stats.rewrites_emitted += 1
        record = Stage2Record(
            id=row_id,
            image=row.get("image"),
            question=row.get("question", ""),
            original_response=row.get("response_text", ""),
            rewrite_response=rewrite_text.strip(),
            critiques=list(row.get("critiques") or []),
            metadata=metadata,
        )
        yield record.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 2 input not found: {args.input}", file=sys.stderr)
        print(
            "Run Stage 1 first:  bash scripts/run_stage1_critiques.sh",
            file=sys.stderr,
        )
        return 2

    backend_kwargs: dict[str, Any] = {}
    if args.model_path:
        backend_kwargs["model_path"] = args.model_path
    if args.model_base:
        backend_kwargs["model_base"] = args.model_base
    if args.conv_mode:
        backend_kwargs["conv_mode"] = args.conv_mode
    if args.max_new_tokens:
        backend_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.image_root:
        backend_kwargs["image_root"] = args.image_root

    try:
        backend = get_backend(args.backend, **backend_kwargs)
    except ValueError as exc:
        print(f"Stage 2 backend error: {exc}", file=sys.stderr)
        return 2

    stats = _Stats(backend_name=backend.name)
    total_rows = count_jsonl_rows(args.input)
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    rows_iter = read_jsonl(args.input)
    ensure_parent_dir(args.output)

    try:
        write_jsonl(
            args.output,
            _run_pipeline(
                backend,
                rows_iter,
                stats,
                strict=args.strict,
                limit=args.limit,
                total=total_rows,
            ),
        )
    except RewriteError as exc:
        print(f"Stage 2 strict rewrite failed: {exc}", file=sys.stderr)
        return 3

    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Stage 2 wrote {stats.rewrites_emitted} rewrite(s) "
        f"({stats.non_hallucinated_skipped} non-hallucinated skipped) -> {args.output}"
    )
    print(f"Stage 2 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
