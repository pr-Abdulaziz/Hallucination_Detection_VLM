"""Paper Stage 1 CLI: parse released FAIF annotations into D_FAIF."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION
from fg_pipeline.paper.prompts import (
    DDG_ANNOTATION_PROMPT_VERSION,
    SEVERITY_RUBRIC_VERSION,
    VCR_ANNOTATION_PROMPT_VERSION,
)
from fg_pipeline.paths import DEFAULT_STAGE1_INPUT
from fg_pipeline.stage1.backends import ReleasedAnnotationBackend
from fg_pipeline.stage1.parser import ParseError


DEFAULT_OUTPUT = Path("output/fghd/paper_stage1/d_faif.jsonl")
DEFAULT_STATS = Path("output/fghd/paper_stage1/stats.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper Stage 1: build D_FAIF from released annotations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_STAGE1_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def _paper_record(row: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    result = ReleasedAnnotationBackend().detect(row, strict=strict)
    record = result.record.to_dict()
    metadata = dict(record.get("metadata") or {})
    source = metadata.get("source") or "released_annotations"
    metadata["annotation_source"] = source
    if "raw_annotation" not in metadata:
        metadata["raw_annotation"] = metadata.get("raw_annotation_text", "")
    metadata["paper_stage"] = "paper_stage1_faif"
    metadata["paper_pipeline_version"] = PAPER_PIPELINE_VERSION
    metadata["ddg_annotation_prompt_version"] = DDG_ANNOTATION_PROMPT_VERSION
    metadata["vcr_annotation_prompt_version"] = VCR_ANNOTATION_PROMPT_VERSION
    metadata["severity_rubric_version"] = SEVERITY_RUBRIC_VERSION
    record["metadata"] = metadata
    return record


def _iter_records(rows: Iterable[dict[str, Any]], *, strict: bool, limit: int | None, total: int) -> Iterator[dict[str, Any]]:
    progress = maybe_tqdm(rows, desc="Paper Stage 1", total=total)
    for i, row in enumerate(progress):
        if limit is not None and i >= limit:
            break
        yield _paper_record(row, strict=strict)


def _compute_stats(path: Path, *, input_path: Path) -> dict[str, Any]:
    total = 0
    hallucinated = 0
    non_hallucinated = 0
    critique_count = 0
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    annotation_sources: Counter[str] = Counter()

    for row in read_jsonl(path):
        total += 1
        metadata = row.get("metadata") or {}
        annotation_sources[metadata.get("annotation_source", "unknown")] += 1
        if row.get("is_hallucinated"):
            hallucinated += 1
        else:
            non_hallucinated += 1
        for critique in row.get("critiques") or []:
            critique_count += 1
            by_type[critique.get("hallucination_type", "unknown")] += 1
            by_severity[critique.get("severity_label", "unknown")] += 1

    return {
        "stage": "paper_stage1_faif",
        "paper_pipeline_version": PAPER_PIPELINE_VERSION,
        "input_path": str(input_path),
        "total_rows": total,
        "hallucinated_rows": hallucinated,
        "non_hallucinated_rows": non_hallucinated,
        "critique_rows": critique_count,
        "critique_count_by_type": dict(by_type),
        "critique_count_by_severity": dict(by_severity),
        "annotation_sources": dict(annotation_sources),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Paper Stage 1 input not found: {args.input}", file=sys.stderr)
        return 2

    total = count_jsonl_rows(args.input)
    if args.limit is not None:
        total = min(total, args.limit)

    try:
        write_jsonl(args.output, _iter_records(read_jsonl(args.input), strict=args.strict, limit=args.limit, total=total))
    except ParseError as exc:
        print(f"Paper Stage 1 strict parse failed: {exc}", file=sys.stderr)
        return 3

    stats = _compute_stats(args.output, input_path=args.input)
    stats_path = ensure_parent_dir(args.stats_out)
    with stats_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Paper Stage 1 wrote {stats['total_rows']} D_FAIF record(s) "
        f"({stats['hallucinated_rows']} hallucinated, {stats['non_hallucinated_rows']} non-hallucinated) -> {args.output}"
    )
    print(f"Paper Stage 1 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
