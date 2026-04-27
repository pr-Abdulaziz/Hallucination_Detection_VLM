"""Paper Stage 2: build detector SFT data from D_faif."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION
from fg_pipeline.paper.prompts import DETECTOR_PROMPT_VERSION, build_detector_prompt
from fg_pipeline.paths import (
    DEFAULT_PAPER_STAGE1_OUTPUT,
    DEFAULT_PAPER_STAGE2_DETECTOR_DATA,
    DEFAULT_PAPER_STAGE2_STATS,
)


DETECTOR_DATASET_VERSION = "paper_detector_sft_v1"
DEFAULT_NON_HALLUCINATED_RATIO = 1.2
DEFAULT_HALLUCINATED_TARGET = 7643
DEFAULT_NON_HALLUCINATED_TARGET = 8500
NO_HALLUCINATION = "NO HALLUCINATION"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper Stage 2: build balanced LLaVA detector SFT data.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PAPER_STAGE1_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAPER_STAGE2_DETECTOR_DATA)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_PAPER_STAGE2_STATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hallucinated-target", type=int, default=DEFAULT_HALLUCINATED_TARGET)
    parser.add_argument("--non-hallucinated-target", type=int, default=DEFAULT_NON_HALLUCINATED_TARGET)
    parser.add_argument(
        "--non-hallucinated-ratio",
        type=float,
        default=None,
        help="Compatibility option; if set, overrides --non-hallucinated-target.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _numbered(text: str, index: int) -> str:
    text = text.strip()
    if not text:
        text = "Hallucination identified."
    if text[:1].isdigit():
        return text
    return f"{index} . {text}"


def normalized_detector_target(row: dict[str, Any]) -> str:
    if not row.get("is_hallucinated"):
        return NO_HALLUCINATION

    critiques = list(row.get("critiques") or [])
    if not critiques:
        metadata = row.get("metadata") or {}
        raw_annotation = str(metadata.get("raw_annotation") or metadata.get("raw_annotation_text") or "").strip()
        if "Tags:" in raw_annotation and "Scores:" in raw_annotation:
            return raw_annotation
        return "Tags:\nScores:"

    by_type: dict[str, list[dict[str, Any]]] = {}
    for critique in critiques:
        by_type.setdefault(str(critique.get("hallucination_type") or "unknown"), []).append(critique)

    lines: list[str] = ["Tags:"]
    for h_type in ("object", "attribute", "relationship", "unknown"):
        items = by_type.get(h_type) or []
        if not items:
            continue
        lines.append(f"<{h_type}>")
        for index, critique in enumerate(items, start=1):
            tag_text = str(critique.get("source_tag_text") or critique.get("rationale") or "")
            lines.append(_numbered(tag_text, index))

    lines.append("Scores:")
    for h_type in ("object", "attribute", "relationship", "unknown"):
        items = by_type.get(h_type) or []
        if not items:
            continue
        lines.append(f"<{h_type}>")
        for index, critique in enumerate(items, start=1):
            source_score = str(critique.get("source_score_text") or "").strip()
            if source_score:
                lines.append(_numbered(source_score, index))
                continue
            evidence = str(critique.get("evidence_text") or "Evidence span").strip()
            label = str(critique.get("severity_label") or "unknown").strip().title()
            score = critique.get("severity_score")
            if isinstance(score, int):
                points = "point" if score == 1 else "points"
                severity = f"{label} ({score} {points})"
            else:
                severity = label
            reason = str(critique.get("rationale") or "").strip() or "Severity assigned from the detected hallucination."
            lines.append(f"{index} . {evidence}: {severity}: {reason}")
    return "\n".join(lines)


def build_detector_example(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or row.get("response_text") or "").strip()
    response_text = str(row.get("response_text") or question).strip()
    prompt = build_detector_prompt(question=question, response_text=response_text)
    return {
        "id": row.get("id"),
        "image": row.get("image"),
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + prompt,
            },
            {
                "from": "gpt",
                "value": normalized_detector_target(row),
            },
        ],
        "metadata": {
            "source_stage": "paper_stage2_detector_dataset",
            "prompt_version": DETECTOR_PROMPT_VERSION,
            "is_hallucinated": bool(row.get("is_hallucinated")),
        },
    }


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    hallucinated_target: int = DEFAULT_HALLUCINATED_TARGET,
    non_hallucinated_target: int = DEFAULT_NON_HALLUCINATED_TARGET,
    non_hallucinated_ratio: float | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hallucinated = [row for row in rows if row.get("is_hallucinated")]
    non_hallucinated = [row for row in rows if not row.get("is_hallucinated")]

    if non_hallucinated_ratio is not None:
        non_hallucinated_target = int(round(min(len(hallucinated), hallucinated_target) * non_hallucinated_ratio))

    selected_hallucinated = hallucinated[: min(hallucinated_target, len(hallucinated))]
    selected_non = non_hallucinated[: min(non_hallucinated_target, len(non_hallucinated))]
    selected = selected_hallucinated + selected_non
    rng = random.Random(seed)
    rng.shuffle(selected)
    if limit is not None:
        selected = selected[:limit]

    selected_h = sum(1 for row in selected if row.get("is_hallucinated"))
    selected_n = len(selected) - selected_h
    stats = {
        "paper_pipeline_version": PAPER_PIPELINE_VERSION,
        "detector_dataset_version": DETECTOR_DATASET_VERSION,
        "prompt_version": DETECTOR_PROMPT_VERSION,
        "seed": seed,
        "source_total_rows": len(rows),
        "available_hallucinated_rows": len(hallucinated),
        "available_non_hallucinated_rows": len(non_hallucinated),
        "requested_hallucinated_rows": hallucinated_target,
        "requested_non_hallucinated_rows": non_hallucinated_target,
        "requested_non_hallucinated_ratio": non_hallucinated_ratio,
        "total_selected_rows": len(selected),
        "selected_hallucinated_rows": selected_h,
        "selected_non_hallucinated_rows": selected_n,
        "actual_hallucinated_to_non_hallucinated_ratio": (selected_h / selected_n) if selected_n else None,
        "actual_non_hallucinated_ratio": (selected_n / selected_h) if selected_h else None,
        "limit": limit,
    }
    return selected, stats


def _target_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        target = example["conversations"][1]["value"]
        if target == NO_HALLUCINATION:
            counts["no_hallucination"] += 1
        elif "Tags:" in target and "Scores:" in target:
            counts["tags_scores"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Paper Stage 2 input not found: {args.input}", file=sys.stderr)
        print("Run Paper Stage 1 first: bash scripts/run_paper_stage1_faif.sh", file=sys.stderr)
        return 2
    if args.hallucinated_target < 0 or args.non_hallucinated_target < 0:
        print("--hallucinated-target and --non-hallucinated-target must be >= 0", file=sys.stderr)
        return 2
    if args.non_hallucinated_ratio is not None and args.non_hallucinated_ratio < 0:
        print("--non-hallucinated-ratio must be >= 0", file=sys.stderr)
        return 2

    rows = list(read_jsonl(args.input))
    selected, stats = select_rows(
        rows,
        seed=args.seed,
        hallucinated_target=args.hallucinated_target,
        non_hallucinated_target=args.non_hallucinated_target,
        non_hallucinated_ratio=args.non_hallucinated_ratio,
        limit=args.limit,
    )
    examples = [build_detector_example(row) for row in selected]
    stats["target_format_counts"] = _target_counts(examples)
    stats["input_path"] = str(args.input)
    stats["output_path"] = str(args.output)

    ensure_parent_dir(args.output)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Paper Stage 2 wrote {stats['total_selected_rows']} detector example(s) "
        f"({stats['selected_hallucinated_rows']} hallucinated, "
        f"{stats['selected_non_hallucinated_rows']} non-hallucinated) -> {args.output}"
    )
    print(f"Paper Stage 2 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
