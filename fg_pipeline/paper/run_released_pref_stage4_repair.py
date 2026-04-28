"""Repair API-rejected released preference pairs with LLaVA and merge outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.io_utils import ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION, normalize_space
from fg_pipeline.stage2.backends import LlavaRewriteBackend, TemplateRewriteBackend


ZERO_SHOT_STAGE3_DIR = Path("output/fghd/released_pref_stage3")
TWO_SHOT_STAGE3_DIR = Path("output/fghd/released_pref_stage3_2shot_experiment")
ZERO_SHOT_OUTPUT_DIR = Path("output/fghd/released_pref_stage4")
TWO_SHOT_OUTPUT_DIR = Path("output/fghd/released_pref_stage4_2shot_experiment")
DEFAULT_EXPERIMENT_MODE = "zero_shot"
DEFAULT_REJECTED = ZERO_SHOT_STAGE3_DIR / "rejected_for_repair.jsonl"
DEFAULT_ACCEPTED = ZERO_SHOT_STAGE3_DIR / "validated_preferences.jsonl"
DEFAULT_IMAGE_ROOT = Path("hsa_dpo/data/images")
DEFAULT_OUTPUT_DIR = ZERO_SHOT_OUTPUT_DIR
DEFAULT_REPAIRS = DEFAULT_OUTPUT_DIR / "repair_records.jsonl"
DEFAULT_REPAIRED_PREFS = DEFAULT_OUTPUT_DIR / "repaired_preferences.jsonl"
DEFAULT_FINAL_PREFS = DEFAULT_OUTPUT_DIR / "final_preference_pairs.jsonl"
DEFAULT_STATS = DEFAULT_OUTPUT_DIR / "stats.json"
STAGE3_DIRS = {
    "zero_shot": ZERO_SHOT_STAGE3_DIR,
    "two_shot": TWO_SHOT_STAGE3_DIR,
}
OUTPUT_DIRS = {
    "zero_shot": ZERO_SHOT_OUTPUT_DIR,
    "two_shot": TWO_SHOT_OUTPUT_DIR,
}

PROMPT_VERSION = "released_pref_repair_v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair API-rejected released preference pairs with LLaVA and merge with accepted pairs.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=("zero_shot", "two_shot"),
        default=DEFAULT_EXPERIMENT_MODE,
        help="Preference experiment path to consume. Select two_shot for the few-shot validation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base repair output directory. Defaults to an experiment-mode-specific folder.",
    )
    parser.add_argument("--rejected-input", type=Path, default=None)
    parser.add_argument("--accepted-input", type=Path, default=None)
    parser.add_argument("--repair-out", type=Path, default=None)
    parser.add_argument("--repaired-preferences-out", type=Path, default=None)
    parser.add_argument("--final-preferences-out", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--backend", choices=("template", "llava"), default="llava")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    stage3_dir = STAGE3_DIRS.get(args.experiment_mode, ZERO_SHOT_STAGE3_DIR)
    output_dir = args.output_dir or OUTPUT_DIRS.get(args.experiment_mode, ZERO_SHOT_OUTPUT_DIR)
    if args.rejected_input is None:
        args.rejected_input = stage3_dir / "rejected_for_repair.jsonl"
    if args.accepted_input is None:
        args.accepted_input = stage3_dir / "validated_preferences.jsonl"
    if args.repair_out is None:
        args.repair_out = output_dir / "repair_records.jsonl"
    if args.repaired_preferences_out is None:
        args.repaired_preferences_out = output_dir / "repaired_preferences.jsonl"
    if args.final_preferences_out is None:
        args.final_preferences_out = output_dir / "final_preference_pairs.jsonl"
    if args.stats_out is None:
        args.stats_out = output_dir / "stats.json"


def _validation_feedback(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    votes = metadata.get("api_votes") or []
    lines: list[str] = []
    for vote in votes:
        lines.append(
            f"- {vote.get('family', 'judge')}: approved={bool(vote.get('approved'))}; "
            f"reason={vote.get('reason', '')}"
        )
    return "\n".join(lines) if lines else "No API feedback was recorded."


def _repair_prompt(row: dict[str, Any]) -> str:
    return (
        "You are repairing a hallucination-mitigation rewrite for a vision-language model.\n"
        "Use the image as the source of truth.\n"
        "Produce only the final corrected answer, with no markdown and no explanation.\n\n"
        "The current chosen rewrite was rejected by the API judges. Revise it so it is better grounded in the image, "
        "preserves correct useful details, removes unsupported details, and avoids adding new unsupported claims.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Original rejected response:\n{row.get('rejected', '')}\n\n"
        f"Current chosen rewrite:\n{row.get('chosen', '')}\n\n"
        f"Original hallucination tags and severity:\n{row.get('rejected_tag_text', '')}\n\n"
        f"API judge feedback:\n{_validation_feedback(row)}\n\n"
        "Final corrected answer:"
    )


class _TemplateRepairBackend:
    name = "template"

    def __init__(self) -> None:
        self._backend = TemplateRewriteBackend()

    def rewrite(self, row: dict[str, Any], *, strict: bool = False) -> str:
        adapted = {
            "response_text": row.get("rejected", ""),
            "critiques": [
                {
                    "rationale": row.get("rejected_tag_text", ""),
                    "hallucination_type": "object",
                    "severity_score": row.get("rejected_score", 1.0),
                }
            ],
        }
        return self._backend.rewrite(adapted, strict=strict)


class _LlavaRepairBackend:
    name = "llava"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None,
        conv_mode: str,
        max_new_tokens: int | None,
        temperature: float,
        image_root: str,
    ) -> None:
        self._backend = LlavaRewriteBackend(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            image_root=image_root,
            prompt_builder=_repair_prompt,
        )

    def rewrite(self, row: dict[str, Any], *, strict: bool = False) -> str:
        return self._backend.rewrite(row, strict=strict)


def _image_for_row(row: dict[str, Any], image_root: Path) -> str | None:
    if row.get("image"):
        return str(row.get("image"))
    if row.get("id") in (None, ""):
        return None
    candidate = image_root / f"{row.get('id')}.jpg"
    return str(candidate) if candidate.exists() else f"{row.get('id')}.jpg"


def _get_backend(args: argparse.Namespace) -> Any:
    if args.backend == "template":
        return _TemplateRepairBackend()
    if not args.model_path:
        raise ValueError("--backend llava requires --model-path")
    return _LlavaRepairBackend(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        image_root=str(args.image_root),
    )


def _repair_preference(row: dict[str, Any], repair_text: str, *, image_root: Path) -> dict[str, Any]:
    repaired = dict(row)
    repaired["chosen"] = repair_text
    repaired["chosen_score"] = 1.0
    repaired["rejected_score"] = float(row.get("rejected_score", 1.0))
    image_value = _image_for_row(row, image_root)
    if image_value is not None:
        repaired["image"] = image_value
    metadata = dict(repaired.get("metadata") or {})
    metadata.update(
        {
            "source_stage": "released_pref_stage4_repair",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "previous_chosen": row.get("chosen", ""),
        }
    )
    repaired["metadata"] = metadata
    return repaired


def _iter_repairs(
    rows: Iterable[dict[str, Any]],
    *,
    backend: Any,
    image_root: Path,
    limit: int | None,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    repair_records: list[dict[str, Any]] = []
    repaired_preferences: list[dict[str, Any]] = []
    stats = {
        "input_rows": 0,
        "repair_rows_processed": 0,
        "repair_records_emitted": 0,
        "empty_rewrite": 0,
        "identical_to_rejected": 0,
    }
    for index, row in enumerate(maybe_tqdm(rows, desc="Released pref Stage 4 repair", total=None)):
        if limit is not None and index >= limit:
            break
        stats["input_rows"] += 1
        stats["repair_rows_processed"] += 1
        text = backend.rewrite(row, strict=strict).strip()
        if not text:
            stats["empty_rewrite"] += 1
            if strict:
                raise ValueError(f"row id={row.get('id')!r} produced empty repair")
            continue
        if normalize_space(text).lower() == normalize_space(row.get("rejected", "")).lower():
            stats["identical_to_rejected"] += 1
            if strict:
                raise ValueError(f"row id={row.get('id')!r} repair equals rejected")
            continue
        repaired = _repair_preference(row, text, image_root=image_root)
        repair_records.append(
            {
                "id": row.get("id"),
                "question": row.get("question", ""),
                "previous_chosen": row.get("chosen", ""),
                "repaired_chosen": text,
                "rejected": row.get("rejected", ""),
                "rejected_score": row.get("rejected_score", 1.0),
                "metadata": repaired.get("metadata") or {},
            }
        )
        repaired_preferences.append(repaired)
        stats["repair_records_emitted"] += 1
    return repair_records, repaired_preferences, stats


def _read_all(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _resolve_paths(args)
    if not args.rejected_input.exists():
        print(f"Rejected preference input not found: {args.rejected_input}", file=sys.stderr)
        return 2
    try:
        backend = _get_backend(args)
    except ValueError as exc:
        print(f"Released pref Stage 4 backend error: {exc}", file=sys.stderr)
        return 2

    try:
        repair_records, repaired_preferences, repair_stats = _iter_repairs(
            read_jsonl(args.rejected_input),
            backend=backend,
            image_root=args.image_root,
            limit=args.limit,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"Released pref Stage 4 repair failed: {exc}", file=sys.stderr)
        return 3

    accepted = _read_all(args.accepted_input)
    final_preferences = accepted + repaired_preferences
    write_jsonl(ensure_parent_dir(args.repair_out), repair_records)
    write_jsonl(ensure_parent_dir(args.repaired_preferences_out), repaired_preferences)
    write_jsonl(ensure_parent_dir(args.final_preferences_out), final_preferences)
    payload = {
        "paper_pipeline_version": PAPER_PIPELINE_VERSION,
        "stage": "released_pref_stage4_repair",
        "prompt_version": PROMPT_VERSION,
        "experiment_mode": args.experiment_mode,
        "backend": backend.name,
        "accepted_input": str(args.accepted_input),
        "rejected_input": str(args.rejected_input),
        "accepted_rows": len(accepted),
        "repaired_rows": len(repaired_preferences),
        "final_preference_rows": len(final_preferences),
        **repair_stats,
        "repair_out": str(args.repair_out),
        "repaired_preferences_out": str(args.repaired_preferences_out),
        "final_preferences_out": str(args.final_preferences_out),
    }
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Released pref Stage 4 repaired {payload['repaired_rows']} row(s); "
        f"final preferences: {payload['final_preference_rows']}"
    )
    print(f"Final preferences -> {args.final_preferences_out}")
    print(f"Repair stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
