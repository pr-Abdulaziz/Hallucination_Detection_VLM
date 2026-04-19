from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable

from fg_pipeline.confidence.parser import parse_detection_response
from fg_pipeline.confidence.scoring import ConfidenceScorer, get_scorer
from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.paths import DEFAULT_DETECTION_INPUT
from fg_pipeline.schemas import DetectionRecord, SentenceSignal

_PROMPT_PREFIX = "<image>\nDescription to Assess:\n"
# D_faif does not preserve the Stage-1 instruction that produced yhat.
# LLaVA-v1.5's standard VG-captioning instruction is used as a stable default.
_CANONICAL_STAGE1_INSTRUCTION = "Describe this image in detail."


def _extract_candidate_response(conversations: list[dict]) -> str:
    """yhat — the Stage-1 LVLM description being assessed."""

    if not conversations:
        return ""
    value = conversations[0].get("value", "")
    return value.replace(_PROMPT_PREFIX, "", 1)


def _extract_teacher_annotation(conversations: list[dict]) -> str:
    """Raw GPT-4/GPT-4V structured annotation (Tags/Scores or ``NO HALLUCINATION``)."""

    if len(conversations) < 2:
        return ""
    return conversations[1].get("value", "")


def _build_signals(
    raw_response: str, scorer: ConfidenceScorer, record_context: dict
) -> list[SentenceSignal]:
    parsed = parse_detection_response(raw_response)
    signals: list[SentenceSignal] = []
    for item in parsed:
        confidence, scorer_meta = scorer.score(item, record_context)
        signals.append(
            SentenceSignal(
                sentence_index=item["sentence_index"],
                hallucination_type=item["hallucination_type"],
                severity=item["severity"],
                confidence=float(confidence),
                rationale=item.get("rationale"),
                raw_label=item.get("raw_label"),
                metadata={
                    "source": "teacher_label",
                    "severity_label": item.get("severity_label"),
                    "span": item.get("span"),
                    "tag_text": item.get("tag_text"),
                    **scorer_meta,
                },
            )
        )
    return signals


def build_detection_record(row: dict, scorer: ConfidenceScorer) -> DetectionRecord:
    conversations = row.get("conversations", [])
    candidate_response = _extract_candidate_response(conversations)
    raw_detection = _extract_teacher_annotation(conversations)
    context = {
        "sample_id": row.get("id", ""),
        "image": row.get("image"),
        "prompt": _CANONICAL_STAGE1_INSTRUCTION,
        "candidate_response": candidate_response,
    }
    return DetectionRecord(
        sample_id=row.get("id", ""),
        image=row.get("image"),
        prompt=_CANONICAL_STAGE1_INSTRUCTION,
        candidate_response=candidate_response,
        signals=_build_signals(raw_detection, scorer, context),
        raw_detection=raw_detection,
        metadata={
            "source_dataset": "hsa_dpo_detection",
            "source_input_role": "fg_pipeline_detection_mirror",
            "scorer": scorer.name,
            "stage1_instruction_source": "canonical_describe_prompt",
        },
    )


def generate_records(
    rows: Iterable[dict], scorer: ConfidenceScorer, limit: int | None = None
) -> list[dict]:
    output: list[dict] = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        output.append(build_detection_record(row, scorer).to_dict())
    return output


def _is_no_hallucination(raw: str | None) -> bool:
    return not raw or raw.strip().upper() == "NO HALLUCINATION"


def _summarize(records: list[dict]) -> dict:
    total = len(records)
    no_hall_source = 0
    parsed_rows = 0
    unparseable_rows = 0
    type_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    confidences: list[float] = []
    total_signals = 0
    for record in records:
        is_no_hall = _is_no_hallucination(record.get("raw_detection"))
        if is_no_hall:
            no_hall_source += 1
        elif record["signals"]:
            parsed_rows += 1
        else:
            unparseable_rows += 1
        for signal in record["signals"]:
            total_signals += 1
            type_counts[signal.get("hallucination_type") or "unknown"] += 1
            severity_counts[signal.get("severity")] += 1
            confidences.append(float(signal.get("confidence", 0.0)))
    if confidences:
        conf_min = min(confidences)
        conf_mean = sum(confidences) / len(confidences)
        conf_max = max(confidences)
    else:
        conf_min = conf_mean = conf_max = 0.0
    return {
        "total_rows": total,
        "no_hallucination_rows": no_hall_source,
        "hallucinated_rows_parsed": parsed_rows,
        "hallucinated_rows_unparseable": unparseable_rows,
        "total_signals": total_signals,
        "type_counts": dict(type_counts),
        "severity_counts": {str(k): v for k, v in severity_counts.items()},
        "confidence": {"min": conf_min, "mean": conf_mean, "max": conf_max},
    }


def _print_summary(summary: dict, output_path: str) -> None:
    print(f"Wrote {summary['total_rows']} detection records to {output_path}")
    print(
        "  rows: "
        f"no_hall={summary['no_hallucination_rows']} "
        f"hall_parsed={summary['hallucinated_rows_parsed']} "
        f"hall_unparseable={summary['hallucinated_rows_unparseable']} "
        f"total_signals={summary['total_signals']}"
    )
    print(f"  by type:     {summary['type_counts']}")
    print(f"  by severity: {summary['severity_counts']}")
    c = summary["confidence"]
    print(f"  confidence:  min={c['min']:.4f} mean={c['mean']:.4f} max={c['max']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build D_det.jsonl from the Stage-3-owned detection data mirror."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_DETECTION_INPUT),
        help=(
            "Path to the detection JSONL. Defaults to "
            f"{DEFAULT_DETECTION_INPUT.as_posix()}."
        ),
    )
    parser.add_argument("--output", required=True, help="Path to write D_det.jsonl")
    parser.add_argument(
        "--scorer",
        default="bootstrap",
        help="Confidence scorer name (default: bootstrap).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke tests"
    )
    # Flags consumed by model-backed scorers (e.g. log_prob). Ignored by bootstrap.
    parser.add_argument("--model-path", default=None, help="Path to the VLM checkpoint.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device spec for the scorer (auto / cuda / cuda:0 / cpu).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature applied to logits before softmax.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory relative to which record image paths are resolved.",
    )
    return parser.parse_args()


_SCORER_KWARG_WHITELIST = {
    "bootstrap": set(),
    "log_prob": {"model_path", "device", "temperature", "image_root"},
}


def _scorer_kwargs(name: str, args: argparse.Namespace) -> dict:
    allowed = _SCORER_KWARG_WHITELIST.get(name, set())
    raw = {
        "model_path": args.model_path,
        "device": args.device,
        "temperature": args.temperature,
        "image_root": args.image_root,
    }
    return {k: v for k, v in raw.items() if k in allowed and v is not None}


def main() -> int:
    args = parse_args()
    kwargs = _scorer_kwargs(args.scorer, args)
    if args.scorer == "log_prob" and not kwargs.get("model_path"):
        raise SystemExit("--model-path is required when --scorer log_prob is used")
    scorer = get_scorer(args.scorer, **kwargs)
    rows = generate_records(read_jsonl(args.input), scorer, limit=args.limit)
    write_jsonl(args.output, rows)
    _print_summary(_summarize(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
