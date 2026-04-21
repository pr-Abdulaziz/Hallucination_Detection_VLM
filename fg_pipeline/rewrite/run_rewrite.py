from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from fg_pipeline.confidence.calibration import lookup_group_threshold
from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.rewrite.backends import RewriteBackend, get_backend
from fg_pipeline.schemas import RewriteRecord, SentenceSignal
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build D_rewrite.jsonl from D_det.jsonl."
    )
    parser.add_argument("--input", required=True, help="Path to D_det.jsonl")
    parser.add_argument(
        "--output", required=True, help="Path to write D_rewrite.jsonl"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Keep signals whose confidence is strictly above this threshold.",
    )
    parser.add_argument(
        "--threshold-report",
        default=None,
        help=(
            "Optional Stage 3 calibration report JSON containing "
            "group-conditional tau_{type,severity}."
        ),
    )
    parser.add_argument(
        "--backend",
        default="template",
        help="Rewrite backend name (template | llava). Default: template.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model checkpoint path for model-backed backends.",
    )
    parser.add_argument(
        "--device", default="auto", help="Device spec for model-backed backends."
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory for resolving record image paths.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for the rewrite model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the rewrite model (0 = greedy).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser.parse_args()


def _coerce_signals(items: list[dict]) -> list[SentenceSignal]:
    return [
        SentenceSignal(
            sentence_index=item.get("sentence_index", 0),
            hallucination_type=item.get("hallucination_type"),
            severity=item.get("severity"),
            confidence=float(item.get("confidence", 0.0)),
            rationale=item.get("rationale"),
            raw_label=item.get("raw_label"),
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]


def filter_signals(
    signals: list[SentenceSignal],
    confidence_threshold: float,
    threshold_policy: dict | None = None,
) -> list[SentenceSignal]:
    """Keep only signals with c^j strictly greater than the threshold."""

    kept: list[SentenceSignal] = []
    for signal in signals:
        threshold = confidence_threshold
        if threshold_policy:
            threshold = lookup_group_threshold(
                threshold_policy,
                signal.hallucination_type,
                signal.severity,
                default=confidence_threshold,
            )
        if signal.confidence > threshold:
            kept.append(signal)
    return kept


def _load_threshold_policy(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("group_threshold_policy")


def build_rewrite_record(
    row: dict,
    backend: RewriteBackend,
    confidence_threshold: float,
    threshold_policy: dict | None = None,
) -> RewriteRecord:
    all_signals = _coerce_signals(row.get("signals", []))
    filtered = filter_signals(all_signals, confidence_threshold, threshold_policy)

    source_response = row.get("candidate_response", "")
    context = {
        "sample_id": row.get("sample_id", ""),
        "image": row.get("image"),
        "prompt": row.get("prompt", ""),
        "candidate_response": source_response,
    }

    base_meta = {
        "rewrite_backend": backend.name,
        "confidence_threshold": confidence_threshold,
        "num_input_signals": len(all_signals),
        "num_filtered_signals": len(filtered),
    }
    if threshold_policy:
        base_meta["threshold_policy"] = "group_conditional"
        base_meta["threshold_policy_global_fallback"] = threshold_policy.get(
            "global_threshold", confidence_threshold
        )

    if not filtered:
        metadata = {
            **base_meta,
            "rewrite_status": "skipped_no_reliable_signals",
        }
        rewritten_response = source_response
    else:
        rewritten_response, backend_meta = backend.rewrite(
            prompt=context["prompt"],
            response=source_response,
            signals=filtered,
            context=context,
        )
        metadata = {**base_meta, **backend_meta}
        metadata.setdefault("rewrite_status", "generated")

    return RewriteRecord(
        sample_id=row.get("sample_id", ""),
        image=row.get("image"),
        prompt=row.get("prompt", ""),
        source_response=source_response,
        rewritten_response=rewritten_response,
        filtered_signals=filtered,
        metadata=metadata,
    )


def generate_records(
    rows: Iterable[dict],
    backend: RewriteBackend,
    confidence_threshold: float,
    threshold_policy: dict | None = None,
    limit: int | None = None,
) -> list[dict]:
    rows_list = list(rows)
    if limit is not None:
        rows_list = rows_list[:limit]

    output: list[dict] = []
    for row in tqdm(
        rows_list,
        desc="Stage 4 rewrite",
        unit="row",
        disable=not sys.stderr.isatty(),
    ):
        output.append(
            build_rewrite_record(
                row,
                backend,
                confidence_threshold,
                threshold_policy=threshold_policy,
            ).to_dict()
        )
    return output


def _summarize(records: list[dict]) -> dict:
    total = len(records)
    generated = 0
    skipped = 0
    smoke_only = 0
    for r in records:
        status = (r.get("metadata") or {}).get("rewrite_status")
        if status == "skipped_no_reliable_signals":
            skipped += 1
        else:
            generated += 1
        if status == "generated_smoke_only":
            smoke_only += 1
    return {
        "total_rows": total,
        "generated": generated,
        "skipped": skipped,
        "smoke_only": smoke_only,
    }


def _print_summary(summary: dict, output_path: str) -> None:
    print(f"Wrote {summary['total_rows']} rewrite records to {output_path}")
    print(
        "  rows: "
        f"generated={summary['generated']} "
        f"skipped={summary['skipped']} "
        f"smoke_only={summary['smoke_only']}"
    )


def _backend_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_path": args.model_path,
        "device": args.device,
        "image_root": args.image_root,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }


def main() -> int:
    args = parse_args()
    kwargs = _backend_kwargs(args)
    if args.backend == "llava" and not kwargs.get("model_path"):
        raise SystemExit("--model-path is required when --backend llava is used")
    backend = get_backend(args.backend, **kwargs)
    threshold_policy = _load_threshold_policy(args.threshold_report)
    rows = generate_records(
        read_jsonl(args.input),
        backend,
        confidence_threshold=args.confidence_threshold,
        threshold_policy=threshold_policy,
        limit=args.limit,
    )
    write_jsonl(args.output, rows)
    _print_summary(_summarize(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
