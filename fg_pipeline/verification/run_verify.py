"""Stage 5 — verify Stage 4 rewrites and emit a clean preference dataset.

Stage 5 is a **filter-and-validate** stage. It never recomputes ``c^j`` —
the Stage 3 confidences carried through Stage 4 pass through unchanged.
A pair is kept only when every rule holds:

- the rewrite backend judged the rewrite verified,
- ``rewritten_response != source_response``,
- at least one reliable filtered signal exists,
- ``pair_confidence > τ_c`` (``--min-pair-confidence``).

Everything else is dropped with a reason recorded in the run summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable

from fg_pipeline.adaptive_dpo.adaptive_loss import adaptive_example_weight
from fg_pipeline.confidence.scoring import adaptive_severity, average_confidence
from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.schemas import PreferenceCleanRecord, SentenceSignal
from fg_pipeline.verification.backends import (
    VerificationBackend,
    VerificationResult,
    get_backend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter D_rewrite.jsonl into D_pref_clean.jsonl."
    )
    parser.add_argument("--input", required=True, help="Path to D_rewrite.jsonl")
    parser.add_argument(
        "--output", required=True, help="Path to write D_pref_clean.jsonl"
    )
    parser.add_argument(
        "--min-pair-confidence",
        type=float,
        default=0.0,
        help="Keep pairs with pair_confidence strictly above this threshold (τ_c).",
    )
    parser.add_argument(
        "--backend",
        default="heuristic",
        help="Verification backend name (default: heuristic).",
    )
    parser.add_argument(
        "--min-rewrite-chars",
        type=int,
        default=8,
        help="Minimum character length for a rewrite to be considered non-trivial.",
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


def evaluate_pair(
    row: dict,
    backend: VerificationBackend,
    min_pair_confidence: float,
) -> tuple[PreferenceCleanRecord | None, str]:
    """Return (record_or_None, drop_reason).

    drop_reason is the empty string when the row is kept.
    """

    filtered = _coerce_signals(row.get("filtered_signals", []))
    source = row.get("source_response", "") or ""
    rewritten = row.get("rewritten_response", "") or ""

    if len(filtered) == 0:
        return None, "no_filtered_signals"

    pair_confidence = average_confidence(filtered)
    if pair_confidence <= min_pair_confidence:
        return None, "below_pair_confidence_threshold"

    if not rewritten.strip():
        return None, "empty_rewrite"
    if rewritten == source:
        return None, "rewrite_equals_source"

    context = {
        "sample_id": row.get("sample_id", ""),
        "image": row.get("image"),
        "prompt": row.get("prompt", ""),
        "rewrite_metadata": row.get("metadata") or {},
    }
    result: VerificationResult = backend.verify(source, rewritten, filtered, context)
    if not result.passed:
        return None, f"verifier:{result.reason}"

    severity_weight = adaptive_severity(filtered)
    record = PreferenceCleanRecord(
        id=row.get("sample_id", ""),
        image=row.get("image"),
        question=row.get("prompt", ""),
        chosen=rewritten,
        rejected=source,
        chosen_score=1.0,
        rejected_score=severity_weight,
        pair_confidence=pair_confidence,
        severity_weight=severity_weight,
        adaptive_weight=adaptive_example_weight(pair_confidence, severity_weight),
        metadata={
            "verification_backend": backend.name,
            "verification_status": "kept",
            "verification_reason": result.reason,
            "pair_confidence_threshold": min_pair_confidence,
            "num_filtered_signals": len(filtered),
            "num_verified_signals": result.num_verified_signals,
            "verifier_metadata": result.metadata,
            "source_rewrite_status": (row.get("metadata") or {}).get(
                "rewrite_status"
            ),
            "source_rewrite_backend": (row.get("metadata") or {}).get(
                "rewrite_backend"
            ),
        },
    )
    return record, ""


def generate_records(
    rows: Iterable[dict],
    backend: VerificationBackend,
    min_pair_confidence: float,
    limit: int | None = None,
) -> tuple[list[dict], Counter]:
    kept: list[dict] = []
    reasons: Counter = Counter()
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        record, reason = evaluate_pair(row, backend, min_pair_confidence)
        if record is None:
            reasons[reason] += 1
            continue
        kept.append(record.to_dict())
        reasons["kept"] += 1
    return kept, reasons


def _print_summary(output_path: str, kept: list[dict], reasons: Counter) -> None:
    total = sum(reasons.values())
    print(f"Wrote {len(kept)} clean preference pairs to {output_path}")
    print(f"  input rows processed: {total}")
    for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {reason}: {count}")


def main() -> int:
    args = parse_args()
    backend = get_backend(args.backend, min_rewrite_chars=args.min_rewrite_chars)
    kept, reasons = generate_records(
        read_jsonl(args.input),
        backend,
        min_pair_confidence=args.min_pair_confidence,
        limit=args.limit,
    )
    write_jsonl(args.output, kept)
    _print_summary(args.output, kept, reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
