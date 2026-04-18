from __future__ import annotations

import argparse

from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.schemas import RewriteRecord, SentenceSignal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D_rewrite.jsonl from D_det.jsonl.")
    parser.add_argument("--input", required=True, help="Path to D_det.jsonl")
    parser.add_argument("--output", required=True, help="Path to write D_rewrite.jsonl")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Keep signals whose confidence is strictly above this threshold.",
    )
    return parser.parse_args()


def _coerce_signals(items: list[dict]) -> list[SentenceSignal]:
    signals = []
    for item in items:
        signals.append(
            SentenceSignal(
                sentence_index=item.get("sentence_index", 0),
                hallucination_type=item.get("hallucination_type"),
                severity=item.get("severity"),
                confidence=float(item.get("confidence", 0.0)),
                rationale=item.get("rationale"),
                raw_label=item.get("raw_label"),
                metadata=item.get("metadata", {}),
            )
        )
    return signals


def build_rewrite_record(row: dict, confidence_threshold: float) -> RewriteRecord:
    signals = _coerce_signals(row.get("signals", []))
    filtered = [signal for signal in signals if signal.confidence > confidence_threshold]

    # Placeholder rewrite behavior:
    # keep the original candidate response until a dedicated rewrite model is added.
    source_response = row.get("candidate_response", "")
    rewritten_response = source_response
    return RewriteRecord(
        sample_id=row.get("sample_id", ""),
        image=row.get("image"),
        prompt=row.get("prompt", ""),
        source_response=source_response,
        rewritten_response=rewritten_response,
        filtered_signals=filtered,
        metadata={"rewrite_mode": "placeholder_passthrough"},
    )


def main() -> int:
    args = parse_args()
    rows = [build_rewrite_record(row, args.confidence_threshold).to_dict() for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rewrite records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
