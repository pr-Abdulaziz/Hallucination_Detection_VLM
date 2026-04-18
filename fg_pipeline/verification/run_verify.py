from __future__ import annotations

import argparse

from fg_pipeline.adaptive_dpo.adaptive_loss import adaptive_example_weight
from fg_pipeline.confidence.scoring import adaptive_severity, average_confidence
from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.schemas import PreferenceCleanRecord, SentenceSignal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter D_rewrite.jsonl into D_pref_clean.jsonl.")
    parser.add_argument("--input", required=True, help="Path to D_rewrite.jsonl")
    parser.add_argument("--output", required=True, help="Path to write D_pref_clean.jsonl")
    parser.add_argument(
        "--min-pair-confidence",
        type=float,
        default=0.0,
        help="Minimum average filtered-signal confidence required to retain a pair.",
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


def maybe_keep_pair(row: dict, min_pair_confidence: float) -> PreferenceCleanRecord | None:
    signals = _coerce_signals(row.get("filtered_signals", []))
    pair_confidence = average_confidence(signals)
    if pair_confidence < min_pair_confidence:
        return None

    chosen = row.get("rewritten_response", "")
    rejected = row.get("source_response", "")
    if not chosen or chosen == rejected:
        return None

    return PreferenceCleanRecord(
        id=row.get("sample_id", ""),
        image=row.get("image"),
        question=row.get("prompt", ""),
        chosen=chosen,
        rejected=rejected,
        chosen_score=1.0,
        rejected_score=adaptive_severity(signals),
        pair_confidence=pair_confidence,
        severity_weight=adaptive_severity(signals),
        adaptive_weight=adaptive_example_weight(pair_confidence, adaptive_severity(signals)),
        metadata={"verification_mode": "heuristic_filter"},
    )


def main() -> int:
    args = parse_args()
    kept = []
    for row in read_jsonl(args.input):
        item = maybe_keep_pair(row, args.min_pair_confidence)
        if item is not None:
            kept.append(item.to_dict())
    write_jsonl(args.output, kept)
    print(f"Wrote {len(kept)} clean preference pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
