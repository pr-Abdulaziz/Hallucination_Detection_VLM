from __future__ import annotations

import argparse
import re
from typing import Iterable

from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.schemas import DetectionRecord, SentenceSignal

_TYPE_RE = re.compile(r"<(object|attribute|relationship)>", re.IGNORECASE)
_SEVERITY_RE = re.compile(r"(Minor|Moderate|Major)\s*\((\d+)\s*points?\)", re.IGNORECASE)
_PROMPT_PREFIX = "<image>\nDescription to Assess:\n"


def _extract_prompt(conversations: list[dict]) -> str:
    if not conversations:
        return ""
    prompt = conversations[0].get("value", "")
    return prompt.replace(_PROMPT_PREFIX, "", 1)


def _extract_response(conversations: list[dict]) -> str:
    if len(conversations) < 2:
        return ""
    return conversations[1].get("value", "")


def _bootstrap_signal(raw_response: str) -> list[SentenceSignal]:
    response = raw_response.strip()
    if not response or response == "NO HALLUCINATION":
        return []

    match_type = _TYPE_RE.search(response)
    match_severity = _SEVERITY_RE.search(response)
    signal = SentenceSignal(
        sentence_index=0,
        hallucination_type=(match_type.group(1).lower() if match_type else "unknown"),
        severity=(int(match_severity.group(2)) if match_severity else None),
        confidence=1.0,
        rationale=response,
        raw_label=response,
        metadata={"bootstrap_source": "teacher_label"},
    )
    return [signal]


def build_detection_record(row: dict) -> DetectionRecord:
    conversations = row.get("conversations", [])
    raw_detection = _extract_response(conversations)
    return DetectionRecord(
        sample_id=row.get("id", ""),
        image=row.get("image"),
        prompt=_extract_prompt(conversations),
        candidate_response=_extract_prompt(conversations),
        signals=_bootstrap_signal(raw_detection),
        raw_detection=raw_detection,
        metadata={"source_dataset": "hsa_dpo_detection"},
    )


def generate_records(rows: Iterable[dict], limit: int | None = None) -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        output.append(build_detection_record(row).to_dict())
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap D_det.jsonl from the released detection data.")
    parser.add_argument("--input", required=True, help="Path to hsa_dpo_detection.jsonl")
    parser.add_argument("--output", required=True, help="Path to write D_det.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = generate_records(read_jsonl(args.input), limit=args.limit)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} detection records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
