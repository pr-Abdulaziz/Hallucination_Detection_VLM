from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fg_pipeline.adaptive_dpo.adaptive_loss import adaptive_example_weight


def normalize_preference_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one preference row for Stage 6 training.

    Stage 6 prefers the explicit Stage 5 ``image`` field and adaptive-weight
    metadata when present, while remaining backward compatible with the
    original HSA-DPO JSONL schema.
    """

    sample = {
        "id": item.get("id", ""),
        "image": item.get("image"),
        "question": item.get("question", ""),
        "chosen": item.get("chosen", ""),
        "rejected": item.get("rejected", ""),
        "chosen_score": float(item.get("chosen_score", 1.0)),
        "rejected_score": float(item.get("rejected_score", 1.0)),
    }

    has_pair_confidence = item.get("pair_confidence") is not None
    has_severity_weight = item.get("severity_weight") is not None
    has_adaptive_weight = item.get("adaptive_weight") is not None
    if not (has_pair_confidence or has_severity_weight or has_adaptive_weight):
        return sample

    pair_confidence = float(item.get("pair_confidence", 0.0))
    severity_weight = float(
        item.get("severity_weight", item.get("rejected_score", 1.0))
    )
    adaptive_weight = item.get("adaptive_weight")
    if adaptive_weight is None:
        adaptive_weight = adaptive_example_weight(pair_confidence, severity_weight)

    sample.update(
        {
            "pair_confidence": pair_confidence,
            "severity_weight": severity_weight,
            "adaptive_weight": float(adaptive_weight),
        }
    )
    return sample


def resolve_image_path(
    image_value: str | None,
    image_root: str | Path,
    fallback_id: int | str | None = None,
) -> Path:
    """Resolve the image path for Stage 6 training.

    Resolution order:
    1. explicit Stage 5 ``image`` field
    2. legacy ``id -> <image_root>/<id>.jpg`` fallback
    """

    root = Path(image_root)
    candidates: list[Path] = []

    if image_value:
        image_path = Path(image_value)
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append(root / image_path)
            candidates.append(image_path)

    if fallback_id not in (None, ""):
        candidates.append(root / f"{fallback_id}.jpg")

    if not candidates:
        return root / "__missing_image__.jpg"

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
