"""Shared helpers for the paper-faithful pipeline path."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any


PAPER_PIPELINE_VERSION = "paper_faithful_v1"


def aggregate_severity(critiques: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        score = critique.get("severity_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return float(mean(scores)) if scores else 0.0


def normalize_space(text: str | None) -> str:
    return " ".join((text or "").split())


def resolve_existing_image(image: str | None, image_root: str | Path | None) -> Path | None:
    if not image:
        return None
    candidate = Path(image)
    if not candidate.is_absolute():
        candidate = Path(image_root or ".") / candidate
    return candidate if candidate.exists() else None
