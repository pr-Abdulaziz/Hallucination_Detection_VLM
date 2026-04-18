from __future__ import annotations


def clip_weight(value: float, minimum: float = 0.1, maximum: float = 5.0) -> float:
    return max(minimum, min(maximum, value))


def adaptive_example_weight(pair_confidence: float, severity_weight: float) -> float:
    """Combine verification confidence with severity into a single example weight."""

    raw_weight = float(pair_confidence) * max(float(severity_weight), 0.0)
    return clip_weight(raw_weight if raw_weight > 0 else 0.1)
