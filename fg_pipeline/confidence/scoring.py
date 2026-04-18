from __future__ import annotations

from typing import Iterable

from fg_pipeline.schemas import SentenceSignal


def average_confidence(signals: Iterable[SentenceSignal]) -> float:
    signals = list(signals)
    if not signals:
        return 0.0
    return sum(signal.confidence for signal in signals) / len(signals)


def max_severity_weighted_confidence(signals: Iterable[SentenceSignal]) -> float:
    weighted = [
        signal.confidence * float(signal.severity or 0)
        for signal in signals
    ]
    return max(weighted, default=0.0)


def adaptive_severity(signals: Iterable[SentenceSignal], alpha: float = 0.5) -> float:
    """Stable aggregation of confidence and severity for later DPO weighting."""

    signals = list(signals)
    if not signals:
        return 0.0
    mean_term = sum(signal.confidence * float(signal.severity or 0) for signal in signals) / len(signals)
    max_term = max_severity_weighted_confidence(signals)
    return alpha * mean_term + (1.0 - alpha) * max_term
