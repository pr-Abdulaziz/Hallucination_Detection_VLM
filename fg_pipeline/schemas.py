from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SentenceSignal:
    """Sentence-level hallucination signal used across the new pipeline stages."""

    sentence_index: int
    hallucination_type: Optional[str] = None
    severity: Optional[int] = None
    confidence: float = 0.0
    rationale: Optional[str] = None
    raw_label: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionRecord:
    """Output record emitted by the confidence-aware detection stage."""

    sample_id: int | str
    image: Optional[str]
    prompt: str
    candidate_response: str
    signals: list[SentenceSignal] = field(default_factory=list)
    raw_detection: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signals"] = [signal.to_dict() for signal in self.signals]
        return payload


@dataclass
class RewriteRecord:
    """Intermediate record emitted by the rewrite stage."""

    sample_id: int | str
    image: Optional[str]
    prompt: str
    source_response: str
    rewritten_response: str
    filtered_signals: list[SentenceSignal] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filtered_signals"] = [signal.to_dict() for signal in self.filtered_signals]
        return payload


@dataclass
class PreferenceCleanRecord:
    """Cleaned preference pair compatible with the original HSA-DPO trainer."""

    id: int | str
    question: str
    chosen: str
    rejected: str
    chosen_score: float = 1.0
    rejected_score: float = 1.0
    image: Optional[str] = None
    pair_confidence: float = 0.0
    severity_weight: float = 0.0
    adaptive_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
