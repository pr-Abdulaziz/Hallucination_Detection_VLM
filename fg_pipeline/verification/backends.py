"""Verification backends for Stage 5.

A backend decides whether a Stage 4 rewrite is actually better than the
Stage 1 source response, given the Stage 3 signals that were carried
through Stage 4. The backend does **not** recompute confidence — Stage 3
``c^j`` values are locked and pass through unchanged.

Only one backend ships in Batch 1:

- ``heuristic`` — strict deterministic rules. Operational but not the
  final research verifier; an LLM/VLM-backed verifier can be added later
  by registering a new class in ``_BACKEND_REGISTRY``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fg_pipeline.schemas import SentenceSignal


_MIN_REWRITE_CHARS = 8


@dataclass
class VerificationResult:
    """Outcome of one verification call."""

    passed: bool
    reason: str
    num_verified_signals: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VerificationBackend(Protocol):
    """Verify a single (source, rewritten, filtered_signals) triple."""

    name: str

    def verify(
        self,
        source: str,
        rewritten: str,
        filtered_signals: list[SentenceSignal],
        context: dict[str, Any],
    ) -> VerificationResult: ...


def _spans(signals: list[SentenceSignal]) -> list[str]:
    out: list[str] = []
    for s in signals:
        meta = s.metadata or {}
        span = meta.get("span")
        if isinstance(span, str) and span.strip():
            out.append(span)
    return out


def _looks_degenerate(text: str) -> bool:
    """Catch trivial collapse patterns: empty, whitespace, single repeated token."""

    stripped = text.strip()
    if not stripped:
        return True
    tokens = stripped.split()
    if len(tokens) >= 3 and len(set(tokens)) == 1:
        return True
    return False


class HeuristicVerificationBackend:
    """Strict deterministic verifier.

    Keeps a pair only if every rule below holds:

    1. ``rewritten`` is non-empty and above the minimum character floor.
    2. ``rewritten != source``.
    3. ``rewritten`` is not a degenerate repeated-token collapse.
    4. When at least one filtered signal carries a text ``span``, at least
       one of those spans must no longer appear in ``rewritten``. This is
       the per-signal "fixed at least some flagged hallucination" check.

    Rule 4 is skipped when no signal carries a span (parser may have failed
    to extract one). In that case the verifier trusts Stage 3's labelling
    and Stage 4's rewrite choice, and reports ``num_verified_signals = 0``.
    """

    name = "heuristic"

    def __init__(self, min_rewrite_chars: int = _MIN_REWRITE_CHARS) -> None:
        self.min_rewrite_chars = int(min_rewrite_chars)

    def verify(
        self,
        source: str,
        rewritten: str,
        filtered_signals: list[SentenceSignal],
        context: dict[str, Any],
    ) -> VerificationResult:
        rewritten = rewritten or ""
        source = source or ""

        if not rewritten.strip():
            return VerificationResult(False, "empty_rewrite")
        if len(rewritten.strip()) < self.min_rewrite_chars:
            return VerificationResult(False, "rewrite_too_short")
        if rewritten == source:
            return VerificationResult(False, "rewrite_equals_source")
        if _looks_degenerate(rewritten):
            return VerificationResult(False, "rewrite_degenerate")

        spans = _spans(filtered_signals)
        if spans:
            removed = [s for s in spans if s not in rewritten]
            if not removed:
                return VerificationResult(
                    False,
                    "no_flagged_span_removed",
                    metadata={"spans_checked": spans},
                )
            return VerificationResult(
                True,
                "passed_span_check",
                num_verified_signals=len(removed),
                metadata={"removed_spans": removed, "spans_checked": spans},
            )

        return VerificationResult(
            True,
            "passed_without_spans",
            num_verified_signals=0,
            metadata={"spans_checked": []},
        )


_BACKEND_REGISTRY: dict[str, str] = {
    "heuristic": "fg_pipeline.verification.backends:HeuristicVerificationBackend",
}

_BACKEND_KWARG_WHITELIST: dict[str, set[str]] = {
    "heuristic": {"min_rewrite_chars"},
}


def _resolve_backend_class(name: str) -> type:
    try:
        target = _BACKEND_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"unknown verification backend {name!r}; available: {available}"
        ) from exc
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def get_backend(name: str, **kwargs: Any) -> VerificationBackend:
    cls = _resolve_backend_class(name)
    allowed = _BACKEND_KWARG_WHITELIST.get(name, set())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return cls(**filtered)
