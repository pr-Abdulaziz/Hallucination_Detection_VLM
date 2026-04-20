from __future__ import annotations

from fg_pipeline.schemas import SentenceSignal


def _format_signal(signal: SentenceSignal) -> str:
    meta = signal.metadata or {}
    span = meta.get("span")
    parts = [
        f"sentence_index={signal.sentence_index}",
        f"type={signal.hallucination_type}",
        f"severity={signal.severity}",
        f"confidence={signal.confidence:.3f}",
    ]
    if span:
        parts.append(f'span="{span}"')
    if signal.rationale:
        parts.append(f"rationale={signal.rationale}")
    return "- " + ", ".join(parts)


def build_rewrite_prompt(
    prompt: str, response: str, signals: list[SentenceSignal]
) -> str:
    """Conservative rewrite prompt that preserves unflagged content.

    Includes span, rationale, and confidence for each reliable signal so the
    rewrite model can target the exact text spans flagged by Stage 3 while
    leaving correct portions of the response untouched.
    """

    signal_block = (
        "\n".join(_format_signal(s) for s in signals)
        if signals
        else "- no flagged hallucinations"
    )
    return (
        "You are correcting a vision-language model response.\n"
        "Rewrite the response so that every flagged span is corrected or "
        "removed. Keep all correct content unchanged. Do not add new "
        "details that are not supported by the image or the original "
        "response.\n\n"
        f"Original prompt:\n{prompt}\n\n"
        f"Original response:\n{response}\n\n"
        f"Flagged hallucination signals:\n{signal_block}\n\n"
        "Rewritten response:"
    )
