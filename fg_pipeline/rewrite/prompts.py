from __future__ import annotations

from fg_pipeline.schemas import SentenceSignal


def build_rewrite_prompt(prompt: str, response: str, signals: list[SentenceSignal]) -> str:
    """Create a conservative rewrite prompt that preserves unflagged content."""

    signal_lines = []
    for signal in signals:
        signal_lines.append(
            f"- sentence_index={signal.sentence_index}, "
            f"type={signal.hallucination_type}, "
            f"severity={signal.severity}, "
            f"confidence={signal.confidence:.3f}"
        )
    signal_block = "\n".join(signal_lines) if signal_lines else "- no flagged hallucinations"
    return (
        "Rewrite the response conservatively.\n"
        "Keep correct details unchanged and only fix flagged hallucinations.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Original response:\n{response}\n\n"
        f"Signals:\n{signal_block}\n"
    )
