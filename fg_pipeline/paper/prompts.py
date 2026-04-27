"""Appendix-aligned prompt and rubric templates for the paper path."""

from __future__ import annotations

from typing import Any


DETECTOR_PROMPT_VERSION = "paper_detector_v1"
DDG_PROMPT_VERSION = "paper_ddg_v1"
VCR_PROMPT_VERSION = "paper_vcr_v1"
REWRITE_PROMPT_VERSION = "paper_rewrite_v1"
API_CRITIC_PROMPT_VERSION = "api_critic_feedback_v1"
REVISION_PROMPT_VERSION = "api_critic_revision_v1"
SEVERITY_RUBRIC_VERSION = "paper_severity_rubric_v1"

# Compatibility names used by the paper-path stage modules.
DDG_ANNOTATION_PROMPT_VERSION = DDG_PROMPT_VERSION
VCR_ANNOTATION_PROMPT_VERSION = VCR_PROMPT_VERSION


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value or {})


def render_severity_rubric() -> str:
    return (
        "Severity rubric:\n"
        "- Minor (1 point): localized visual error that does not alter the main scene understanding.\n"
        "- Moderate (2 points): noticeable object, attribute, or relation error that changes part of the scene.\n"
        "- Major (3 points): central object, count, action, or relation error that substantially changes the answer.\n"
        "Use only Minor, Moderate, or Major."
    )


def severity_rubric_text() -> str:
    return render_severity_rubric()


def build_ddg_annotation_prompt(*, regions: str, description: str) -> str:
    return (
        "Task instructions:\n"
        "You are tasked with evaluating the accuracy of a detailed description of an image "
        "against provided Visual Genome regions. Identify hallucinated object, attribute, "
        "and relationship claims, explain why they are hallucinations, assign severity scores, "
        "and provide a corrected rewritten description.\n\n"
        f"{render_severity_rubric()}\n\n"
        f"Image Regions Information:\n{regions}\n\n"
        f"Description:\n{description}\n\n"
        "Output Format:\nTagged Description:\nReasons:\nScores:\nRewritten Description:"
    )


def build_vcr_annotation_prompt(*, reasoning: str) -> str:
    return (
        "Task instructions:\n"
        "You are tasked with evaluating the accuracy of a complex reasoning answer for an image. "
        "Identify hallucinated object, attribute, and relationship claims, explain why they are "
        "hallucinations, assign severity scores, and provide a corrected rewritten answer. "
        "Do not tag more than three hallucination sentences.\n\n"
        f"{render_severity_rubric()}\n\n"
        f"Complex reasoning:\n{reasoning}\n\n"
        "Output Format:\nQuestion and Tagged Answer:\nReasons:\nScores:\nRewritten Answer:"
    )


def render_ddg_prompt(*, question: str, response_text: str) -> str:
    return (
        "DDG task: identify fine-grained hallucination descriptions in a vision-language response.\n"
        "Inspect the image, question, and response. List hallucinated object, attribute, and relationship claims only.\n\n"
        f"Question:\n{question or response_text}\n\n"
        f"Response:\n{response_text or question}\n\n"
        "Return the hallucinated claim spans grouped under <object>, <attribute>, and <relationship>."
    )


def render_vcr_prompt(*, question: str, response_text: str, critique_report: str) -> str:
    return (
        "VCR task: verify whether each critique is visually grounded.\n"
        "Keep critiques only when the image contradicts the response claim.\n\n"
        f"Question:\n{question or response_text}\n\n"
        f"Response:\n{response_text or question}\n\n"
        f"Critiques:\n{critique_report}\n\n"
        "Return the verified critiques in the same Tags/Scores structure."
    )


def build_detector_prompt(*, question: str, response_text: str) -> str:
    return (
        "You are the FAIF hallucination detector used for paper-path reproduction.\n"
        "Given the image, the task context, and a candidate response, output exactly one of:\n"
        "NO HALLUCINATION\n"
        "or a fine-grained Tags/Scores report.\n\n"
        "Tags format:\n"
        "Tags:\n"
        "<object>\n"
        "1. hallucinated object claim\n"
        "<attribute>\n"
        "1. hallucinated attribute claim\n"
        "<relationship>\n"
        "1. hallucinated relationship claim\n\n"
        "Scores format:\n"
        "Scores:\n"
        "<object>\n"
        "1. Evidence span: Major (3 points): concise visual rationale\n\n"
        f"{render_severity_rubric()}\n\n"
        "Rules:\n"
        "- Use only the three hallucination types: object, attribute, relationship.\n"
        "- If no hallucination is present, output NO HALLUCINATION exactly.\n"
        "- Do not include confidence, calibration, prose preambles, or markdown.\n\n"
        f"Question or task context:\n{question or response_text}\n\n"
        f"Candidate response:\n{response_text or question}"
    )


def _format_critiques(critiques: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in critiques:
        critique = _as_dict(item)
        h_type = critique.get("hallucination_type", "unknown")
        severity = critique.get("severity_label", "unknown")
        score = critique.get("severity_score")
        evidence = (critique.get("evidence_text") or "").strip()
        rationale = (critique.get("rationale") or "").strip()
        severity_text = f"{severity}/{score}" if score is not None else str(severity)
        if evidence and rationale:
            lines.append(f"- [{h_type}; {severity_text}] {evidence}: {rationale}")
        elif rationale:
            lines.append(f"- [{h_type}; {severity_text}] {rationale}")
        elif evidence:
            lines.append(f"- [{h_type}; {severity_text}] {evidence}")
    return "\n".join(lines) if lines else "- (no structured critiques)"


def build_rewrite_prompt(record: Any) -> str:
    row = _as_dict(record)
    question = row.get("question", "") or row.get("response_text", "") or ""
    original = row.get("response_text", "") or row.get("original_response", "") or ""
    critiques = list(row.get("critiques") or row.get("detected_critiques") or [])
    return (
        "You are rewriting a hallucinated vision-language response for the FAIF paper path.\n"
        "Use the image and verified critiques to produce the preferred answer.\n\n"
        f"Question or task context:\n{question}\n\n"
        f"Original response:\n{original}\n\n"
        f"Detected hallucination tags and reasons:\n{_format_critiques(critiques)}\n\n"
        "Instructions:\n"
        "- The tags may not always be correct; rely on the image and remove only unsupported claims.\n"
        "- Correct or remove every identified hallucination.\n"
        "- Preserve accurate visual content from the original response.\n"
        "- Do not add unsupported objects, attributes, relations, counts, actions, or identities.\n"
        "- Output only the rewritten answer."
    )


def build_paper_rewrite_prompt(record: Any) -> str:
    return build_rewrite_prompt(record)


def build_api_critic_feedback_prompt(record: Any, initial_rewrite: str) -> str:
    row = _as_dict(record)
    question = row.get("question", "") or row.get("response_text", "") or ""
    original = row.get("response_text", "") or row.get("original_response", "") or ""
    critiques = list(row.get("critiques") or row.get("detected_critiques") or [])
    return (
        "You are a critic for a hallucination-mitigation rewrite of a vision-language response.\n"
        "Your job is to provide feedback only. Do not write the final answer.\n\n"
        f"Question or task context:\n{question}\n\n"
        f"Original response:\n{original}\n\n"
        f"Detected hallucination tags and reasons from the local detector:\n{_format_critiques(critiques)}\n\n"
        f"Initial rewritten response:\n{initial_rewrite}\n\n"
        "Check the initial rewrite against the image and detector findings.\n"
        "Return concise feedback about remaining hallucinations, unsupported additions, removed correct details, "
        "or over-edits. If the rewrite is acceptable, return exactly: NO CRITICAL ISSUES.\n"
        "Do not provide a replacement answer. Do not use markdown tables."
    )


def build_feedback_revision_prompt(record: Any) -> str:
    row = _as_dict(record)
    question = row.get("question", "") or row.get("response_text", "") or ""
    original = row.get("response_text", "") or row.get("original_response", "") or ""
    initial = row.get("initial_rewrite_response", "") or row.get("initial_rewrite", "") or ""
    feedback = row.get("api_feedback", "") or ""
    critiques = list(row.get("critiques") or row.get("detected_critiques") or [])
    return (
        "You are revising a hallucination-mitigation rewrite for the final preference pair.\n"
        "Use the image, local detector findings, the initial rewrite, and critic feedback.\n\n"
        f"Question or task context:\n{question}\n\n"
        f"Original response:\n{original}\n\n"
        f"Detected hallucination tags and reasons:\n{_format_critiques(critiques)}\n\n"
        f"Initial rewritten response:\n{initial}\n\n"
        f"Critic feedback:\n{feedback}\n\n"
        "Instructions:\n"
        "- The detector and critic feedback may be wrong; rely on the image.\n"
        "- Remove unsupported claims and preserve correct supported details.\n"
        "- Do not add unsupported objects, attributes, relations, counts, actions, or identities.\n"
        "- Output only the final revised answer."
    )


__all__ = [
    "API_CRITIC_PROMPT_VERSION",
    "DETECTOR_PROMPT_VERSION",
    "DDG_ANNOTATION_PROMPT_VERSION",
    "DDG_PROMPT_VERSION",
    "VCR_ANNOTATION_PROMPT_VERSION",
    "VCR_PROMPT_VERSION",
    "REWRITE_PROMPT_VERSION",
    "SEVERITY_RUBRIC_VERSION",
    "REVISION_PROMPT_VERSION",
    "build_api_critic_feedback_prompt",
    "build_ddg_annotation_prompt",
    "build_detector_prompt",
    "build_feedback_revision_prompt",
    "build_paper_rewrite_prompt",
    "build_rewrite_prompt",
    "build_vcr_annotation_prompt",
    "render_ddg_prompt",
    "render_severity_rubric",
    "render_vcr_prompt",
    "severity_rubric_text",
]
