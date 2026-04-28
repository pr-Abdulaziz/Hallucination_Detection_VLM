"""Validate released HSA-DPO preference pairs with API vision judges.

This path starts from the released preference dataset rather than rebuilding
preference pairs from detector outputs. Gemini/OpenAI judge whether the released
chosen response is a usable improvement over the rejected response.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request

from fg_pipeline.io_utils import count_jsonl_rows, ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION, normalize_space


DEFAULT_INPUT = Path("hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl")
DEFAULT_IMAGE_ROOT = Path("hsa_dpo/data/images")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
ZERO_SHOT_OUTPUT_DIR = Path("output/fghd/released_pref_stage3")
TWO_SHOT_OUTPUT_DIR = Path("output/fghd/released_pref_stage3_2shot_experiment")
DEFAULT_OUTPUT_DIR = ZERO_SHOT_OUTPUT_DIR
DEFAULT_ACCEPTED = DEFAULT_OUTPUT_DIR / "validated_preferences.jsonl"
DEFAULT_REJECTED = DEFAULT_OUTPUT_DIR / "rejected_for_repair.jsonl"
DEFAULT_AUDIT = DEFAULT_OUTPUT_DIR / "judgement_records.jsonl"
DEFAULT_STATS = DEFAULT_OUTPUT_DIR / "stats.json"

PROMPT_VERSION = "released_pref_validation_v1"
TWO_SHOT_PROMPT_VERSION = "released_pref_validation_2shot_v1"
DEFAULT_PROMPT_MODE = "zero_shot"
PROMPT_VERSIONS = {
    "zero_shot": PROMPT_VERSION,
    "two_shot": TWO_SHOT_PROMPT_VERSION,
}
OUTPUT_DIRS = {
    "zero_shot": ZERO_SHOT_OUTPUT_DIR,
    "two_shot": TWO_SHOT_OUTPUT_DIR,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate released HSA-DPO preference pairs with Gemini/OpenAI vision judges.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory. Defaults to a prompt-mode-specific folder.",
    )
    parser.add_argument("--accepted-out", type=Path, default=None)
    parser.add_argument("--rejected-out", type=Path, default=None)
    parser.add_argument("--audit-out", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument(
        "--api-judge",
        type=str,
        default="gemini_openai",
        help="Judge family: gemini, openai, or gemini_openai.",
    )
    parser.add_argument("--gemini-model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--decision-rule", choices=("either", "both"), default="either")
    parser.add_argument(
        "--prompt-mode",
        choices=("zero_shot", "two_shot"),
        default=DEFAULT_PROMPT_MODE,
        help="Validation prompt style. Select zero_shot for the paper baseline or two_shot for the few-shot experiment.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Number of preference rows to validate concurrently.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def _shorten(raw: str, limit: int = 700) -> str:
    text = normalize_space(raw)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


def _image_for_row(row: dict[str, Any], image_root: Path) -> Path | None:
    image_value = row.get("image")
    candidates: list[Path] = []
    if image_value:
        image_path = Path(str(image_value))
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.extend([image_root / image_path, image_path])
    if row.get("id") not in (None, ""):
        candidates.append(image_root / f"{row.get('id')}.jpg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _prompt_version(prompt_mode: str) -> str:
    return PROMPT_VERSIONS.get(prompt_mode, PROMPT_VERSION)


def _output_dir(prompt_mode: str) -> Path:
    return OUTPUT_DIRS.get(prompt_mode, DEFAULT_OUTPUT_DIR)


def _resolve_output_paths(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or _output_dir(args.prompt_mode)
    if args.accepted_out is None:
        args.accepted_out = output_dir / "validated_preferences.jsonl"
    if args.rejected_out is None:
        args.rejected_out = output_dir / "rejected_for_repair.jsonl"
    if args.audit_out is None:
        args.audit_out = output_dir / "judgement_records.jsonl"
    if args.stats_out is None:
        args.stats_out = output_dir / "stats.json"


def _two_shot_examples() -> str:
    return (
        "Two calibration examples follow. They are text-only examples for the decision rubric; "
        "do not assume their visual details are present in the target image.\n\n"
        "Example 1:\n"
        "Question:\nDescribe the image.\n"
        "Rejected/original response:\nThe image shows a child holding a red umbrella while a dog walks nearby.\n"
        "Chosen/rewrite response:\nThe image shows a child holding an umbrella.\n"
        "Original hallucination tags and severity:\n"
        "object hallucination: dog nearby; attribute hallucination: red umbrella; severity major\n"
        "Expected JSON:\n"
        '{"approved": true, "reason": "The chosen response removes the unsupported dog and color claims while preserving the grounded umbrella detail."}\n\n'
        "Example 2:\n"
        "Question:\nWhat is happening in the scene?\n"
        "Rejected/original response:\nA man is sitting on a bench with a bicycle beside him.\n"
        "Chosen/rewrite response:\nA man is sitting on a bench beside a motorcycle in a park.\n"
        "Original hallucination tags and severity:\n"
        "object hallucination: bicycle beside him; severity moderate\n"
        "Expected JSON:\n"
        '{"approved": false, "reason": "The chosen response introduces unsupported motorcycle and park details, so it is not a reliable improvement."}\n\n'
    )


def _build_validation_prompt(row: dict[str, Any], *, prompt_mode: str = "zero_shot") -> str:
    example_block = _two_shot_examples() if prompt_mode == "two_shot" else ""
    return (
        "You are validating a hallucination-mitigation preference pair for a vision-language model.\n"
        "Use the image as the source of truth.\n"
        "Return exactly one JSON object and no other text.\n"
        "The JSON object must have exactly these keys: approved, reason.\n"
        "approved must be a JSON boolean.\n"
        "reason must be one concise sentence.\n\n"
        "Approve when the chosen rewrite clearly improves factual visual grounding over the rejected response, "
        "removes or reduces the tagged hallucination, and does not introduce a new important hallucination.\n"
        "Reject when the chosen rewrite is still unsupported by the image, loses essential correct content, "
        "or is not clearly better than the rejected response.\n"
        "Be conservative, but do not reject only because the chosen response is shorter.\n\n"
        f"{example_block}"
        "Target pair to judge:\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Rejected/original response:\n{row.get('rejected', '')}\n\n"
        f"Chosen/rewrite response:\n{row.get('chosen', '')}\n\n"
        f"Original hallucination tags and severity:\n{row.get('rejected_tag_text', '')}\n\n"
        "Return only the JSON object now."
    )


class _JudgeRuntime:
    family = "api"

    def __init__(
        self,
        *,
        image_root: Path,
        max_output_tokens: int | None,
        prompt_mode: str = "zero_shot",
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        retries: int = 3,
    ) -> None:
        self._image_root = image_root
        self._max_output_tokens = max_output_tokens
        self._prompt_mode = prompt_mode
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._retries = max(1, retries)

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class _GeminiRuntime(_JudgeRuntime):
    family = "gemini"

    def __init__(self, *, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise EnvironmentError("Gemini validation requires GEMINI_API_KEY or GOOGLE_API_KEY")
        self._model = model

    def _image_part(self, image_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        image_path = _image_for_row(row, self._image_root)
        if image_path is not None:
            parts.append(self._image_part(image_path))
        parts.append({"text": _build_validation_prompt(row, prompt_mode=self._prompt_mode)})

        generation_config: dict[str, Any] = {"temperature": self._temperature}
        if self._max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self._max_output_tokens
        payload = {"contents": [{"parts": parts}], "generationConfig": generation_config}
        raw = self._request(payload)
        try:
            text = str(raw["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini validation response: {_shorten(json.dumps(raw))}") from exc
        return _judge_payload(self.family, text, model=self._model)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self._retries):
            req = urllib_request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            )
            try:
                with urllib_request.urlopen(req, timeout=self._timeout_seconds) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self._retries - 1:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Gemini validation HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"Gemini validation failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini validation failed: {last_error}")


class _OpenAIRuntime(_JudgeRuntime):
    family = "openai"

    def __init__(self, *, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise EnvironmentError("OpenAI validation requires OPENAI_API_KEY")
        self._model = model

    def _image_url_part(self, image_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "low"},
        }

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _build_validation_prompt(row, prompt_mode=self._prompt_mode)}
        ]
        image_path = _image_for_row(row, self._image_root)
        if image_path is not None:
            content.append(self._image_url_part(image_path))
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
        }
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        raw = self._request(payload)
        try:
            text = str(raw["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected OpenAI validation response: {_shorten(json.dumps(raw))}") from exc
        return _judge_payload(self.family, text, model=self._model)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self._retries):
            req = urllib_request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib_request.urlopen(req, timeout=self._timeout_seconds) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self._retries - 1:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenAI validation HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"OpenAI validation failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI validation failed: {last_error}")


def _judge_payload(family: str, raw_text: str, *, model: str | None = None) -> dict[str, Any]:
    parsed = _extract_json(raw_text)
    payload = {
        "family": family,
        "approved": bool(parsed.get("approved")),
        "reason": normalize_space(str(parsed.get("reason", ""))),
        "raw_output": raw_text,
    }
    if model:
        payload["model"] = model
    return payload


def _build_judges(args: argparse.Namespace) -> list[_JudgeRuntime]:
    common = {
        "image_root": args.image_root,
        "max_output_tokens": args.max_output_tokens,
        "prompt_mode": args.prompt_mode,
        "temperature": 0.0,
        "timeout_seconds": args.timeout_seconds,
        "retries": args.retries,
    }
    selected = args.api_judge.lower()
    selected_families = set(selected.split("_"))
    judges: list[_JudgeRuntime] = []
    if "gemini" in selected_families:
        judges.append(_GeminiRuntime(model=args.gemini_model, **common))
    if "openai" in selected_families:
        judges.append(_OpenAIRuntime(model=args.openai_model, **common))
    if not judges:
        raise ValueError("--api-judge must include one of: gemini, openai")
    return judges


def _validate_row(
    row: dict[str, Any],
    judges: list[_JudgeRuntime],
    *,
    decision_rule: str,
    strict: bool,
    image_root: Path,
    prompt_mode: str = "zero_shot",
) -> tuple[dict[str, Any], dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    for judge in judges:
        try:
            votes.append(judge.judge(row))
        except Exception as exc:
            if strict:
                raise
            votes.append(
                {
                    "family": judge.family,
                    "approved": False,
                    "reason": f"judge_error: {exc}",
                    "raw_output": "",
                    "error": str(exc),
                }
            )

    approved = all(vote["approved"] for vote in votes) if decision_rule == "both" else any(
        vote["approved"] for vote in votes
    )
    image_path = _image_for_row(row, image_root)
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_stage": "released_pref_stage3_validation",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "prompt_version": _prompt_version(prompt_mode),
            "prompt_mode": prompt_mode,
            "decision_rule": decision_rule,
            "api_votes": votes,
        }
    )
    preference = dict(row)
    if image_path is not None:
        preference["image"] = str(image_path)
    preference["validation_approved"] = approved
    preference["metadata"] = metadata
    audit = {
        "id": row.get("id"),
        "approved": approved,
        "decision_rule": decision_rule,
        "votes": votes,
        "image": str(image_path) if image_path else None,
        "question": row.get("question", ""),
    }
    return preference, audit


class _Stats:
    def __init__(
        self,
        *,
        input_path: str,
        api_judge: str,
        decision_rule: str,
        prompt_mode: str = "zero_shot",
        gemini_model: str | None = None,
        openai_model: str | None = None,
        workers: int = 1,
    ) -> None:
        self.input_path = input_path
        self.api_judge = api_judge
        self.decision_rule = decision_rule
        self.prompt_mode = prompt_mode
        self.gemini_model = gemini_model
        self.openai_model = openai_model
        self.workers = workers
        self.total_rows = 0
        self.accepted_rows = 0
        self.rejected_rows = 0
        self.vote_counts: Counter[str] = Counter()

    def record(self, audit: dict[str, Any]) -> None:
        self.total_rows += 1
        if audit.get("approved"):
            self.accepted_rows += 1
        else:
            self.rejected_rows += 1
        for vote in audit.get("votes") or []:
            key = f"{vote.get('family')}:{bool(vote.get('approved'))}"
            self.vote_counts[key] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "stage": "released_pref_stage3_validation",
            "prompt_version": _prompt_version(self.prompt_mode),
            "prompt_mode": self.prompt_mode,
            "input_path": self.input_path,
            "api_judge": self.api_judge,
            "gemini_model": self.gemini_model,
            "openai_model": self.openai_model,
            "workers": self.workers,
            "decision_rule": self.decision_rule,
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "vote_counts": dict(self.vote_counts),
        }


def _iter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    judges: list[_JudgeRuntime],
    stats: _Stats,
    decision_rule: str,
    strict: bool,
    image_root: Path,
    limit: int | None,
    total: int | None,
    prompt_mode: str = "zero_shot",
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    selected_rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        selected_rows.append((index, row))

    def validate_indexed(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        index, row = item
        preference, audit = _validate_row(
            row,
            judges,
            decision_rule=decision_rule,
            strict=strict,
            image_root=image_root,
            prompt_mode=prompt_mode,
        )
        return index, preference, audit

    workers = max(1, int(workers or 1))
    results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    if workers == 1:
        iterator = maybe_tqdm(selected_rows, desc="Released pref Stage 3 validate", total=total)
        for item in iterator:
            results.append(validate_indexed(item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(validate_indexed, item) for item in selected_rows]
            iterator = maybe_tqdm(
                as_completed(futures),
                desc=f"Released pref Stage 3 validate ({workers} workers)",
                total=len(futures),
            )
            for future in iterator:
                results.append(future.result())

    for _, preference, audit in sorted(results, key=lambda result: result[0]):
        stats.record(audit)
        audits.append(audit)
        if audit["approved"]:
            accepted.append(preference)
        else:
            rejected.append(preference)
    return accepted, rejected, audits


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _resolve_output_paths(args)
    if not args.input.exists():
        print(f"Released preference input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        judges = _build_judges(args)
    except (ValueError, EnvironmentError) as exc:
        print(f"Released pref Stage 3 validation setup error: {exc}", file=sys.stderr)
        return 2

    total = count_jsonl_rows(args.input)
    if args.limit is not None:
        total = min(total, args.limit)
    stats = _Stats(
        input_path=str(args.input),
        api_judge=args.api_judge,
        decision_rule=args.decision_rule,
        prompt_mode=args.prompt_mode,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        workers=args.workers,
    )
    try:
        accepted, rejected, audits = _iter_rows(
            read_jsonl(args.input),
            judges=judges,
            stats=stats,
            decision_rule=args.decision_rule,
            strict=args.strict,
            image_root=args.image_root,
            limit=args.limit,
            total=total,
            prompt_mode=args.prompt_mode,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"Released pref Stage 3 validation failed: {exc}", file=sys.stderr)
        return 3

    write_jsonl(ensure_parent_dir(args.accepted_out), accepted)
    write_jsonl(ensure_parent_dir(args.rejected_out), rejected)
    write_jsonl(ensure_parent_dir(args.audit_out), audits)
    payload = stats.to_dict()
    payload.update(
        {
            "accepted_out": str(args.accepted_out),
            "rejected_out": str(args.rejected_out),
            "audit_out": str(args.audit_out),
        }
    )
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Released pref Stage 3 validated {payload['total_rows']} row(s): "
        f"{payload['accepted_rows']} accepted, {payload['rejected_rows']} rejected"
    )
    print(f"Accepted preferences -> {args.accepted_out}")
    print(f"Rejected for repair -> {args.rejected_out}")
    print(f"Validation stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
