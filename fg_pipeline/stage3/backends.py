"""Stage 3 verification backends.

The default backend remains a deterministic heuristic verifier suitable for
local smoke runs. Research runs use hosted Gemini judges, optionally paired
with a local LLaVA judge or hosted OpenAI judge for cross-family validation.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request

from fg_pipeline.stage3.prompts import PROMPT_VERSION, build_vote_prompt
from fg_pipeline.stage3.schemas import VoteDecision


VOTE_COUNT = 3
APPROVALS_REQUIRED = 2
VOTE_POLICY_VERSION = "heuristic_v1"
GEMINI_LLAVA_TWO_VOTE_POLICY_VERSION = "gemini_llava_two_vote_v1"
GEMINI_TWO_VOTE_POLICY_VERSION = "gemini_two_vote_v1"
GEMINI_OPENAI_TWO_VOTE_POLICY_VERSION = "gemini_openai_two_vote_v1"

_CRITERIA = {
    1: "hallucination_removal",
    2: "content_preservation",
    3: "overall_preference",
}
_GEMINI_LLAVA_FAMILY_BY_VOTE = {
    1: "gemini",
    2: "llava",
}
_GEMINI_TWO_VOTE_FAMILY_BY_VOTE = {
    1: "gemini",
    2: "gemini",
}
_GEMINI_OPENAI_FAMILY_BY_VOTE = {
    1: "gemini",
    2: "openai",
}
_WORD_RE = re.compile(r"\w+")


def _require_existing_path(path_value: str | None, *, context: str) -> None:
    path = Path(path_value or "")
    if path.exists():
        return
    raise FileNotFoundError(f"{context} path not found: {path_value}")


@runtime_checkable
class VerificationBackend(Protocol):
    """Minimal Stage-3-facing contract for any verification backend."""

    name: str
    policy_version: str
    approval_families_required: tuple[str, ...]

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        """Return one vote over a Stage 2 rewrite candidate."""
        ...


class VerificationError(ValueError):
    """Raised when Stage 3 verification cannot proceed in strict mode."""


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str | None) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _extract_fields(record: Any) -> tuple[str, str, list[dict[str, Any]]]:
    if hasattr(record, "original_response"):
        original = record.original_response or ""
        rewrite = record.rewrite_response or ""
        critiques = list(record.critiques or [])
    else:
        original = record.get("original_response", "") or ""
        rewrite = record.get("rewrite_response", "") or ""
        critiques = list(record.get("critiques") or [])
    return original, rewrite, critiques


def _evidence_terms(critiques: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        evidence = (critique.get("evidence_text") or "").strip()
        if evidence:
            terms.append(evidence)
    return terms


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if not candidate:
        raise ValueError("empty judge response")
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = _extract_balanced_json(candidate)
        if payload is None:
            payload = _fallback_parse_judge_text(candidate)
        if payload is None:
            snippet = _shorten_raw_output(candidate)
            raise ValueError(f"could not parse judge response: {snippet}")
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON is not an object")
    return payload


def _extract_balanced_json(text: str) -> dict[str, Any] | None:
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[start:index + 1]
                    try:
                        payload = json.loads(fragment)
                    except json.JSONDecodeError:
                        break
                    return payload if isinstance(payload, dict) else None
    return None


def _fallback_parse_judge_text(text: str) -> dict[str, Any] | None:
    compact = " ".join(text.strip().split())
    if not compact:
        return None

    approved: bool | None = None
    decision_match = re.search(
        r"\b(?:approved?|decision)\s*[:=\-]\s*(true|false|yes|no|approve|reject|pass|fail)\b",
        compact,
        flags=re.IGNORECASE,
    )
    if decision_match:
        approved = _parse_boolean(decision_match.group(1))
    else:
        lowered = compact.lower()
        if re.search(r"\b(do not approve|not approved|reject|rejected|false|no|fail|failed)\b", lowered):
            approved = False
        elif re.search(r"\b(approve|approved|true|yes|pass|passed)\b", lowered):
            approved = True

    if approved is None:
        return None

    reason = " ".join(compact.split()[:24])
    reason_match = re.search(r"\breason\s*[:=\-]\s*(.+)$", compact, flags=re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    return {
        "approved": approved,
        "reason": reason or "fallback parsed judge response",
    }


def _shorten_raw_output(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return repr(compact)
    return repr(compact[:limit] + "...")


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "approve", "approved", "pass"}:
            return True
        if lowered in {"false", "no", "reject", "rejected", "fail"}:
            return False
    raise ValueError(f"cannot coerce {value!r} to bool")


def _build_parse_failure_vote(
    *,
    vote_index: int,
    criterion: str,
    backend_name: str,
    model_family: str,
    error: Exception,
) -> VoteDecision:
    return VoteDecision(
        vote_index=vote_index,
        criterion=criterion,
        approved=False,
        reason=f"judge parse failure: {error}",
        model_family=model_family,
        backend_name=backend_name,
    )


class HeuristicVerificationBackend:
    """Deterministic local verifier for Stage 3.

    This backend is deliberately simple. It is useful for pipeline bring-up and
    tests, not as a final research judge.
    """

    name = "heuristic"
    policy_version = VOTE_POLICY_VERSION
    approval_families_required: tuple[str, ...] = ()

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _CRITERIA:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..{VOTE_COUNT}")

        original, rewrite, critiques = _extract_fields(record)
        original_norm = _normalize_text(original)
        rewrite_norm = _normalize_text(rewrite)
        if not original_norm or not rewrite_norm:
            raise VerificationError("Stage 2 record is missing original_response or rewrite_response")

        changed = original_norm != rewrite_norm
        evidence_terms = _evidence_terms(critiques)
        present_terms = [
            term for term in evidence_terms if term and re.search(r"\b" + re.escape(term) + r"\b", original, flags=re.IGNORECASE)
        ]
        removed_terms = [
            term for term in present_terms if not re.search(r"\b" + re.escape(term) + r"\b", rewrite, flags=re.IGNORECASE)
        ]
        corrected_marker = "[corrected]" in rewrite_norm

        original_tokens = _tokenize(original)
        rewrite_tokens = _tokenize(rewrite)
        overlap = len(set(original_tokens) & set(rewrite_tokens))
        overlap_ratio = overlap / len(set(original_tokens) or {""})
        length_ratio = len(rewrite_tokens) / max(len(original_tokens), 1)

        criterion = _CRITERIA[vote_index]
        approved = False
        reason = ""

        if criterion == "hallucination_removal":
            if present_terms:
                required = max(1, (len(present_terms) + 1) // 2)
                approved = changed and len(removed_terms) >= required
                reason = (
                    f"removed {len(removed_terms)}/{len(present_terms)} evidence span(s); "
                    f"changed={changed}"
                )
            else:
                approved = changed and not corrected_marker
                reason = f"no evidence spans available; changed={changed}; corrected_marker={corrected_marker}"

        elif criterion == "content_preservation":
            approved = (
                changed
                and not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            reason = (
                f"length_ratio={length_ratio:.3f}; overlap_ratio={overlap_ratio:.3f}; "
                f"changed={changed}; corrected_marker={corrected_marker}"
            )

        elif criterion == "overall_preference":
            removal_ok = (
                (len(removed_terms) > 0 if present_terms else changed and not corrected_marker)
            )
            preservation_ok = (
                not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            approved = changed and removal_ok and preservation_ok
            reason = (
                f"removal_ok={removal_ok}; preservation_ok={preservation_ok}; "
                f"changed={changed}"
            )

        return VoteDecision(
            vote_index=vote_index,
            criterion=criterion,
            approved=approved,
            reason=reason,
            model_family="heuristic",
            backend_name=self.name,
        )


class _LlavaJudgeRuntime:
    family = "llava"

    def __init__(
        self,
        model_path: str,
        *,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._model_base = model_base
        self._conv_mode = conv_mode
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = device
        self._bundle: tuple[Any, Any, Any] | None = None

    def _ensure_llava_on_path(self) -> None:
        llava_root = (
            Path(__file__).resolve().parent.parent.parent
            / "hsa_dpo"
            / "models"
            / "llava-v1_5"
        )
        path_str = str(llava_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    def _load(self) -> tuple[Any, Any, Any]:
        if self._bundle is not None:
            return self._bundle
        self._ensure_llava_on_path()
        from llava.model.builder import load_pretrained_model  # type: ignore[import]
        from llava.mm_utils import get_model_name_from_path  # type: ignore[import]

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=self._model_path,
            model_base=self._model_base,
            model_name=get_model_name_from_path(self._model_path),
            device_map={"": self._device} if self._device else "auto",
            device=self._device or "cuda",
        )
        model.eval()
        self._bundle = (tokenizer, model, image_processor)
        return self._bundle

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def judge(self, record: dict[str, Any], criterion: str) -> str:
        import torch
        from PIL import Image as PILImage

        tokenizer, model, image_processor = self._load()
        self._ensure_llava_on_path()
        from llava.constants import (  # type: ignore[import]
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # type: ignore[import]
        from llava.mm_utils import process_images, tokenizer_image_token  # type: ignore[import]

        prompt_text = build_vote_prompt(record, criterion)
        image_path = self._resolved_image(record)
        image_tensor = None
        use_image = False
        if image_path:
            pil_image = PILImage.open(image_path).convert("RGB")
            image_tensor = process_images([pil_image], image_processor, model.config).to(
                model.device,
                dtype=torch.float16,
            )
            use_image = True

        if use_image:
            if getattr(model.config, "mm_use_im_start_end", False):
                user_prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt_text
                )
            else:
                user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        else:
            user_prompt = prompt_text

        conv = conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        if use_image:
            input_ids = tokenizer_image_token(
                full_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

        generate_kwargs: dict[str, Any] = {
            "do_sample": self._temperature > 0.0,
            "max_new_tokens": self._max_new_tokens,
            "use_cache": True,
        }
        if self._temperature > 0.0:
            generate_kwargs["temperature"] = self._temperature
        else:
            generate_kwargs["num_beams"] = 1
        if use_image and image_tensor is not None:
            generate_kwargs["images"] = image_tensor

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generate_kwargs)
        return tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()


class _GeminiJudgeRuntime:
    family = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        image_root: str | None = None,
        max_output_tokens: int = 64,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise EnvironmentError("gemini_llava_two_vote requires GEMINI_API_KEY or GOOGLE_API_KEY")
        self._model = model
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._retries = max(1, retries)

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def _image_part(self, image_path: Path) -> dict[str, Any]:
        data = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }

    def judge(self, record: dict[str, Any], criterion: str) -> str:
        prompt = build_vote_prompt(record, criterion)
        parts: list[dict[str, Any]] = []
        image_path = self._resolved_image(record)
        if image_path is not None:
            parts.append(self._image_part(image_path))
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "approved": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["approved", "reason"],
                    "propertyOrdering": ["approved", "reason"],
                },
            },
        }
        response = self._request(payload)
        try:
            return str(response["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini response shape: {_shorten_raw_output(json.dumps(response))}") from exc

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
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
            )
            try:
                with urllib_request.urlopen(req, timeout=self._timeout_seconds) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self._retries - 1:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Gemini API HTTP {exc.code}: {_shorten_raw_output(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"Gemini API request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini API request failed: {last_error}")


class _OpenAIJudgeRuntime:
    family = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        image_root: str | None = None,
        max_output_tokens: int = 128,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise EnvironmentError("gemini_openai_two_vote requires OPENAI_API_KEY")
        self._model = model
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._retries = max(1, retries)

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def _image_url_part(self, image_path: Path) -> dict[str, Any]:
        data = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": "low",
            },
        }

    def judge(self, record: dict[str, Any], criterion: str) -> str:
        prompt = build_vote_prompt(record, criterion)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_path = self._resolved_image(record)
        if image_path is not None:
            content.append(self._image_url_part(image_path))

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._request(payload)
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected OpenAI response shape: {_shorten_raw_output(json.dumps(response))}") from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self._retries):
            req = urllib_request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib_request.urlopen(req, timeout=self._timeout_seconds) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self._retries - 1:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenAI API HTTP {exc.code}: {_shorten_raw_output(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI API request failed: {last_error}")


class GeminiLlavaTwoVoteBackend:
    """Hosted Gemini + local LLaVA backend with a fixed two-vote decision.

    Vote 1: Gemini hallucination removal
    Vote 2: LLaVA content preservation

    A pair passes only when both votes approve, so the final decision uses one
    hosted Gemini vote and one local LLaVA vote.
    """

    name = "gemini_llava_two_vote"
    policy_version = GEMINI_LLAVA_TWO_VOTE_POLICY_VERSION
    approval_families_required = ("gemini", "llava")
    vote_count = 2
    approvals_required = 2

    def __init__(
        self,
        *,
        llava_model_path: str,
        llava_model_base: str | None = None,
        llava_conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        gemini_model: str = "gemini-2.5-flash-lite",
        gemini_max_output_tokens: int = 64,
        llava_max_new_tokens: int = 64,
        gemini_temperature: float = 0.0,
        llava_temperature: float = 0.0,
        llava_device: str | None = None,
        gemini_runtime: Any | None = None,
        llava_runtime: Any | None = None,
    ) -> None:
        if llava_runtime is None:
            _require_existing_path(llava_model_path, context="gemini_llava_two_vote llava_model_path")
            if llava_model_base is not None:
                _require_existing_path(llava_model_base, context="gemini_llava_two_vote llava_model_base")
        self._gemini = gemini_runtime or _GeminiJudgeRuntime(
            model=gemini_model,
            image_root=image_root,
            max_output_tokens=gemini_max_output_tokens,
            temperature=gemini_temperature,
        )
        self._llava = llava_runtime or _LlavaJudgeRuntime(
            llava_model_path,
            model_base=llava_model_base,
            conv_mode=llava_conv_mode,
            image_root=image_root,
            max_new_tokens=llava_max_new_tokens,
            temperature=llava_temperature,
            device=llava_device,
        )

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _GEMINI_LLAVA_FAMILY_BY_VOTE:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..2")
        criterion = _CRITERIA[vote_index]
        family = _GEMINI_LLAVA_FAMILY_BY_VOTE[vote_index]
        runtime = self._gemini if family == "gemini" else self._llava
        try:
            raw = runtime.judge(record, criterion)
            payload = _extract_json_object(raw)
            approved = _parse_boolean(payload.get("approved"))
            reason = str(payload.get("reason") or "").strip() or "no reason provided"
            return VoteDecision(
                vote_index=vote_index,
                criterion=criterion,
                approved=approved,
                reason=reason,
                model_family=family,
                backend_name=self.name,
            )
        except Exception as exc:
            if strict:
                raise VerificationError(f"{family} judge failed for vote {vote_index}: {exc}") from exc
            return _build_parse_failure_vote(
                vote_index=vote_index,
                criterion=criterion,
                backend_name=self.name,
                model_family=family,
                error=exc,
            )


class GeminiTwoVoteBackend:
    """Hosted Gemini backend with a fixed two-vote decision.

    Vote 1: Gemini hallucination removal
    Vote 2: Gemini content preservation

    This is the fastest Stage 3 path. A pair passes only when both Gemini votes
    approve. It does not provide cross-family validation.
    """

    name = "gemini_two_vote"
    policy_version = GEMINI_TWO_VOTE_POLICY_VERSION
    approval_families_required = ("gemini",)
    vote_count = 2
    approvals_required = 2
    supports_row_parallelism = True

    def __init__(
        self,
        *,
        image_root: str | None = None,
        gemini_model: str = "gemini-2.5-flash-lite",
        gemini_max_output_tokens: int = 64,
        gemini_temperature: float = 0.0,
        gemini_runtime: Any | None = None,
    ) -> None:
        self._gemini = gemini_runtime or _GeminiJudgeRuntime(
            model=gemini_model,
            image_root=image_root,
            max_output_tokens=gemini_max_output_tokens,
            temperature=gemini_temperature,
        )

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _GEMINI_TWO_VOTE_FAMILY_BY_VOTE:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..2")
        criterion = _CRITERIA[vote_index]
        family = _GEMINI_TWO_VOTE_FAMILY_BY_VOTE[vote_index]
        try:
            raw = self._gemini.judge(record, criterion)
            payload = _extract_json_object(raw)
            approved = _parse_boolean(payload.get("approved"))
            reason = str(payload.get("reason") or "").strip() or "no reason provided"
            return VoteDecision(
                vote_index=vote_index,
                criterion=criterion,
                approved=approved,
                reason=reason,
                model_family=family,
                backend_name=self.name,
            )
        except Exception as exc:
            if strict:
                raise VerificationError(f"{family} judge failed for vote {vote_index}: {exc}") from exc
            return _build_parse_failure_vote(
                vote_index=vote_index,
                criterion=criterion,
                backend_name=self.name,
                model_family=family,
                error=exc,
            )


class GeminiOpenAITwoVoteBackend:
    """Hosted Gemini + OpenAI backend with a fixed two-vote decision.

    Vote 1: Gemini hallucination removal
    Vote 2: OpenAI content preservation

    A pair passes only when both judges approve, so API failures or either
    rejection send the row to Stage 4 repair.
    """

    name = "gemini_openai_two_vote"
    policy_version = GEMINI_OPENAI_TWO_VOTE_POLICY_VERSION
    approval_families_required = ("gemini", "openai")
    vote_count = 2
    approvals_required = 2
    supports_row_parallelism = True

    def __init__(
        self,
        *,
        image_root: str | None = None,
        gemini_model: str = "gemini-2.5-flash-lite",
        gemini_max_output_tokens: int = 128,
        gemini_temperature: float = 0.0,
        openai_model: str = "gpt-4o-mini",
        openai_max_output_tokens: int = 128,
        openai_temperature: float = 0.0,
        gemini_runtime: Any | None = None,
        openai_runtime: Any | None = None,
    ) -> None:
        self._gemini = gemini_runtime or _GeminiJudgeRuntime(
            model=gemini_model,
            image_root=image_root,
            max_output_tokens=gemini_max_output_tokens,
            temperature=gemini_temperature,
        )
        self._openai = openai_runtime or _OpenAIJudgeRuntime(
            model=openai_model,
            image_root=image_root,
            max_output_tokens=openai_max_output_tokens,
            temperature=openai_temperature,
        )

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _GEMINI_OPENAI_FAMILY_BY_VOTE:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..2")
        criterion = _CRITERIA[vote_index]
        family = _GEMINI_OPENAI_FAMILY_BY_VOTE[vote_index]
        runtime = self._gemini if family == "gemini" else self._openai
        try:
            raw = runtime.judge(record, criterion)
            payload = _extract_json_object(raw)
            approved = _parse_boolean(payload.get("approved"))
            reason = str(payload.get("reason") or "").strip() or "no reason provided"
            return VoteDecision(
                vote_index=vote_index,
                criterion=criterion,
                approved=approved,
                reason=reason,
                model_family=family,
                backend_name=self.name,
            )
        except Exception as exc:
            if strict:
                raise VerificationError(f"{family} judge failed for vote {vote_index}: {exc}") from exc
            return _build_parse_failure_vote(
                vote_index=vote_index,
                criterion=criterion,
                backend_name=self.name,
                model_family=family,
                error=exc,
            )


def evaluate_votes(backend: VerificationBackend, votes: list[VoteDecision]) -> tuple[bool, dict[str, Any]]:
    approvals = sum(1 for vote in votes if vote.approved)
    approved_families = sorted(
        {vote.model_family for vote in votes if vote.approved and vote.model_family}
    )
    approvals_required = int(getattr(backend, "approvals_required", APPROVALS_REQUIRED))
    required_families = tuple(getattr(backend, "approval_families_required", ()) or ())
    families_ok = all(family in approved_families for family in required_families)
    passed = approvals >= approvals_required and families_ok
    return passed, {
        "approved_families": approved_families,
        "approval_families_required": list(required_families),
        "cross_family_approval_ok": families_ok,
        "prompt_version": PROMPT_VERSION,
    }


def get_backend(name: str, **kwargs: Any) -> VerificationBackend:
    key = (name or "").strip().lower()
    if key == HeuristicVerificationBackend.name:
        return HeuristicVerificationBackend()
    if key == GeminiLlavaTwoVoteBackend.name:
        llava_model_path = kwargs.get("llava_model_path") or ""
        if not llava_model_path:
            raise ValueError("gemini_llava_two_vote requires --llava-model-path")
        return GeminiLlavaTwoVoteBackend(
            llava_model_path=llava_model_path,
            llava_model_base=kwargs.get("llava_model_base"),
            llava_conv_mode=kwargs.get("llava_conv_mode", "vicuna_v1"),
            image_root=kwargs.get("image_root"),
            gemini_model=kwargs.get("gemini_model", "gemini-2.5-flash-lite"),
            gemini_max_output_tokens=int(kwargs.get("gemini_max_output_tokens", 64)),
            llava_max_new_tokens=int(kwargs.get("llava_max_new_tokens", 64)),
            gemini_temperature=float(kwargs.get("gemini_temperature", 0.0)),
            llava_temperature=float(kwargs.get("llava_temperature", 0.0)),
            llava_device=kwargs.get("llava_device"),
            gemini_runtime=kwargs.get("gemini_runtime"),
            llava_runtime=kwargs.get("llava_runtime"),
        )
    if key == GeminiTwoVoteBackend.name:
        return GeminiTwoVoteBackend(
            image_root=kwargs.get("image_root"),
            gemini_model=kwargs.get("gemini_model", "gemini-2.5-flash-lite"),
            gemini_max_output_tokens=int(kwargs.get("gemini_max_output_tokens", 64)),
            gemini_temperature=float(kwargs.get("gemini_temperature", 0.0)),
            gemini_runtime=kwargs.get("gemini_runtime"),
        )
    if key == GeminiOpenAITwoVoteBackend.name:
        return GeminiOpenAITwoVoteBackend(
            image_root=kwargs.get("image_root"),
            gemini_model=kwargs.get("gemini_model", "gemini-2.5-flash-lite"),
            gemini_max_output_tokens=int(kwargs.get("gemini_max_output_tokens", 128)),
            gemini_temperature=float(kwargs.get("gemini_temperature", 0.0)),
            openai_model=kwargs.get("openai_model", "gpt-4o-mini"),
            openai_max_output_tokens=int(kwargs.get("openai_max_output_tokens", 128)),
            openai_temperature=float(kwargs.get("openai_temperature", 0.0)),
            gemini_runtime=kwargs.get("gemini_runtime"),
            openai_runtime=kwargs.get("openai_runtime"),
        )
    available = ", ".join((
        HeuristicVerificationBackend.name,
        GeminiLlavaTwoVoteBackend.name,
        GeminiTwoVoteBackend.name,
        GeminiOpenAITwoVoteBackend.name,
    ))
    raise ValueError(f"unknown stage3 backend {name!r}; available: {available}")


__all__ = [
    "VerificationBackend",
    "VerificationError",
    "HeuristicVerificationBackend",
    "GeminiLlavaTwoVoteBackend",
    "GeminiTwoVoteBackend",
    "GeminiOpenAITwoVoteBackend",
    "VOTE_COUNT",
    "APPROVALS_REQUIRED",
    "VOTE_POLICY_VERSION",
    "GEMINI_LLAVA_TWO_VOTE_POLICY_VERSION",
    "GEMINI_TWO_VOTE_POLICY_VERSION",
    "GEMINI_OPENAI_TWO_VOTE_POLICY_VERSION",
    "evaluate_votes",
    "get_backend",
]
