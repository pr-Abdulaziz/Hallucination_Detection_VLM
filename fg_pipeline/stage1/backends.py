"""Backend seam for Stage 1 critique detection.

The Stage 1 contract is fixed around :class:`Stage1Record`. Two backends are
provided:

* ``released_annotations`` — parser over the released HSA-DPO supervision.
* ``llava_detector`` — local LLaVA-based detector inference backend that emits
  the same normalized Stage 1 contract.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request

from fg_pipeline.stage1.parser import ParseResult, parse_assessed_annotation, parse_detection_row
from fg_pipeline.stage1.prompts import PROMPT_VERSION, build_detector_prompt, coerce_stage1_inputs
from fg_pipeline.stage1.schemas import CritiqueItem, Stage1Record


@runtime_checkable
class CritiqueDetectorBackend(Protocol):
    """Minimal Stage-1-facing contract for any critique detection backend."""

    name: str

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        """Return a :class:`ParseResult` for one source row."""
        ...


class ReleasedAnnotationBackend:
    """Default backend: parse released HSA-DPO detection supervision."""

    name: str = "released_annotations"

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        return parse_detection_row(row, strict=strict)


class LlavaDetectorBackend:
    """Local LLaVA detector backend for Stage 1 research runs."""

    name: str = "llava_detector"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._model_base = model_base
        self._conv_mode = conv_mode
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
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
        )
        model.eval()
        self._bundle = (tokenizer, model, image_processor)
        return self._bundle

    def _resolved_image(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def _generate_annotation(self, row: dict[str, Any]) -> tuple[str, str, str]:
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

        question, response_text = coerce_stage1_inputs(row)
        prompt_text = build_detector_prompt(question=question, response_text=response_text)

        image_path = self._resolved_image(row.get("image"))
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
        annotation_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
        return question, response_text, annotation_text

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        question, response_text, annotation_text = self._generate_annotation(row)
        result = parse_assessed_annotation(
            row_id=row.get("id"),
            image=row.get("image"),
            assessed_text=response_text,
            annotation_text=annotation_text,
            source=self.name,
            strict=strict,
        )
        result.record.question = question
        result.record.metadata["prompt_version"] = PROMPT_VERSION
        return result


def _shorten(raw: str, limit: int = 700) -> str:
    text = " ".join((raw or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_api_judge_prompt(*, question: str, response_text: str) -> str:
    base_prompt = build_detector_prompt(question=question, response_text=response_text)
    return (
        f"{base_prompt}\n"
        "Additional API judge constraints:\n"
        "- Judge against the image, not against prior assumptions.\n"
        "- Only report claims that are false, unsupported, or visually unverifiable.\n"
        "- Do not report true visual details as hallucinations.\n"
        "- In Tags and Scores, section headers must be exactly one of: <object>, <attribute>, <relationship>.\n"
        "- Do not create object-specific headers such as <umbrella>, <person>, or <color>.\n"
        "- Every tag item must have one matching score item under the same exact type header.\n"
        "- Each score item must contain Minor (1 points), Moderate (2 points), or Major (3 points).\n"
        "- If a quantity, color, object, or relation is uncertain or not clearly visible, mark it as hallucinated.\n"
        "Return only the required NO HALLUCINATION or Tags/Scores report now.\n"
    )


class _ApiJudgeRuntime:
    family = "api"

    def __init__(
        self,
        *,
        image_root: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        retries: int = 3,
    ) -> None:
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._retries = max(1, retries)

    def _resolved_image(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def judge(self, row: dict[str, Any], *, question: str, response_text: str) -> str:
        raise NotImplementedError


class _GeminiJudgeRuntime(_ApiJudgeRuntime):
    family = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise EnvironmentError("Gemini judge requires GEMINI_API_KEY or GOOGLE_API_KEY")
        self._model = model

    def _image_part(self, image_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    def judge(self, row: dict[str, Any], *, question: str, response_text: str) -> str:
        prompt_text = _build_api_judge_prompt(question=question, response_text=response_text)
        parts: list[dict[str, Any]] = []
        image_path = self._resolved_image(row.get("image"))
        if image_path is not None:
            parts.append(self._image_part(image_path))
        parts.append({"text": prompt_text})

        generation_config: dict[str, Any] = {"temperature": self._temperature}
        if self._max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self._max_output_tokens
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }
        response = self._request(payload)
        try:
            return str(response["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini judge response: {_shorten(json.dumps(response))}") from exc

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
                    raise RuntimeError(f"Gemini judge HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"Gemini judge request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini judge request failed: {last_error}")


class _OpenAIJudgeRuntime(_ApiJudgeRuntime):
    family = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise EnvironmentError("OpenAI judge requires OPENAI_API_KEY")
        self._model = model

    def _image_url_part(self, image_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "low"},
        }

    def judge(self, row: dict[str, Any], *, question: str, response_text: str) -> str:
        prompt_text = _build_api_judge_prompt(question=question, response_text=response_text)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        image_path = self._resolved_image(row.get("image"))
        if image_path is not None:
            content.append(self._image_url_part(image_path))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
        }
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        response = self._request(payload)
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected OpenAI judge response: {_shorten(json.dumps(response))}") from exc

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
                    raise RuntimeError(f"OpenAI judge HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"OpenAI judge request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI judge request failed: {last_error}")


class ApiJudgeDetectorBackend:
    """Gemini/OpenAI API hallucination judge backend for paper-path Stage 3."""

    name = "api_judge"

    def __init__(
        self,
        *,
        image_root: str | None = None,
        api_judge: str = "gemini_openai",
        gemini_model: str = "gemini-2.5-flash-lite",
        openai_model: str = "gpt-4o-mini",
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        retries: int = 3,
        decision_rule: str = "either",
    ) -> None:
        common = {
            "image_root": image_root,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        }
        selected = api_judge.lower()
        selected_families = set(selected.split("_"))
        runtimes: list[_ApiJudgeRuntime] = []
        if "gemini" in selected_families:
            runtimes.append(_GeminiJudgeRuntime(model=gemini_model, **common))
        if "openai" in selected_families:
            runtimes.append(_OpenAIJudgeRuntime(model=openai_model, **common))
        if not runtimes:
            raise ValueError("api_judge must include one of: gemini, openai")
        if decision_rule not in {"either", "both"}:
            raise ValueError("api_judge decision_rule must be either or both")
        self._runtimes = runtimes
        self._api_judge = selected
        self._decision_rule = decision_rule

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        question, response_text = coerce_stage1_inputs(row)
        parsed: list[ParseResult] = []
        raw_outputs: list[dict[str, str]] = []
        warnings: list[str] = []

        for runtime in self._runtimes:
            annotation_text = runtime.judge(row, question=question, response_text=response_text)
            raw_outputs.append({"family": runtime.family, "raw_detector_output": annotation_text})
            result = parse_assessed_annotation(
                row_id=row.get("id"),
                image=row.get("image"),
                assessed_text=response_text,
                annotation_text=annotation_text,
                source=f"{self.name}:{runtime.family}",
                strict=strict,
            )
            parsed.append(result)
            warnings.extend(result.warnings)

        hallucinated_votes = [result.record.is_hallucinated for result in parsed]
        if self._decision_rule == "both":
            is_hallucinated = all(hallucinated_votes)
        else:
            is_hallucinated = any(hallucinated_votes)

        critiques: list[CritiqueItem] = []
        if is_hallucinated:
            for result in parsed:
                if not result.record.is_hallucinated:
                    continue
                for critique in result.record.critiques:
                    critiques.append(
                        CritiqueItem(
                            index=len(critiques) + 1,
                            hallucination_type=critique.hallucination_type,
                            severity_label=critique.severity_label,
                            severity_score=critique.severity_score,
                            rationale=critique.rationale,
                            evidence_text=critique.evidence_text,
                            source_tag_text=critique.source_tag_text,
                            source_score_text=critique.source_score_text,
                        )
                    )

        raw_annotation_text = "\n\n".join(
            f"[{item['family']}]\n{item['raw_detector_output']}" for item in raw_outputs
        )
        record = Stage1Record(
            id=row.get("id"),
            image=row.get("image"),
            question=question,
            response_text=response_text,
            is_hallucinated=is_hallucinated,
            critiques=critiques,
            metadata={
                "source": self.name,
                "prompt_version": PROMPT_VERSION,
                "api_judge": self._api_judge,
                "api_judge_decision_rule": self._decision_rule,
                "api_votes": [
                    {
                        "family": raw_outputs[idx]["family"],
                        "is_hallucinated": parsed[idx].record.is_hallucinated,
                        "critique_count": len(parsed[idx].record.critiques),
                    }
                    for idx in range(len(parsed))
                ],
                "raw_api_outputs": raw_outputs,
                "raw_annotation_text": raw_annotation_text,
            },
        )
        if warnings:
            record.metadata["parse_warnings"] = warnings
        return ParseResult(record=record, warnings=warnings)


def get_backend(name: str, **kwargs: Any) -> CritiqueDetectorBackend:
    """Return an instantiated backend by registered name."""

    key = (name or "").strip().lower()
    if key == ReleasedAnnotationBackend.name:
        return ReleasedAnnotationBackend()
    if key == LlavaDetectorBackend.name:
        model_path = kwargs.get("model_path") or ""
        if not model_path:
            raise ValueError("llava_detector requires --model-path / model_path kwarg")
        return LlavaDetectorBackend(
            model_path=model_path,
            model_base=kwargs.get("model_base"),
            conv_mode=kwargs.get("conv_mode", "vicuna_v1"),
            image_root=kwargs.get("image_root"),
            max_new_tokens=int(kwargs.get("max_new_tokens", 384)),
            temperature=float(kwargs.get("temperature", 0.0)),
        )
    if key == ApiJudgeDetectorBackend.name:
        max_output_tokens = kwargs.get("judge_max_output_tokens")
        return ApiJudgeDetectorBackend(
            image_root=kwargs.get("image_root"),
            api_judge=kwargs.get("api_judge", "gemini_openai"),
            gemini_model=kwargs.get("gemini_model", "gemini-2.5-flash-lite"),
            openai_model=kwargs.get("openai_model", "gpt-4o-mini"),
            max_output_tokens=int(max_output_tokens) if max_output_tokens not in {None, ""} else None,
            temperature=float(kwargs.get("temperature", 0.0)),
            timeout_seconds=int(kwargs.get("judge_timeout_seconds", 60)),
            retries=int(kwargs.get("judge_retries", 3)),
            decision_rule=kwargs.get("api_decision_rule", "either"),
        )
    available = ", ".join(
        (
            ReleasedAnnotationBackend.name,
            LlavaDetectorBackend.name,
            ApiJudgeDetectorBackend.name,
        )
    )
    raise ValueError(
        f"unknown stage1 backend {name!r}; available: {available}"
    )


__all__ = [
    "CritiqueDetectorBackend",
    "ReleasedAnnotationBackend",
    "LlavaDetectorBackend",
    "ApiJudgeDetectorBackend",
    "Stage1Record",
    "get_backend",
]
