"""Paper Stage 4: detect-then-rewrite preference construction."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request

from fg_pipeline.io_utils import ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import (
    PAPER_PIPELINE_VERSION,
    aggregate_severity,
    normalize_space,
    resolve_existing_image,
)
from fg_pipeline.paper.prompts import (
    API_CRITIC_PROMPT_VERSION,
    REVISION_PROMPT_VERSION,
    REWRITE_PROMPT_VERSION,
    build_api_critic_feedback_prompt,
    build_feedback_revision_prompt,
    build_paper_rewrite_prompt,
)
from fg_pipeline.paths import (
    DEFAULT_PAPER_STAGE3_DETECTIONS,
    DEFAULT_PAPER_STAGE4_PREFERENCES,
    DEFAULT_PAPER_STAGE4_REWRITES,
    DEFAULT_PAPER_STAGE4_STATS,
)
from fg_pipeline.schemas import PreferenceCleanRecord
from fg_pipeline.stage2.backends import LlavaRewriteBackend, TemplateRewriteBackend


@runtime_checkable
class PaperRewriteBackend(Protocol):
    name: str

    def rewrite(self, record: dict[str, Any], *, strict: bool = False) -> str:
        ...


class TemplatePaperRewriteBackend:
    name = "template"

    def __init__(self) -> None:
        self._backend = TemplateRewriteBackend()

    def rewrite(self, record: dict[str, Any], *, strict: bool = False) -> str:
        adapted = {
            "response_text": record.get("original_response", ""),
            "critiques": record.get("detected_critiques") or [],
        }
        return self._backend.rewrite(adapted, strict=strict)


class LlavaPaperRewriteBackend:
    name = "llava"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        image_root: str | None = None,
    ) -> None:
        self._runtime = LlavaRewriteBackend(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            image_root=image_root,
            prompt_builder=build_paper_rewrite_prompt,
        )

    def rewrite(self, record: dict[str, Any], *, strict: bool = False) -> str:
        return self._runtime.rewrite(record, strict=strict)


def _shorten(raw: str, limit: int = 500) -> str:
    text = normalize_space(raw)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class _ApiCriticRuntime:
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

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None


class _GeminiCriticRuntime(_ApiCriticRuntime):
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
            raise EnvironmentError("Gemini critic requires GEMINI_API_KEY or GOOGLE_API_KEY")
        self._model = model

    def _image_part(self, image_path: Path) -> dict[str, Any]:
        data = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }

    def feedback(self, record: dict[str, Any], initial_rewrite: str) -> str:
        prompt = build_api_critic_feedback_prompt(record, initial_rewrite)
        parts: list[dict[str, Any]] = []
        image_path = self._resolved_image(record)
        if image_path is not None:
            parts.append(self._image_part(image_path))
        parts.append({"text": prompt})

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
            raise ValueError(f"unexpected Gemini critic response: {_shorten(json.dumps(response))}") from exc

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
                    raise RuntimeError(f"Gemini critic HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"Gemini critic request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini critic request failed: {last_error}")


class _OpenAICriticRuntime(_ApiCriticRuntime):
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
            raise EnvironmentError("OpenAI critic requires OPENAI_API_KEY")
        self._model = model

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

    def feedback(self, record: dict[str, Any], initial_rewrite: str) -> str:
        prompt = build_api_critic_feedback_prompt(record, initial_rewrite)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_path = self._resolved_image(record)
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
            raise ValueError(f"unexpected OpenAI critic response: {_shorten(json.dumps(response))}") from exc

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
                    raise RuntimeError(f"OpenAI critic HTTP {exc.code}: {_shorten(detail)}") from exc
            except urllib_error.URLError as exc:
                last_error = exc
                if attempt == self._retries - 1:
                    raise RuntimeError(f"OpenAI critic request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI critic request failed: {last_error}")


class _CombinedCritic:
    def __init__(self, runtimes: list[_ApiCriticRuntime]) -> None:
        if not runtimes:
            raise ValueError("at least one critic runtime is required")
        self._runtimes = runtimes
        self.name = "_".join(runtime.family for runtime in runtimes)

    def feedback(self, record: dict[str, Any], initial_rewrite: str) -> tuple[str, list[dict[str, str]]]:
        items: list[dict[str, str]] = []
        lines: list[str] = []
        for runtime in self._runtimes:
            text = runtime.feedback(record, initial_rewrite).strip()
            items.append({"family": runtime.family, "feedback": text})
            lines.append(f"[{runtime.family}]\n{text}")
        return "\n\n".join(lines), items


class LlavaApiCriticPaperRewriteBackend:
    name = "llava_api_critic"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        image_root: str | None = None,
        critic: _CombinedCritic,
    ) -> None:
        self._critic = critic
        self._runtime = LlavaRewriteBackend(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            image_root=image_root,
            prompt_builder=self._prompt_for_record,
        )

    def _prompt_for_record(self, record: dict[str, Any]) -> str:
        if record.get("_stage4_revision_mode"):
            return build_feedback_revision_prompt(record)
        return build_paper_rewrite_prompt(record)

    def rewrite_with_metadata(
        self,
        record: dict[str, Any],
        *,
        strict: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        initial = self._runtime.rewrite(record, strict=strict).strip()
        feedback, feedback_items = self._critic.feedback(record, initial)
        revision_record = dict(record)
        revision_record.update(
            {
                "_stage4_revision_mode": True,
                "initial_rewrite_response": initial,
                "api_feedback": feedback,
                "api_feedback_items": feedback_items,
            }
        )
        final = self._runtime.rewrite(revision_record, strict=strict).strip()
        return final, {
            "initial_rewrite_response": initial,
            "api_feedback": feedback,
            "api_feedback_items": feedback_items,
            "api_critic": self._critic.name,
            "api_critic_prompt_version": API_CRITIC_PROMPT_VERSION,
            "revision_prompt_version": REVISION_PROMPT_VERSION,
        }

    def rewrite(self, record: dict[str, Any], *, strict: bool = False) -> str:
        final, _ = self.rewrite_with_metadata(record, strict=strict)
        return final


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper Stage 4: rewrite detector-flagged hallucinated rows into D_pref.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PAPER_STAGE3_DETECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAPER_STAGE4_REWRITES)
    parser.add_argument("--preferences-out", type=Path, default=DEFAULT_PAPER_STAGE4_PREFERENCES)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_PAPER_STAGE4_STATS)
    parser.add_argument("--backend", type=str, default="template")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--api-critic",
        type=str,
        default="gemini_openai",
        help="Critic family for --backend llava_api_critic: gemini, openai, gemini_openai, or none.",
    )
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic-max-output-tokens", type=int, default=None)
    parser.add_argument("--critic-timeout-seconds", type=int, default=60)
    parser.add_argument("--critic-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not drop rows whose image file is missing; useful only for template smoke tests.",
    )
    return parser


def _get_backend(args: argparse.Namespace) -> PaperRewriteBackend:
    if args.backend == "template":
        return TemplatePaperRewriteBackend()
    if args.backend == "llava":
        if not args.model_path:
            raise ValueError("--backend llava requires --model-path")
        return LlavaPaperRewriteBackend(
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            image_root=args.image_root,
        )
    if args.backend == "llava_api_critic":
        if not args.model_path:
            raise ValueError("--backend llava_api_critic requires --model-path")
        critic = _build_critic(args)
        return LlavaApiCriticPaperRewriteBackend(
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            image_root=args.image_root,
            critic=critic,
        )
    raise ValueError(
        "unknown paper Stage 4 backend {!r}; expected template, llava, or llava_api_critic".format(
            args.backend
        )
    )


def _build_critic(args: argparse.Namespace) -> _CombinedCritic:
    common = {
        "image_root": args.image_root,
        "max_output_tokens": args.critic_max_output_tokens,
        "temperature": 0.0,
        "timeout_seconds": args.critic_timeout_seconds,
        "retries": args.critic_retries,
    }
    selected = args.api_critic.lower()
    runtimes: list[_ApiCriticRuntime] = []
    if selected in {"gemini", "gemini_openai", "openai_gemini"}:
        runtimes.append(_GeminiCriticRuntime(model=args.gemini_model, **common))
    if selected in {"openai", "gemini_openai", "openai_gemini"}:
        runtimes.append(_OpenAICriticRuntime(model=args.openai_model, **common))
    if selected == "none":
        raise ValueError("--api-critic none is only valid with --backend llava")
    if not runtimes:
        raise ValueError("--api-critic must be one of: gemini, openai, gemini_openai")
    return _CombinedCritic(runtimes)


class _Stats:
    def __init__(self, *, backend_name: str, input_path: str) -> None:
        self.backend = backend_name
        self.input_path = input_path
        self.input_rows = 0
        self.predicted_non_hallucinated_skipped = 0
        self.rewrite_rows_processed = 0
        self.rewrite_records_emitted = 0
        self.preference_pairs_emitted = 0
        self.skipped: Counter[str] = Counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "backend": self.backend,
            "input_path": self.input_path,
            "input_rows": self.input_rows,
            "predicted_non_hallucinated_skipped": self.predicted_non_hallucinated_skipped,
            "rewrite_rows_processed": self.rewrite_rows_processed,
            "rewrite_records_emitted": self.rewrite_records_emitted,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "skipped_rows_by_reason": dict(self.skipped),
            "prompt_version": REWRITE_PROMPT_VERSION,
        }


def _skip_reason(row: dict[str, Any], *, image_root: str, allow_missing_images: bool) -> str | None:
    if not row.get("is_hallucinated_pred"):
        return "predicted_non_hallucinated"
    if not normalize_space(row.get("original_response")):
        return "empty_original_response"
    if not row.get("image"):
        return "missing_image_path"
    if not allow_missing_images and resolve_existing_image(row.get("image"), image_root) is None:
        return "missing_image_file"
    return None


def _rewrite_record(
    row: dict[str, Any],
    rewrite_text: str,
    *,
    backend_name: str,
    rewrite_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    critiques = list(row.get("detected_critiques") or [])
    severity = row.get("response_severity_score")
    if not isinstance(severity, (int, float)):
        severity = aggregate_severity(critiques)
    rewrite_metadata = dict(rewrite_metadata or {})
    record = {
        "id": row.get("id"),
        "image": row.get("image"),
        "question": row.get("question", ""),
        "original_response": row.get("original_response", ""),
        "rewrite_response": rewrite_text,
        "detected_critiques": critiques,
        "response_severity_score": float(severity),
        "chosen": rewrite_text,
        "rejected": row.get("original_response", ""),
        "metadata": {
            "source_stage": "paper_stage4_rewrite",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "rewrite_backend": backend_name,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "detector_metadata": row.get("metadata") or {},
            **rewrite_metadata,
        },
    }
    if rewrite_metadata.get("initial_rewrite_response"):
        record["initial_rewrite_response"] = rewrite_metadata["initial_rewrite_response"]
    if rewrite_metadata.get("api_feedback"):
        record["api_feedback"] = rewrite_metadata["api_feedback"]
    if rewrite_metadata.get("api_feedback_items"):
        record["api_feedback_items"] = rewrite_metadata["api_feedback_items"]
    return record


def _preference_from_rewrite(record: dict[str, Any], *, backend_name: str) -> dict[str, Any]:
    severity = record.get("response_severity_score")
    if not isinstance(severity, (int, float)):
        severity = aggregate_severity(list(record.get("detected_critiques") or []))
    return PreferenceCleanRecord(
        id=record.get("id"),
        question=record.get("question") or record.get("original_response", ""),
        chosen=record.get("chosen") or record.get("rewrite_response", ""),
        rejected=record.get("rejected") or record.get("original_response", ""),
        chosen_score=1.0,
        rejected_score=float(severity),
        image=record.get("image"),
        metadata={
            "source_stage": "paper_stage4_preference",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "rewrite_backend": backend_name,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "response_severity_score": float(severity),
            "api_critic": (record.get("metadata") or {}).get("api_critic"),
            "api_critic_prompt_version": (record.get("metadata") or {}).get("api_critic_prompt_version"),
            "revision_prompt_version": (record.get("metadata") or {}).get("revision_prompt_version"),
        },
    ).to_dict()


def preference_from_rewrite(row: dict[str, Any], rewrite_text: str, *, backend_name: str) -> dict[str, Any]:
    """Build a paper Stage 4 preference from a detection row and rewrite."""

    detection_row = dict(row)
    if "original_response" not in detection_row:
        detection_row["original_response"] = detection_row.get("response_text", "")
    if "detected_critiques" not in detection_row:
        detection_row["detected_critiques"] = list(detection_row.get("critiques") or [])
    rewrite_record = _rewrite_record(detection_row, rewrite_text, backend_name=backend_name)
    return _preference_from_rewrite(rewrite_record, backend_name=backend_name)


def _backend_rewrite_with_metadata(
    backend: PaperRewriteBackend,
    row: dict[str, Any],
    *,
    strict: bool,
) -> tuple[str, dict[str, Any]]:
    rewrite_with_metadata = getattr(backend, "rewrite_with_metadata", None)
    if callable(rewrite_with_metadata):
        text, metadata = rewrite_with_metadata(row, strict=strict)
        return str(text), dict(metadata or {})
    return backend.rewrite(row, strict=strict), {}


def _iter_outputs(
    backend: PaperRewriteBackend,
    rows: Iterable[dict[str, Any]],
    *,
    stats: _Stats,
    strict: bool,
    limit: int | None,
    image_root: str,
    allow_missing_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewrite_records: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    progress = maybe_tqdm(rows, desc="Paper Stage 4 rewrite", total=None)
    processed = 0
    for row in progress:
        stats.input_rows += 1
        reason = _skip_reason(row, image_root=image_root, allow_missing_images=allow_missing_images)
        if reason == "predicted_non_hallucinated":
            stats.predicted_non_hallucinated_skipped += 1
            continue
        if reason:
            stats.skipped[reason] += 1
            if strict:
                raise ValueError(f"row id={row.get('id')!r} skipped: {reason}")
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        stats.rewrite_rows_processed += 1
        rewrite_text, rewrite_metadata = _backend_rewrite_with_metadata(backend, row, strict=strict)
        rewrite_text = rewrite_text.strip()
        if not rewrite_text:
            stats.skipped["empty_rewrite"] += 1
            if strict:
                raise ValueError(f"row id={row.get('id')!r} produced empty rewrite")
            continue
        if normalize_space(rewrite_text).lower() == normalize_space(row.get("original_response")).lower():
            stats.skipped["identical_rewrite"] += 1
            if strict:
                raise ValueError(f"row id={row.get('id')!r} produced identical rewrite")
            continue
        rewrite_record = _rewrite_record(
            row,
            rewrite_text,
            backend_name=backend.name,
            rewrite_metadata=rewrite_metadata,
        )
        preference_row = _preference_from_rewrite(rewrite_record, backend_name=backend.name)
        rewrite_records.append(rewrite_record)
        preference_rows.append(preference_row)
        stats.rewrite_records_emitted += 1
        stats.preference_pairs_emitted += 1
    return rewrite_records, preference_rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Paper Stage 4 input not found: {args.input}", file=sys.stderr)
        print("Run Paper Stage 3 first: bash scripts/run_paper_stage3_detect.sh", file=sys.stderr)
        return 2
    try:
        backend = _get_backend(args)
    except ValueError as exc:
        print(f"Paper Stage 4 backend error: {exc}", file=sys.stderr)
        return 2

    stats = _Stats(backend_name=backend.name, input_path=str(args.input))
    try:
        rewrite_records, preference_rows = _iter_outputs(
            backend,
            read_jsonl(args.input),
            stats=stats,
            strict=args.strict,
            limit=args.limit,
            image_root=args.image_root,
            allow_missing_images=args.allow_missing_images,
        )
    except ValueError as exc:
        print(f"Paper Stage 4 rewrite failed: {exc}", file=sys.stderr)
        return 3

    write_jsonl(ensure_parent_dir(args.output), rewrite_records)
    write_jsonl(ensure_parent_dir(args.preferences_out), preference_rows)
    payload = stats.to_dict()
    payload["output_path"] = str(args.output)
    payload["preferences_out"] = str(args.preferences_out)
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Paper Stage 4 wrote {payload['preference_pairs_emitted']} preference pair(s) "
        f"and {payload['rewrite_records_emitted']} rewrite record(s)"
    )
    print(f"Paper Stage 4 preferences -> {args.preferences_out}")
    print(f"Paper Stage 4 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
