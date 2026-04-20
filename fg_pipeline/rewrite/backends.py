"""Rewrite backends for Stage 4.

A backend takes the original prompt, candidate response, and the *already
filtered* list of reliable Stage-3 signals and returns a rewritten response
plus backend metadata. Heavy dependencies (torch, LLaVA) are imported lazily
so this module can be imported on a CPU box for smoke tests.

Two backends are provided:

- ``template`` — deterministic, offline, smoke-only scaffolding. Not the
  real Stage 4 rewriter; used to exercise the pipeline plumbing and tests.
- ``llava``   — the real rewrite path backed by the vendored LLaVA-v1.5
  model. Loads weights on first call; requires a GPU in practice.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Protocol, runtime_checkable

from fg_pipeline.rewrite.prompts import build_rewrite_prompt
from fg_pipeline.schemas import SentenceSignal


@runtime_checkable
class RewriteBackend(Protocol):
    """Return (rewritten_response, backend_metadata) for one record."""

    name: str

    def rewrite(
        self,
        prompt: str,
        response: str,
        signals: list[SentenceSignal],
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]: ...


def _span_of(signal: SentenceSignal) -> Optional[str]:
    meta = signal.metadata or {}
    span = meta.get("span")
    return span if isinstance(span, str) and span.strip() else None


class TemplateRewriteBackend:
    """Deterministic, offline rewriter for CPU smoke tests.

    Rewrites by removing flagged spans from the source response. It is *not*
    a real rewrite model — it is only meant to validate Stage 4 plumbing
    without a GPU. Every record produced by this backend is tagged
    ``rewrite_status="generated_smoke_only"`` in metadata so downstream
    consumers can refuse to treat its outputs as real Stage 4 data.
    """

    name = "template"

    def rewrite(
        self,
        prompt: str,
        response: str,
        signals: list[SentenceSignal],
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        out = response
        removed: list[str] = []
        for signal in signals:
            span = _span_of(signal)
            if span and span in out:
                out = out.replace(span, "[removed]", 1)
                removed.append(span)
        if out == response and signals:
            # No spans matched; append a conservative caveat so the rewritten
            # response still differs from the source, matching Stage 5's
            # expectation that chosen != rejected when rewrite ran.
            out = response.rstrip() + " [flagged content removed]"
        meta = {
            "rewrite_backend": self.name,
            "rewrite_status": "generated_smoke_only",
            "removed_spans": removed,
            "note": "template backend is smoke-only; not the real Stage 4 model",
        }
        return out, meta


class LLaVARewriteBackend:
    """Real Stage 4 rewriter backed by LLaVA-v1.5.

    Weights and torch deps are loaded lazily on first ``rewrite`` call. The
    prompt is built from :func:`build_rewrite_prompt` and the model generates
    the corrected response. Runs on GPU in practice; CPU is not supported.
    """

    name = "llava"

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        image_root: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        conv_template_name: str = "llava_v1",
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.image_root = image_root
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.conv_template_name = conv_template_name

        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._resolved_device: Optional[str] = None

    def _load(self) -> None:
        if self._loaded:
            return
        # Reuse the Stage 3 loader plumbing (path shim + AutoConfig patch).
        scorers = importlib.import_module("fg_pipeline.confidence.scorers_logprob")
        scorers._patch_autoconfig_register()
        scorers._ensure_vendored_llava_on_path()

        from llava.model.builder import load_pretrained_model  # type: ignore
        from llava.mm_utils import get_model_name_from_path  # type: ignore
        import torch  # type: ignore

        if self.device == "auto":
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = "auto"
        else:
            device_arg = self.device
            device_map = {"": device_arg}
        self._resolved_device = device_arg

        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.model_path,
            None,
            get_model_name_from_path(self.model_path),
            load_8bit=False,
            load_4bit=False,
            device_map=device_map,
            device=device_arg,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self._loaded = True

    def rewrite(
        self,
        prompt: str,
        response: str,
        signals: list[SentenceSignal],
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        self._load()

        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from llava.conversation import conv_templates  # type: ignore
        from llava.constants import (  # type: ignore
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.mm_utils import process_images, tokenizer_image_token  # type: ignore

        rewrite_prompt = build_rewrite_prompt(prompt, response, signals)

        image_path = context.get("image")
        image_tensor = None
        if image_path:
            from pathlib import Path

            root = self.image_root or "."
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = Path(root) / resolved
            if resolved.exists():
                image = Image.open(resolved).convert("RGB")
                image_tensor = process_images(
                    [image], self._image_processor, self._model.config
                ).to(self._model.device, dtype=torch.float16)

        use_im_tokens = getattr(self._model.config, "mm_use_im_start_end", False)
        if image_tensor is not None:
            if use_im_tokens:
                image_block = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
            else:
                image_block = DEFAULT_IMAGE_TOKEN
            user_turn = image_block + "\n" + rewrite_prompt
        else:
            user_turn = rewrite_prompt

        conv = conv_templates[self.conv_template_name].copy()
        conv.append_message(conv.roles[0], user_turn)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._model.device)
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        rewritten = self._tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        meta = {
            "rewrite_backend": self.name,
            "rewrite_status": "generated",
            "rewrite_model": self.model_path,
            "device": self._resolved_device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "image_used": image_tensor is not None,
        }
        return rewritten, meta


_BACKEND_REGISTRY: dict[str, str] = {
    "template": "fg_pipeline.rewrite.backends:TemplateRewriteBackend",
    "llava": "fg_pipeline.rewrite.backends:LLaVARewriteBackend",
}

_BACKEND_KWARG_WHITELIST: dict[str, set[str]] = {
    "template": set(),
    "llava": {
        "model_path",
        "device",
        "image_root",
        "max_new_tokens",
        "temperature",
        "conv_template_name",
    },
}


def _resolve_backend_class(name: str) -> type:
    try:
        target = _BACKEND_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"unknown rewrite backend {name!r}; available: {available}"
        ) from exc
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def get_backend(name: str, **kwargs: Any) -> RewriteBackend:
    cls = _resolve_backend_class(name)
    allowed = _BACKEND_KWARG_WHITELIST.get(name, set())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return cls(**filtered)


def backend_kwargs_whitelist(name: str) -> set[str]:
    return set(_BACKEND_KWARG_WHITELIST.get(name, set()))
