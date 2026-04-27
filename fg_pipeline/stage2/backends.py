"""Stage 2 rewrite backends.

``RewriteBackend`` is the protocol seam — any future backend (trained
LVLM, API call, etc.) can be plugged in without changing the Stage 2
output schema or pipeline logic.

Two backends are provided:

* ``TemplateRewriteBackend`` — deterministic smoke backend for local
  development and testing.  **NOT research-quality.**  It removes
  hallucination evidence spans from the original text using regex
  substitution.  Same input always yields the same output.

* ``LlavaRewriteBackend`` — real multimodal backend using the vendored
  LLaVA-v1.5 stack already in the repo.  Requires a local model path;
  lazy-loads the model on first call.  Not exercised by unit tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class RewriteBackend(Protocol):
    """Minimal Stage-2-facing contract for any rewrite backend."""

    name: str

    def rewrite(self, record: Any, *, strict: bool = False) -> str:
        """Return one corrected rewrite string for a hallucinated Stage 1 record."""
        ...


class TemplateRewriteBackend:
    """Deterministic smoke backend.

    WARNING: this backend is for local development and testing only.
    It is NOT research-quality.  It applies simple regex substitution
    to remove identified evidence spans from the original text, and
    appends a marker when nothing could be removed.
    """

    name: str = "template"

    def rewrite(self, record: Any, *, strict: bool = False) -> str:
        if hasattr(record, "response_text"):
            original: str = record.response_text or ""
            critiques = record.critiques or []
        else:
            original = record.get("response_text", "") or ""
            critiques = record.get("critiques", []) or []

        if not original.strip():
            if strict:
                raise RewriteError("empty response_text in Stage 1 record")
            return "[no original response to rewrite]"

        modified = original
        removed: list[str] = []

        for c in critiques:
            if hasattr(c, "to_dict"):
                c = c.to_dict()
            evidence = (c.get("evidence_text") or "").strip()
            if not evidence:
                continue
            # Remove the evidence span (case-insensitive) from the text.
            new_text, n_subs = re.subn(
                r"\b" + re.escape(evidence) + r"\b",
                "",
                modified,
                flags=re.IGNORECASE,
            )
            if n_subs:
                removed.append(evidence)
                modified = new_text

        # Normalize whitespace after removals.
        modified = re.sub(r"\s{2,}", " ", modified).strip()

        if not modified or modified == original.strip():
            # Nothing was removed — append a deterministic marker.
            modified = original.strip() + " [corrected]"

        return modified


class LlavaRewriteBackend:
    """Real multimodal rewrite backend using the vendored LLaVA-v1.5 stack.

    The model is lazy-loaded on the first :meth:`rewrite` call.

    Args:
        model_path: Local path to the LLaVA-v1.5 model (or LoRA adapter).
        model_base: Base model path when ``model_path`` is a LoRA adapter.
        conv_mode: Conversation template name (default ``"vicuna_v1"``).
        max_new_tokens: Maximum tokens to generate per rewrite.
        temperature: Sampling temperature; 0 for greedy decoding.
        image_root: Root directory used to resolve relative image paths.
    """

    name: str = "llava"

    def __init__(
        self,
        model_path: str,
        *,
        model_base: Optional[str] = None,
        conv_mode: str = "vicuna_v1",
        max_new_tokens: Optional[int] = 256,
        temperature: float = 0.0,
        image_root: Optional[str] = None,
        prompt_builder: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self._model_path = model_path
        self._model_base = model_base
        self._conv_mode = conv_mode
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._image_root = Path(image_root) if image_root else Path(".")
        self._prompt_builder = prompt_builder
        self._bundle: Optional[tuple] = None  # lazy loaded

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

    def _load(self) -> None:
        if self._bundle is not None:
            return
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

    def rewrite(self, record: Any, *, strict: bool = False) -> str:
        import torch
        from PIL import Image as PILImage

        self._load()
        tokenizer, model, image_processor = self._bundle  # type: ignore[misc]

        self._ensure_llava_on_path()
        from llava.constants import (  # type: ignore[import]
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # type: ignore[import]
        from llava.mm_utils import (  # type: ignore[import]
            process_images,
            tokenizer_image_token,
        )

        if self._prompt_builder is None:
            from fg_pipeline.stage2.prompts import build_rewrite_prompt

            prompt_text = build_rewrite_prompt(record)
        else:
            prompt_text = self._prompt_builder(record)

        # Resolve image path — prepend image_root if path is relative.
        image_path_str = (
            record.image if hasattr(record, "image") else record.get("image")
        )
        image_tensor = None
        use_image = False
        if image_path_str:
            candidate = Path(image_path_str)
            if not candidate.is_absolute():
                candidate = self._image_root / candidate
            if candidate.exists():
                pil_image = PILImage.open(candidate).convert("RGB")
                image_tensor = process_images(
                    [pil_image], image_processor, model.config
                ).to(model.device, dtype=torch.float16)
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
                full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer(
                full_prompt, return_tensors="pt"
            ).input_ids.to(model.device)

        generate_kwargs: dict[str, Any] = {
            "do_sample": self._temperature > 0.0,
            "use_cache": True,
        }
        if self._max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = self._max_new_tokens
        if self._temperature > 0.0:
            generate_kwargs["temperature"] = self._temperature
        else:
            generate_kwargs["num_beams"] = 1

        if use_image and image_tensor is not None:
            generate_kwargs["images"] = image_tensor

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generate_kwargs)

        result = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        return result


class RewriteError(ValueError):
    """Raised when a Stage 2 rewrite fails in strict mode."""


def get_backend(name: str, **kwargs: Any) -> RewriteBackend:
    """Return an instantiated rewrite backend by name.

    ``kwargs`` are forwarded to backends that require construction args
    (e.g. ``LlavaRewriteBackend`` needs ``model_path``).
    """

    key = (name or "").strip().lower()
    if key == TemplateRewriteBackend.name:
        return TemplateRewriteBackend()
    if key == LlavaRewriteBackend.name:
        model_path = kwargs.get("model_path") or ""
        if not model_path:
            raise ValueError(
                "llava rewrite backend requires --model-path / model_path kwarg"
            )
        return LlavaRewriteBackend(
            model_path=model_path,
            model_base=kwargs.get("model_base"),
            conv_mode=kwargs.get("conv_mode", "vicuna_v1"),
            max_new_tokens=int(kwargs.get("max_new_tokens", 256)),
            temperature=float(kwargs.get("temperature", 0.0)),
            image_root=kwargs.get("image_root"),
        )
    available = f"{TemplateRewriteBackend.name}, {LlavaRewriteBackend.name}"
    raise ValueError(
        f"unknown stage2 backend {name!r}; available: {available}"
    )


__all__ = [
    "RewriteBackend",
    "RewriteError",
    "TemplateRewriteBackend",
    "LlavaRewriteBackend",
    "get_backend",
]
