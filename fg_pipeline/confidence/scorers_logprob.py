"""LogProbScorer — the locked Stage 3 c^j implementation.

c^j = teacher-forced next-token log-probability of the teacher's severity label
(Minor / Moderate / Major), given the image and the candidate response, under
the base LLaVA-v1.5 VLM.

Heavy dependencies (torch, transformers, vendored LLaVA code, PIL) are imported
lazily on first call to ``score`` so the module can be imported on a CPU-only
box for mocked unit tests.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDORED_LLAVA_PATH = REPO_ROOT / "hsa_dpo" / "models" / "llava-v1_5"

# LLaMA/LLaVA tokenizer splits the three severity labels into distinct leading
# BPE tokens. Using only the first token is sufficient to distinguish them and
# avoids dealing with multi-token sequences.
SEVERITY_LABELS = ("Minor", "Moderate", "Major")


def _patch_autoconfig_register() -> None:
    """transformers>=4.40 ships a native ``llava`` config; the vendored code
    re-registers it at import time and raises ``ValueError``. Tolerate it.
    """

    from transformers import AutoConfig

    original = AutoConfig.register

    def safe_register(model_type, config, exist_ok=False):
        try:
            return original(model_type, config, exist_ok=exist_ok)
        except ValueError:
            return original(model_type, config, exist_ok=True)

    AutoConfig.register = safe_register  # type: ignore[assignment]


def _ensure_vendored_llava_on_path() -> None:
    path_str = str(VENDORED_LLAVA_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class LogProbScorer:
    """Teacher-forced next-token log-probability scorer for Stage 3.

    One forward pass per signal. For each signal, the prompt is constructed to
    end at ``<type>\\n<span>: `` so the next-token distribution predicts the
    severity label. c^j = exp(log_softmax(logits/T)[teacher_first_token]).
    """

    name = "log_prob"

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        temperature: float = 1.0,
        image_root: Optional[str] = None,
        conv_template_name: str = "llava_v1",
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.temperature = float(temperature)
        self.image_root = image_root or str(REPO_ROOT)
        self.conv_template_name = conv_template_name

        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._conv_template = None
        self._severity_first_token_id: dict[str, int] = {}
        self._image_cache: dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        _patch_autoconfig_register()
        _ensure_vendored_llava_on_path()

        import torch  # noqa: F401  (warms up torch for the following imports)
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        device_arg = "cuda" if self.device in ("auto", "cuda") else self.device
        device_map = "auto" if self.device == "auto" else {"": device_arg}

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
        self._conv_template = conv_templates[self.conv_template_name]

        for label in SEVERITY_LABELS:
            ids = tokenizer(label, add_special_tokens=False).input_ids
            if not ids:
                raise RuntimeError(f"tokenizer produced no ids for severity label {label!r}")
            self._severity_first_token_id[label] = int(ids[0])

        self._loaded = True

    def _resolve_image(self, image_rel: Optional[str]):
        if not image_rel:
            raise FileNotFoundError("record is missing an image path")
        if image_rel in self._image_cache:
            return self._image_cache[image_rel]
        from PIL import Image

        candidate = Path(image_rel)
        if not candidate.is_absolute():
            candidate = Path(self.image_root) / candidate
        if not candidate.exists():
            raise FileNotFoundError(str(candidate))
        image = Image.open(candidate).convert("RGB")
        self._image_cache[image_rel] = image
        return image

    def _build_prompt(self, candidate_response: str, h_type: str, span: str) -> str:
        """USER turn + partial ASSISTANT turn ending at '<type>\\n<span>: '."""

        conv = self._conv_template.copy()
        user_msg = (
            "<image>\n"
            f"Candidate description: {candidate_response}\n\n"
            "Identify any hallucinations in the description. For each, output one "
            "line in the form:\n<type>\n<span>: <Severity> (<points> points): <reason>\n"
            "where <type> is one of object / attribute / relationship and "
            "<Severity> is one of Minor / Moderate / Major.\n"
            "If none, output: NO HALLUCINATION"
        )
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        base = conv.get_prompt()
        return f"{base}<{h_type}>\n{span}: "

    def score(
        self, signal_data: dict[str, Any], record_context: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        severity_label = (signal_data.get("severity_label") or "").capitalize()
        if severity_label not in self._severity_first_token_id and not self._loaded:
            # ensure tokenizer is available before validating
            pass
        try:
            self._ensure_loaded()
        except Exception as exc:
            return 0.0, {
                "scorer": self.name,
                "is_placeholder": False,
                "error": f"load_failed: {exc}",
            }

        if severity_label not in self._severity_first_token_id:
            return 0.0, {
                "scorer": self.name,
                "is_placeholder": False,
                "error": f"unknown_severity_label: {severity_label!r}",
            }

        try:
            image = self._resolve_image(record_context.get("image"))
        except FileNotFoundError as exc:
            return 0.0, {
                "scorer": self.name,
                "is_placeholder": False,
                "error": f"image_missing: {exc}",
            }

        import torch
        import torch.nn.functional as F
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        image_tensor = process_images([image], self._image_processor, self._model.config)
        image_tensor = image_tensor.to(self._model.device, dtype=self._model.dtype)

        prompt = self._build_prompt(
            candidate_response=record_context.get("candidate_response", ""),
            h_type=signal_data.get("hallucination_type", ""),
            span=signal_data.get("span", ""),
        )
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, images=image_tensor, use_cache=False)
        logits = outputs.logits[0, -1].float()

        temperature = max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        token_id = self._severity_first_token_id[severity_label]
        log_prob = float(log_probs[token_id].item())
        confidence = math.exp(log_prob)

        # Companion probs for diagnostics (useful for calibration / ECE later).
        companion = {
            label: float(log_probs[self._severity_first_token_id[label]].item())
            for label in SEVERITY_LABELS
        }

        return confidence, {
            "scorer": self.name,
            "method": "teacher_forced_next_token_logprob",
            "severity_label": severity_label,
            "token_id": token_id,
            "log_prob": log_prob,
            "temperature": self.temperature,
            "companion_log_probs": companion,
            "is_placeholder": False,
        }
