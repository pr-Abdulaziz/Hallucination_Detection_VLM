from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from fg_pipeline.eval.schemas import ModelSpec


def _ensure_llava_on_path() -> None:
    llava_root = (
        Path(__file__).resolve().parent.parent.parent
        / "hsa_dpo"
        / "models"
        / "llava-v1_5"
    )
    path_str = str(llava_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def validate_model_spec(spec: ModelSpec) -> None:
    if spec.kind == "lora" and not spec.model_base:
        raise ValueError(f"model {spec.model_id!r} is kind='lora' but model_base is missing")


def load_model_bundle(spec: ModelSpec):
    validate_model_spec(spec)
    _ensure_llava_on_path()
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    if spec.kind == "lora":
        from peft import PeftModel

        base_path = spec.model_base or spec.model_path
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=base_path,
            model_base=None,
            model_name=get_model_name_from_path(base_path),
        )
        model = PeftModel.from_pretrained(model, spec.model_path)
        model = model.merge_and_unload()
        model.eval()
        return tokenizer, model, image_processor, context_len

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=spec.model_path,
        model_base=spec.model_base,
        model_name=get_model_name_from_path(spec.model_path),
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def _load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_answers_for_records(
    spec: ModelSpec,
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    _ensure_llava_on_path()
    tokenizer, model, image_processor, _ = load_model_bundle(spec)

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    outputs: list[dict[str, Any]] = []
    for record in records:
        prompt = record.get("question") or record.get("prompt") or ""
        image_path = record.get("image")
        if not image_path:
            raise FileNotFoundError(f"record {record.get('id')} is missing image path")
        image = _load_image(image_path)
        image_tensor = process_images([image], image_processor, model.config).to(
            model.device,
            dtype=torch.float16,
        )
        if getattr(model.config, "mm_use_im_start_end", False):
            user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[spec.conv_mode].copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=spec.temperature > 0.0,
                temperature=spec.temperature,
                num_beams=spec.num_beams,
                max_new_tokens=spec.max_new_tokens,
                use_cache=True,
            )
        text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0].strip()
        outputs.append(
            {
                "id": record.get("id"),
                "question": prompt,
                "image": str(image_path),
                "text": text,
            }
        )
    del tokenizer, model, image_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
