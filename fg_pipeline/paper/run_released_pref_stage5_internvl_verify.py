"""Verify LLaVA-repaired released preference rows with InternVL.

This stage is intentionally narrow: it only checks the rows that failed both
API judges and were then repaired by LLaVA. Rows approved by either API judge
are carried through unchanged. InternVL acts as a local VLM verifier over the
repaired rows and rejects repairs that still appear visually unsupported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.io_utils import ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paper.common import PAPER_PIPELINE_VERSION, normalize_space


DEFAULT_REPAIRED_INPUT = Path("output/fghd/released_pref_stage4_and_gate/repaired_preferences.jsonl")
DEFAULT_ACCEPTED_INPUT = Path("output/fghd/released_pref_stage3_and_gate/passed_by_either.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/fghd/released_pref_stage5_internvl_verify")
DEFAULT_VERIFY_OUT = DEFAULT_OUTPUT_DIR / "verification_records.jsonl"
DEFAULT_APPROVED_OUT = DEFAULT_OUTPUT_DIR / "approved_repaired_preferences.jsonl"
DEFAULT_FAILED_OUT = DEFAULT_OUTPUT_DIR / "failed_repaired_preferences.jsonl"
DEFAULT_FINAL_OUT = DEFAULT_OUTPUT_DIR / "final_verified_preference_pairs.jsonl"
DEFAULT_STATS_OUT = DEFAULT_OUTPUT_DIR / "stats.json"
DEFAULT_MODEL_PATH = Path("/root/models/InternVL-Chat-V1-2-Plus")
DEFAULT_IMAGE_ROOT = Path("hsa_dpo/data/images")

PROMPT_VERSION = "internvl_repair_verify_v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify LLaVA-repaired released preference rows with InternVL.",
    )
    parser.add_argument("--repaired-input", type=Path, default=DEFAULT_REPAIRED_INPUT)
    parser.add_argument("--accepted-input", type=Path, default=DEFAULT_ACCEPTED_INPUT)
    parser.add_argument("--verification-out", type=Path, default=DEFAULT_VERIFY_OUT)
    parser.add_argument("--approved-out", type=Path, default=DEFAULT_APPROVED_OUT)
    parser.add_argument("--failed-out", type=Path, default=DEFAULT_FAILED_OUT)
    parser.add_argument("--final-preferences-out", type=Path, default=DEFAULT_FINAL_OUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-load-in-8bit", action="store_true")
    return parser


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = ensure_parent_dir(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _image_candidates(row: dict[str, Any], image_root: Path) -> Iterable[Path]:
    image_value = str(row.get("image") or "").strip()
    if image_value:
        candidate = Path(image_value)
        if candidate.is_absolute():
            yield candidate
        else:
            yield candidate
            yield image_root / candidate
            yield image_root / candidate.name
    row_id = row.get("id")
    if row_id not in (None, ""):
        yield image_root / f"{row_id}.jpg"
        yield image_root / f"{row_id}.png"


def _resolve_image(row: dict[str, Any], image_root: Path) -> Path | None:
    for candidate in _image_candidates(row, image_root):
        if candidate.exists():
            return candidate
    return None


def _validation_feedback(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    votes = metadata.get("api_votes") or []
    if not votes:
        return "No API feedback was recorded."
    lines: list[str] = []
    for vote in votes:
        lines.append(
            f"- {vote.get('family', 'judge')}: approved={bool(vote.get('approved'))}; "
            f"reason={vote.get('reason', '')}"
        )
    return "\n".join(lines)


def _build_prompt(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    previous_chosen = metadata.get("previous_chosen", "")
    return (
        "You are a conservative vision-language verifier for hallucination repair.\n"
        "Use the image as the source of truth. Evaluate whether the final repaired answer is visually grounded, "
        "answers the question, and is preferable to the original rejected answer.\n"
        "Return exactly one JSON object and no other text.\n"
        "The JSON object must have these keys: approved, reason, remaining_hallucinations, severity_score.\n"
        "approved must be a JSON boolean.\n"
        "reason must be one concise sentence of 15 words or fewer.\n"
        "remaining_hallucinations must be a list of objects with keys type, evidence_text, reason, severity_score. "
        "Use an empty list if approved is true. If approved is false, include only the single most important remaining "
        "hallucination. Keep evidence_text to 12 words or fewer and reason to 15 words or fewer.\n"
        "severity_score must be an integer from 0 to 3 for the repaired answer.\n"
        "Reject if the repaired answer still contains unsupported object, attribute, relationship, count, action, "
        "or spatial claims. Do not reward fluency alone.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Original rejected answer:\n{row.get('rejected', '')}\n\n"
        f"Previous chosen answer before repair:\n{previous_chosen}\n\n"
        f"Final repaired answer to verify:\n{row.get('chosen', '')}\n\n"
        f"Original hallucination tags and severity:\n{row.get('rejected_tag_text', '')}\n\n"
        f"API judge feedback that triggered repair:\n{_validation_feedback(row)}\n\n"
        "Return only the JSON object now."
    )


def _extract_balanced_json(text: str) -> dict[str, Any] | None:
    starts = [index for index, char in enumerate(text or "") if char == "{"]
    fallback: dict[str, Any] | None = None
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
                    fragment = text[start : index + 1]
                    try:
                        payload = json.loads(fragment)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict):
                        if "approved" in payload:
                            return payload
                        if fallback is None:
                            fallback = payload
                    break
    return fallback


def _fallback_payload_from_raw(text: str) -> dict[str, Any] | None:
    compact = " ".join((text or "").split())
    if not compact:
        return None
    decision_match = re.search(
        r'"?approved"?\s*:\s*(true|false|True|False|yes|no|approved|rejected|pass|fail)',
        compact,
        flags=re.IGNORECASE,
    )
    if not decision_match:
        return None
    approved = _coerce_bool(decision_match.group(1))
    reason = ""
    reason_match = re.search(r'"?reason"?\s*:\s*"([^"]+)"', compact, flags=re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    if not reason:
        reason = _short_text(compact)
    severity = 0 if approved else 3
    severity_match = re.search(r'"?severity_score"?\s*:\s*([0-3])', compact, flags=re.IGNORECASE)
    if severity_match:
        severity = _coerce_severity(severity_match.group(1))
    return {
        "approved": approved,
        "reason": reason or "fallback parsed truncated verifier output",
        "remaining_hallucinations": [],
        "severity_score": severity,
    }


def _parse_json_response(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    candidate = re.sub(r":\s*True\b", ": true", candidate)
    candidate = re.sub(r":\s*False\b", ": false", candidate)
    candidate = re.sub(r":\s*None\b", ": null", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = _extract_balanced_json(candidate)
    if not isinstance(payload, dict):
        payload = _fallback_payload_from_raw(candidate)
    if not isinstance(payload, dict):
        raise ValueError(f"could not parse InternVL JSON response: {candidate[:180]!r}")
    return payload


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "approved", "approve", "pass", "passed"}:
            return True
        if lowered in {"false", "no", "rejected", "reject", "fail", "failed"}:
            return False
    raise ValueError(f"cannot parse approved boolean from {value!r}")


def _short_text(text: str, limit_words: int = 24) -> str:
    words = (text or "").strip().split()
    return " ".join(words[:limit_words])


def _infer_approved(payload: dict[str, Any], raw_output: str) -> bool:
    if "approved" in payload:
        return _coerce_bool(payload.get("approved"))
    remaining = _normalise_remaining(payload.get("remaining_hallucinations"))
    severity = payload.get("severity_score")
    if remaining:
        return False
    if severity is not None:
        return _coerce_severity(severity) == 0
    compact = " ".join((raw_output or "").split()).lower()
    if re.search(r"\b(reject|rejected|not approved|false|fail|failed|unsupported|hallucination)\b", compact):
        return False
    if re.search(r"\b(approve|approved|true|pass|passed|grounded)\b", compact):
        return True
    raise ValueError("could not infer approved decision")


def _coerce_severity(value: Any) -> int:
    try:
        severity = int(float(value))
    except (TypeError, ValueError):
        severity = 3
    return min(3, max(0, severity))


def _normalise_remaining(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalised: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalised.append(
            {
                "type": str(item.get("type") or "other"),
                "evidence_text": str(item.get("evidence_text") or ""),
                "reason": str(item.get("reason") or ""),
                "severity_score": _coerce_severity(item.get("severity_score", 3)),
            }
        )
    return normalised


def _verification_record(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    raw_output: str,
    image_path: Path | None,
) -> dict[str, Any]:
    approved = _infer_approved(payload, raw_output)
    remaining = _normalise_remaining(payload.get("remaining_hallucinations"))
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = _short_text(raw_output) or "InternVL verifier did not provide a reason."
    return {
        "id": row.get("id"),
        "approved": approved,
        "reason": reason,
        "remaining_hallucinations": remaining,
        "severity_score": _coerce_severity(payload.get("severity_score", 0 if approved else 3)),
        "raw_output": raw_output,
        "image": str(image_path) if image_path else row.get("image"),
        "question": row.get("question", ""),
        "rejected": row.get("rejected", ""),
        "repaired_chosen": row.get("chosen", ""),
        "metadata": {
            "source_stage": "released_pref_stage5_internvl_verify",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "verifier_model": "InternVL-Chat-V1-2-Plus",
        },
    }


def _parse_failure_record(row: dict[str, Any], error: Exception, raw_output: str = "") -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "approved": False,
        "reason": f"InternVL verification parse/load failure: {error}",
        "remaining_hallucinations": [],
        "severity_score": 3,
        "raw_output": raw_output,
        "image": row.get("image"),
        "question": row.get("question", ""),
        "rejected": row.get("rejected", ""),
        "repaired_chosen": row.get("chosen", ""),
        "metadata": {
            "source_stage": "released_pref_stage5_internvl_verify",
            "paper_pipeline_version": PAPER_PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "verifier_model": "InternVL-Chat-V1-2-Plus",
            "error": str(error),
        },
    }


def _with_verification_metadata(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata") or {})
    compact_record = {k: v for k, v in record.items() if k != "raw_output"}
    metadata.update(
        {
            "source_stage": "released_pref_stage5_internvl_verify",
            "internvl_verification": compact_record,
        }
    )
    out["metadata"] = metadata
    out["validation_approved"] = bool(record.get("approved"))
    return out


class InternVLVerifier:
    name = "internvl_chat_v1_2_plus"

    def __init__(
        self,
        *,
        model_path: Path,
        max_new_tokens: int,
        temperature: float,
        load_in_8bit: bool,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self._bundle: tuple[Any, Any, Any, Any] | None = None

    def _load(self) -> None:
        if self._bundle is not None:
            return
        import types

        import torch
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": False,
        }
        if self.load_in_8bit:
            kwargs.update(
                {
                    "load_in_8bit": True,
                    "device_map": "auto",
                }
            )
        else:
            kwargs.update({"device_map": "auto"})
        model = AutoModel.from_pretrained(str(self.model_path), **kwargs).eval()
        self._patch_generate_for_current_transformers(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
        )
        image_processor = CLIPImageProcessor.from_pretrained(str(self.model_path))
        device = next(
            (parameter.device for parameter in model.parameters() if parameter.device.type == "cuda"),
            next(model.parameters()).device,
        )
        self._bundle = (model, tokenizer, image_processor, device)

    @staticmethod
    def _patch_generate_for_current_transformers(model: Any) -> None:
        """Patch InternVL V1.2 generate for newer Transformers.

        The model's remote code passes ``return_dict`` directly into
        ``language_model.generate``. Newer Transformers also controls this
        internally, which raises "got multiple values for keyword argument
        'return_dict'". This instance-level patch keeps the original logic but
        omits the incompatible argument unless explicitly needed.
        """

        import types

        import torch

        def patched_generate(
            self: Any,
            pixel_values: Any = None,
            input_ids: Any = None,
            attention_mask: Any = None,
            visual_features: Any = None,
            generation_config: Any = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **generate_kwargs: Any,
        ) -> Any:
            assert self.img_context_token_id is not None
            if pixel_values is not None:
                if visual_features is not None:
                    vit_embeds = visual_features
                else:
                    vit_embeds = self.extract_feature(pixel_values)
                input_embeds = self.language_model.get_input_embeddings()(input_ids)
                batch_size, seq_len, hidden = input_embeds.shape
                input_embeds = input_embeds.reshape(batch_size * seq_len, hidden)
                flat_input_ids = input_ids.reshape(batch_size * seq_len)
                selected = flat_input_ids == self.img_context_token_id
                assert selected.sum() != 0
                input_embeds[selected] = vit_embeds.reshape(-1, hidden).to(input_embeds.device)
                input_embeds = input_embeds.reshape(batch_size, seq_len, hidden)
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids)

            call_kwargs: dict[str, Any] = {
                "inputs_embeds": input_embeds,
                "attention_mask": attention_mask,
                "generation_config": generation_config,
                "use_cache": True,
                **generate_kwargs,
            }
            if output_hidden_states is not None:
                call_kwargs["output_hidden_states"] = output_hidden_states
            if return_dict is not None:
                call_kwargs["return_dict_in_generate"] = return_dict
            return self.language_model.generate(**call_kwargs)

        model.generate = types.MethodType(torch.no_grad()(patched_generate), model)

    def verify(self, row: dict[str, Any], *, image_path: Path) -> dict[str, Any]:
        import torch
        from PIL import Image

        self._load()
        model, tokenizer, image_processor, device = self._bundle  # type: ignore[misc]
        image = Image.open(image_path).convert("RGB").resize((448, 448))
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        generation_config = {
            "num_beams": 1,
            "do_sample": self.temperature > 0.0,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0.0:
            generation_config["temperature"] = self.temperature
        prompt = _build_prompt(row)
        with torch.inference_mode():
            raw_output = model.chat(tokenizer, pixel_values, prompt, generation_config)
        try:
            payload = _parse_json_response(raw_output)
            return _verification_record(row=row, payload=payload, raw_output=raw_output, image_path=image_path)
        except Exception as exc:
            raise VerificationParseError(str(exc), raw_output=raw_output) from exc


class VerificationParseError(ValueError):
    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def _processed_ids(path: Path) -> set[Any]:
    if not path.exists():
        return set()
    return {row.get("id") for row in read_jsonl(path)}


def _reset_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_stats(
    *,
    args: argparse.Namespace,
    accepted_rows: int,
    total_repaired_rows: int,
    processed_rows: int,
    approved_rows: int,
    failed_rows: int,
    missing_images: int,
    parse_or_runtime_failures: int,
) -> None:
    final_rows = _jsonl_count(args.final_preferences_out)
    payload = {
        "paper_pipeline_version": PAPER_PIPELINE_VERSION,
        "stage": "released_pref_stage5_internvl_verify",
        "prompt_version": PROMPT_VERSION,
        "verifier_model": "InternVL-Chat-V1-2-Plus",
        "model_path": str(args.model_path),
        "accepted_input": str(args.accepted_input),
        "repaired_input": str(args.repaired_input),
        "accepted_rows": accepted_rows,
        "total_repaired_rows": total_repaired_rows,
        "processed_repaired_rows": processed_rows,
        "approved_repaired_rows": approved_rows,
        "failed_repaired_rows": failed_rows,
        "missing_images": missing_images,
        "parse_or_runtime_failures": parse_or_runtime_failures,
        "final_preference_rows": final_rows,
        "verification_out": str(args.verification_out),
        "approved_out": str(args.approved_out),
        "failed_out": str(args.failed_out),
        "final_preferences_out": str(args.final_preferences_out),
    }
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.repaired_input.exists():
        print(f"Repaired preference input not found: {args.repaired_input}", file=sys.stderr)
        return 2
    if not args.accepted_input.exists():
        print(f"Accepted preference input not found: {args.accepted_input}", file=sys.stderr)
        return 2
    if not args.model_path.exists():
        print(f"InternVL model path not found: {args.model_path}", file=sys.stderr)
        return 2

    accepted = _load_jsonl(args.accepted_input)
    repaired_rows = _load_jsonl(args.repaired_input)
    if args.limit is not None:
        repaired_rows = repaired_rows[: args.limit]

    if not args.resume:
        _reset_outputs(
            [
                args.verification_out,
                args.approved_out,
                args.failed_out,
                args.final_preferences_out,
                args.stats_out,
            ]
        )
        write_jsonl(ensure_parent_dir(args.final_preferences_out), accepted)
        processed: set[Any] = set()
    else:
        processed = _processed_ids(args.verification_out)
        if not args.final_preferences_out.exists():
            write_jsonl(ensure_parent_dir(args.final_preferences_out), accepted)

    verifier = InternVLVerifier(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_8bit=not args.no_load_in_8bit,
    )

    processed_rows = 0
    approved_rows = _jsonl_count(args.approved_out)
    failed_rows = _jsonl_count(args.failed_out)
    missing_images = 0
    parse_or_runtime_failures = 0

    total = len(repaired_rows)
    for row in maybe_tqdm(repaired_rows, desc="Released pref Stage 5 InternVL verify", total=total):
        row_id = row.get("id")
        if row_id in processed:
            continue
        image_path = _resolve_image(row, args.image_root)
        if image_path is None:
            record = _parse_failure_record(row, FileNotFoundError("image file not found"))
            missing_images += 1
        else:
            try:
                record = verifier.verify(row, image_path=image_path)
            except Exception as exc:
                if args.strict:
                    raise
                raw_output = getattr(exc, "raw_output", "")
                record = _parse_failure_record(row, exc, raw_output=raw_output)
                parse_or_runtime_failures += 1

        _append_jsonl(args.verification_out, record)
        verified_row = _with_verification_metadata(row, record)
        if bool(record.get("approved")):
            if normalize_space(verified_row.get("chosen", "")).lower() != normalize_space(verified_row.get("rejected", "")).lower():
                _append_jsonl(args.approved_out, verified_row)
                _append_jsonl(args.final_preferences_out, verified_row)
                approved_rows += 1
            else:
                verified_row["metadata"]["internvl_verification"]["approved"] = False
                verified_row["metadata"]["internvl_verification"]["reason"] = "Approved repair matched rejected answer."
                _append_jsonl(args.failed_out, verified_row)
                failed_rows += 1
        else:
            _append_jsonl(args.failed_out, verified_row)
            failed_rows += 1
        processed_rows += 1
        processed.add(row_id)
        _write_stats(
            args=args,
            accepted_rows=len(accepted),
            total_repaired_rows=total,
            processed_rows=processed_rows,
            approved_rows=approved_rows,
            failed_rows=failed_rows,
            missing_images=missing_images,
            parse_or_runtime_failures=parse_or_runtime_failures,
        )

    _write_stats(
        args=args,
        accepted_rows=len(accepted),
        total_repaired_rows=total,
        processed_rows=processed_rows,
        approved_rows=approved_rows,
        failed_rows=failed_rows,
        missing_images=missing_images,
        parse_or_runtime_failures=parse_or_runtime_failures,
    )
    print(
        f"Released pref Stage 5 InternVL verified {processed_rows} repaired row(s): "
        f"{approved_rows} approved, {failed_rows} failed."
    )
    print(f"Final verified preferences -> {args.final_preferences_out}")
    print(f"Verification stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
