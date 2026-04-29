#!/usr/bin/env python3
"""Compare base LLaVA against the trained HSA-DPO adapter on the same image.

The script has two useful modes:

* ``live``: loads the base model and the HSA-DPO LoRA adapter, then runs both
  models on one image/prompt. This requires a GPU and local model weights.
* ``cached``: reads already generated evaluation predictions and displays a
  base-vs-trained example without loading the models. This is useful for slides
  and quick demos.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "models/llava-v1.5-7b"
DEFAULT_ADAPTER = "output/fghd/exp_2shot_verified_margin_hsa_b32_e1"
DEFAULT_EVAL_DIR = "output/eval/2shot_verified_margin_eval_full_20260428_172728"
DEFAULT_IMAGE = "hsa_dpo/models/llava-v1_5/llava/serve/examples/waterview.jpg"
DEFAULT_PROMPT = (
    "Describe this image in detail. Mention only visible objects and avoid "
    "unsupported assumptions."
)


@dataclass
class ComparisonResult:
    image: str
    prompt: str
    base_output: str
    hsa_dpo_output: str
    source: str
    example_id: str | None = None
    benchmark: str | None = None


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_image_path(path: str | Path) -> str:
    """Map remote evaluation paths back to this repo when possible."""

    raw = str(path)
    direct = Path(raw)
    if direct.exists():
        return str(direct)

    markers = (
        "/workspace/Hallucination_Detection_VLM/",
        "\\workspace\\Hallucination_Detection_VLM\\",
    )
    for marker in markers:
        if marker in raw:
            suffix = raw.split(marker, 1)[1].replace("\\", "/")
            candidate = REPO_ROOT / suffix
            if candidate.exists():
                return str(candidate)

    relative = _repo_path(raw)
    if relative.exists():
        return str(relative)
    return raw


def _prediction_path(eval_dir: str | Path, model_id: str, benchmark: str) -> Path:
    return _repo_path(eval_dir) / "models" / model_id / "predictions" / f"{benchmark}.jsonl"


def load_cached_comparison(
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    *,
    benchmark: str = "object_halbench",
    base_model_id: str = "llava15_base_7b",
    hsa_model_id: str = "two_shot_verified_margin_hsa_b32_e1",
    example_id: str | None = None,
) -> ComparisonResult:
    """Load a before/after example from saved evaluation predictions."""

    base_rows = _read_jsonl(_prediction_path(eval_dir, base_model_id, benchmark))
    hsa_rows = _read_jsonl(_prediction_path(eval_dir, hsa_model_id, benchmark))
    hsa_by_id = {str(row.get("id")): row for row in hsa_rows}

    selected_base: dict[str, Any] | None = None
    selected_hsa: dict[str, Any] | None = None

    if example_id is not None:
        selected_base = next((row for row in base_rows if str(row.get("id")) == str(example_id)), None)
        selected_hsa = hsa_by_id.get(str(example_id))
    else:
        for row in base_rows:
            row_id = str(row.get("id"))
            other = hsa_by_id.get(row_id)
            if other and (row.get("text") or "").strip() != (other.get("text") or "").strip():
                selected_base = row
                selected_hsa = other
                break

    if selected_base is None or selected_hsa is None:
        raise ValueError(
            f"Could not find a matched cached example for benchmark={benchmark!r}, "
            f"example_id={example_id!r}."
        )

    return ComparisonResult(
        image=_normalize_image_path(selected_base.get("image") or ""),
        prompt=str(selected_base.get("question") or selected_base.get("prompt") or ""),
        base_output=str(selected_base.get("text") or ""),
        hsa_dpo_output=str(selected_hsa.get("text") or ""),
        source=f"cached evaluation: {_repo_path(eval_dir)}",
        example_id=str(selected_base.get("id")),
        benchmark=benchmark,
    )


def run_live_comparison(
    *,
    image: str | Path = DEFAULT_IMAGE,
    prompt: str = DEFAULT_PROMPT,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    adapter_path: str | Path = DEFAULT_ADAPTER,
    conv_mode: str = "vicuna_v1",
    temperature: float = 0.0,
    num_beams: int = 1,
    max_new_tokens: int = 256,
) -> ComparisonResult:
    """Run base and trained models on the same image/prompt."""

    from fg_pipeline.eval.model_loader import generate_answers_for_records
    from fg_pipeline.eval.schemas import ModelSpec

    image_path = _normalize_image_path(image)
    record = {"id": "demo", "image": image_path, "question": prompt}
    base_spec = ModelSpec(
        model_id="llava15_base_7b",
        model_path=str(base_model),
        model_base=None,
        kind="base",
        conv_mode=conv_mode,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    hsa_spec = ModelSpec(
        model_id="two_shot_verified_margin_hsa_b32_e1",
        model_path=str(adapter_path),
        model_base=str(base_model),
        kind="lora",
        conv_mode=conv_mode,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

    base_output = generate_answers_for_records(base_spec, [record])[0]["text"]
    hsa_output = generate_answers_for_records(hsa_spec, [record])[0]["text"]
    return ComparisonResult(
        image=image_path,
        prompt=prompt,
        base_output=base_output,
        hsa_dpo_output=hsa_output,
        source="live inference",
        example_id="demo",
        benchmark=None,
    )


def load_metric_summary(summary_csv: str | Path = f"{DEFAULT_EVAL_DIR}/comparison/summary.csv") -> list[dict[str, str]]:
    path = _repo_path(summary_csv)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_comparison_markdown(result: ComparisonResult) -> str:
    base_output = result.base_output.replace("|", "&#124;").replace("\n", "<br>")
    hsa_output = result.hsa_dpo_output.replace("|", "&#124;").replace("\n", "<br>")
    header = f"### Base LLaVA vs HSA-DPO\n\nSource: `{result.source}`"
    if result.benchmark:
        header += f"\n\nBenchmark: `{result.benchmark}`"
    if result.example_id:
        header += f"\n\nExample id: `{result.example_id}`"
    body = (
        f"\n\n**Prompt**\n\n{result.prompt}\n\n"
        "| Model | Response |\n"
        "| --- | --- |\n"
        f"| Base LLaVA | {base_output} |\n"
        f"| HSA-DPO after training | {hsa_output} |\n"
    )
    return header + body


def print_metric_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    print("\nEvaluation metric summary:")
    for row in rows:
        metric = f"{row['benchmark']} / {row['metric']}"
        print(
            f"- {metric}: base={row['baseline_value'] or 'NA'}, "
            f"HSA-DPO={row['our_value'] or 'NA'}, delta={row['delta_vs_baseline'] or 'NA'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base LLaVA and trained HSA-DPO inference.")
    parser.add_argument("--mode", choices=("live", "cached"), default="cached")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image path for live inference.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for live inference.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base LLaVA model path.")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER, help="Trained HSA-DPO LoRA adapter path.")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, help="Evaluation run directory for cached mode.")
    parser.add_argument("--benchmark", default="object_halbench", help="Cached benchmark to read.")
    parser.add_argument("--example-id", default=None, help="Specific cached example id.")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--summary-csv", default=f"{DEFAULT_EVAL_DIR}/comparison/summary.csv")
    parser.add_argument("--output-markdown", default=None, help="Optional path to save the comparison as Markdown.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "live":
        result = run_live_comparison(
            image=args.image,
            prompt=args.prompt,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            conv_mode=args.conv_mode,
            temperature=args.temperature,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = load_cached_comparison(
            args.eval_dir,
            benchmark=args.benchmark,
            example_id=args.example_id,
        )

    markdown = format_comparison_markdown(result)
    print(markdown)
    print_metric_summary(load_metric_summary(args.summary_csv))

    if args.output_markdown:
        output_path = _repo_path(args.output_markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nSaved comparison markdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
