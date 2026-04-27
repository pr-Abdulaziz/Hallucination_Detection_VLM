from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fg_pipeline.eval.model_loader import generate_answers_for_records
from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import (
    binary_classification_metrics,
    default_dataset_root,
    dump_json,
    mkdir,
    normalize_text,
    read_jsonl,
)


def _yes_no(text: str | None) -> str:
    normalized = normalize_text(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return normalized.split(" ")[0] if normalized else ""


def _percent(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


class PopeBenchmark:
    name = "pope_adv"
    judge_required = False
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "pope")
        question_file = str(Path(dataset_root) / "llava_pope_test.jsonl")
        annotation_file = str(Path(dataset_root) / "llava_pope_test.jsonl")
        image_root = image_root_override or str(Path(dataset_root) / "val2014")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=question_file,
            annotation_file=annotation_file,
            image_root=image_root,
            dataset_root=dataset_root,
            judge_required=False,
            split="adversarial",
        )

    def evaluate(
        self,
        model: ModelSpec | None,
        spec: BenchmarkSpec,
        *,
        run_root: str,
        limit: int | None = None,
        openai_judge_model: str | None = None,
    ) -> tuple[PredictionArtifact | None, MetricArtifact, JudgeArtifact | None]:
        if model is None:
            raise ValueError("POPE requires a model spec")
        question_path = Path(spec.question_file or "")
        if not question_path.exists():
            raise FileNotFoundError(f"Missing POPE question file: {question_path}")
        raw_questions = list(read_jsonl(question_path))
        if spec.split:
            raw_questions = [
                row
                for row in raw_questions
                if str(row.get("category") or row.get("split") or "").strip().lower()
                in {"", str(spec.split).strip().lower()}
            ]
        if limit is not None:
            raw_questions = raw_questions[:limit]
        questions = []
        labels: dict[str, str] = {}
        for idx, row in enumerate(raw_questions):
            record_id = str(row.get("question_id", row.get("id", idx)))
            image_name = row.get("image")
            image_path = Path(spec.image_root or "") / image_name if image_name else None
            questions.append(
                {
                    "id": record_id,
                    "question": row.get("text") or row.get("question") or "",
                    "image": str(image_path) if image_path else None,
                }
            )
            labels[record_id] = _yes_no(
                row.get("label") or row.get("answer") or row.get("gt_answer")
            )

        answers = generate_answers_for_records(model, questions)
        prediction_dir = mkdir(Path(run_root) / "models" / model.model_id / "predictions")
        prediction_path = prediction_dir / f"{self.name}.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        y_true = [1 if labels.get(str(row["id"])) == "yes" else 0 for row in answers]
        y_pred = [1 if _yes_no(row["text"]) == "yes" else 0 for row in answers]
        metrics = binary_classification_metrics(y_true, y_pred)
        metric_payload = {
            "f1": _percent(metrics["f1"]),
            "pope_adv_f1": _percent(metrics["f1"]),
            "precision": _percent(metrics["precision"]),
            "recall": _percent(metrics["recall"]),
            "accuracy": _percent(metrics["accuracy"]),
        }
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            metrics=metric_payload,
            comparable_to_paper=(spec.split == "adversarial" and model.temperature == 0.0 and model.num_beams == 1),
            comparison_note=None if spec.split == "adversarial" else "POPE split is not adversarial",
        )
        metric_dir = mkdir(Path(run_root) / "models" / model.model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        metadata = {
            "question_file": spec.question_file,
            "annotation_file": spec.annotation_file,
            "image_root": spec.image_root,
            "num_examples": len(answers),
        }
        dump_json(metric_dir / f"{self.name}.meta.json", metadata)
        prediction_artifact = PredictionArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            path=str(prediction_path),
            num_examples=len(answers),
            decode_config={
                "temperature": model.temperature,
                "num_beams": model.num_beams,
                "max_new_tokens": model.max_new_tokens,
            },
        )
        return prediction_artifact, metric_artifact, None
