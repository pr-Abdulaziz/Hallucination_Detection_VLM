from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fg_pipeline.eval.model_loader import generate_answers_for_records
from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import default_dataset_root, dump_json, mkdir, safe_div


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(item)
    return rows


def _load_nltk_runtime():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise RuntimeError("Object HalBench evaluation requires nltk.") from exc
    return nltk, WordNetLemmatizer()


class _CocoObjectExtractor:
    def __init__(self, synonyms_path: Path) -> None:
        if not synonyms_path.exists():
            raise FileNotFoundError(f"Missing Object HalBench synonym file: {synonyms_path}")
        synonym_rows = [
            [part.strip() for part in line.strip().split(", ") if part.strip()]
            for line in synonyms_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.mscoco_objects: set[str] = set()
        self.inverse_synonym_dict: dict[str, str] = {}
        for row in synonym_rows:
            if not row:
                continue
            canonical = row[0]
            self.mscoco_objects.update(row)
            for synonym in row:
                self.inverse_synonym_dict[synonym] = canonical

        coco_double_words = [word for word in self.inverse_synonym_dict if len(word.split()) >= 2]
        coco_double_words += ["home plate", "train track"]
        animal_words = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
        vehicle_words = ["jet", "train"]
        self.double_word_dict = {word: word for word in coco_double_words}
        for animal_word in animal_words:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"
        self._nltk, self._lemmatizer = _load_nltk_runtime()

    def caption_to_coco_objects(self, caption: str) -> list[str]:
        try:
            tokens = self._nltk.word_tokenize(caption.lower())
        except LookupError as exc:
            raise RuntimeError(
                "Object HalBench evaluation requires NLTK data: punkt, punkt_tab, wordnet, and omw-1.4."
            ) from exc
        words = [self._lemmatizer.lemmatize(token) for token in tokens]

        merged_words: list[str] = []
        idx = 0
        while idx < len(words):
            double_word = " ".join(words[idx : idx + 2])
            if double_word in self.double_word_dict:
                merged_words.append(self.double_word_dict[double_word])
                idx += 2
            else:
                merged_words.append(words[idx])
                idx += 1

        if "toilet" in merged_words and "seat" in merged_words:
            merged_words = [word for word in merged_words if word != "seat"]

        canonical_objects: list[str] = []
        for word in merged_words:
            if word not in self.mscoco_objects:
                continue
            canonical = self.inverse_synonym_dict[word]
            canonical_objects.append(canonical)
        return canonical_objects


class ObjectHalBenchBenchmark:
    name = "object_halbench"
    judge_required = False
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "object-halbench")
        question_file = str(Path(dataset_root) / "questions.jsonl")
        annotation_file = str(Path(dataset_root) / "annotations.jsonl")
        image_root = image_root_override or str(Path(dataset_root) / "images")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=question_file,
            annotation_file=annotation_file,
            image_root=image_root,
            dataset_root=dataset_root,
            judge_required=False,
            split="rule_based_chair",
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
            raise ValueError("Object HalBench requires a model spec")
        question_path = Path(spec.question_file or "")
        annotation_path = Path(spec.annotation_file or "")
        if not question_path.exists():
            raise FileNotFoundError(f"Missing Object HalBench question file: {question_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing Object HalBench annotation file: {annotation_path}")

        raw_questions = _read_jsonl(question_path)
        annotations = _read_jsonl(annotation_path)
        if limit is not None:
            raw_questions = raw_questions[:limit]
            keep_ids = {str(row.get("id")) for row in raw_questions}
            annotations = [row for row in annotations if str(row.get("id")) in keep_ids]
        annotations_by_id = {str(row.get("id")): row for row in annotations}

        questions = []
        for idx, row in enumerate(raw_questions):
            record_id = str(row.get("id", idx))
            image_name = row.get("image")
            questions.append(
                {
                    "id": record_id,
                    "question": row.get("question") or row.get("text") or "",
                    "image": str(Path(spec.image_root or "") / image_name) if image_name else None,
                }
            )
        answers = generate_answers_for_records(model, questions)
        prediction_dir = mkdir(Path(run_root) / "models" / model.model_id / "predictions")
        prediction_path = prediction_dir / f"{self.name}.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        extractor = _CocoObjectExtractor(Path(spec.dataset_root or question_path.parent) / "synonyms_refine.txt")
        response_hallucination_count = 0
        hallucinated_mentions = 0
        object_mentions = 0
        coco_sentence_count = 0
        for answer in answers:
            annotation = annotations_by_id.get(str(answer.get("id")))
            if annotation is None:
                raise ValueError(f"Missing Object HalBench annotation for id={answer.get('id')!r}")
            gt_objects = set(annotation.get("gt_objects") or [])
            predicted_objects = extractor.caption_to_coco_objects(str(answer.get("text") or ""))
            object_mentions += len(predicted_objects)
            if predicted_objects:
                coco_sentence_count += 1
            hallucinated = [obj for obj in predicted_objects if obj not in gt_objects]
            hallucinated_mentions += len(hallucinated)
            if hallucinated:
                response_hallucination_count += 1

        chairs = safe_div(response_hallucination_count, len(answers))
        chairi = safe_div(hallucinated_mentions, object_mentions)
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            metrics={
                "chairs": None if chairs is None else chairs * 100.0,
                "chairi": None if chairi is None else chairi * 100.0,
            },
            comparable_to_paper=False,
            comparison_note="rule-based Object HalBench CHAIR; no GPT/OpenAI object extraction",
        )
        metric_dir = mkdir(Path(run_root) / "models" / model.model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        dump_json(
            metric_dir / f"{self.name}.meta.json",
            {
                "annotation_file": spec.annotation_file,
                "question_file": spec.question_file,
                "num_examples": len(answers),
                "object_mentions": object_mentions,
                "coco_sentence_count": coco_sentence_count,
                "protocol": "rule-based CHAIR without external judge API",
            },
        )
        return (
            PredictionArtifact(
                benchmark=self.name,
                model_id=model.model_id,
                path=str(prediction_path),
                num_examples=len(answers),
                decode_config={
                    "temperature": model.temperature,
                    "num_beams": model.num_beams,
                    "max_new_tokens": model.max_new_tokens,
                },
            ),
            metric_artifact,
            None,
        )
