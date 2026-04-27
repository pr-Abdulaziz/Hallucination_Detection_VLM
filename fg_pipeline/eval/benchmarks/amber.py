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


def _load_amber_runtime():
    try:
        import nltk
        import spacy
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise RuntimeError(
            "AMBER evaluation requires nltk and spacy. Install them before running Stage 5 evaluation."
        ) from exc

    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError as exc:
        raise RuntimeError(
            "AMBER evaluation requires the spaCy model en_core_web_lg. "
            "Install it with: python -m spacy download en_core_web_lg"
        ) from exc
    return nltk, WordNetLemmatizer(), nlp


def _extract_nouns(text: str, nltk_module: Any, lemmatizer: Any) -> list[str]:
    try:
        tokens = nltk_module.word_tokenize(text)
        tagged = nltk_module.pos_tag(tokens)
    except LookupError as exc:
        raise RuntimeError(
            "AMBER evaluation requires NLTK data: punkt, punkt_tab, averaged_perceptron_tagger, "
            "averaged_perceptron_tagger_eng, wordnet, and omw-1.4."
        ) from exc
    return [lemmatizer.lemmatize(word.lower()) for word, pos in tagged if pos.startswith("NN")]


def _synonym_match(word1: str, word2: str, nlp: Any, threshold: float = 0.8) -> bool:
    return bool(nlp(word1).similarity(nlp(word2)) > threshold)


def _score_amber_generative(
    answers: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> dict[str, float | None]:
    nltk_module, lemmatizer, nlp = _load_amber_runtime()

    relation_path = dataset_root / "relation.json"
    safe_words_path = dataset_root / "safe_words.txt"
    if not relation_path.exists():
        raise FileNotFoundError(f"Missing AMBER relation file: {relation_path}")
    if not safe_words_path.exists():
        raise FileNotFoundError(f"Missing AMBER safe words file: {safe_words_path}")

    association: dict[str, list[str]] = json.loads(relation_path.read_text(encoding="utf-8"))
    hallucination_words = set(association.keys())
    for related_words in association.values():
        hallucination_words.update(related_words)
    global_safe_words = {
        line.strip()
        for line in safe_words_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    annotations_by_id = {str(row.get("id")): row for row in annotations}

    chair_score = 0
    chair_num = 0
    safe_cover_score = 0
    safe_cover_num = 0
    hallu_cover_score = 0
    hallu_cover_num = 0
    non_hallu_score = 0
    non_hallu_num = 0

    for answer in answers:
        annotation = annotations_by_id.get(str(answer.get("id")))
        if annotation is None:
            raise ValueError(f"Missing AMBER annotation for id={answer.get('id')!r}")
        if annotation.get("type") not in (None, "generative"):
            continue

        nouns = _extract_nouns(str(answer.get("text") or ""), nltk_module, lemmatizer)
        candidate_nouns = [noun for noun in nouns if noun in hallucination_words]

        safe_words: list[str] = []
        safe_list: list[int] = []
        truth_words = list(annotation.get("truth") or [])
        for idx, word in enumerate(truth_words):
            related = list(association.get(word, []))
            safe_words.extend(related)
            safe_list.extend([idx] * len(related))

        hallu_words: list[str] = []
        hallu_list: list[int] = []
        hallucinated_words = list(annotation.get("hallu") or [])
        for idx, word in enumerate(hallucinated_words):
            related = list(association.get(word, []))
            hallu_words.extend(related)
            hallu_list.extend([idx] * len(related))

        safe_words.extend(truth_words)
        safe_len = len(truth_words)
        safe_list.extend([0] * safe_len)
        safe_flag_list = [0] * len(candidate_nouns)

        hallu_words.extend(hallucinated_words)
        hallu_len = len(hallucinated_words)
        hallu_list.extend([0] * hallu_len)

        for idx, noun in enumerate(candidate_nouns):
            if noun in global_safe_words:
                continue

            if noun in safe_words:
                for j, safe_word in enumerate(safe_words):
                    if noun == safe_word:
                        target = safe_list[j] + len(safe_list) - safe_len if j < len(safe_list) - safe_len else j
                        if 0 <= target < len(safe_list):
                            safe_list[target] = 1
                        break
                continue

            if noun in hallu_words:
                for j, hallu_word in enumerate(hallu_words):
                    if noun == hallu_word:
                        target = hallu_list[j] + len(hallu_list) - hallu_len if j < len(hallu_list) - hallu_len else j
                        if 0 <= target < len(hallu_list):
                            hallu_list[target] = 1
                        break

            for j, hallu_word in enumerate(hallu_words):
                if _synonym_match(noun, hallu_word, nlp):
                    target = hallu_list[j] + len(hallu_list) - hallu_len if j < len(hallu_list) - hallu_len else j
                    if 0 <= target < len(hallu_list):
                        hallu_list[target] = 1
                    break

            safe_match = False
            for j, safe_word in enumerate(safe_words):
                if _synonym_match(noun, safe_word, nlp):
                    safe_match = True
                    target = safe_list[j] + len(safe_list) - safe_len if j < len(safe_list) - safe_len else j
                    if 0 <= target < len(safe_list):
                        safe_list[target] = 1
                    break
            if safe_match:
                continue

            safe_flag_list[idx] = 1

        chair_score += sum(safe_flag_list)
        chair_num += len(safe_flag_list)
        safe_cover_score += sum(safe_list[-safe_len:]) if safe_len else 0
        safe_cover_num += safe_len
        hallu_cover_score += sum(hallu_list[-hallu_len:]) if hallu_len else 0
        hallu_cover_num += hallu_len
        if sum(safe_flag_list) == 0:
            non_hallu_score += 1
        non_hallu_num += 1

    chair = safe_div(chair_score, chair_num)
    cover = safe_div(safe_cover_score, safe_cover_num)
    cog = safe_div(hallu_cover_score, hallu_cover_num)
    non_hallu = safe_div(non_hallu_score, non_hallu_num)
    return {
        "amber_chair": None if chair is None else round(chair * 100.0, 1),
        "amber_cover": None if cover is None else round(cover * 100.0, 1),
        "amber_hal": None if non_hallu is None else round(100.0 - non_hallu * 100.0, 1),
        "amber_cog": None if cog is None else round(cog * 100.0, 1),
    }


class AmberBenchmark:
    name = "amber"
    judge_required = False
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "amber")
        question_file = str(Path(dataset_root) / "query_generative.jsonl")
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
            split="generative",
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
            raise ValueError("AMBER requires a model spec")
        question_path = Path(spec.question_file or "")
        annotation_path = Path(spec.annotation_file or "")
        if not question_path.exists():
            raise FileNotFoundError(f"Missing AMBER question file: {question_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing AMBER annotation file: {annotation_path}")
        raw_questions = _read_jsonl(question_path)
        annotations = _read_jsonl(annotation_path)
        if limit is not None:
            raw_questions = raw_questions[:limit]
            keep_ids = {str(row.get("id")) for row in raw_questions}
            annotations = [row for row in annotations if str(row.get("id")) in keep_ids]

        questions = []
        for idx, row in enumerate(raw_questions):
            record_id = str(row.get("id", idx))
            image_name = row.get("image")
            questions.append(
                {
                    "id": record_id,
                    "question": row.get("question") or row.get("query") or row.get("text") or "",
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

        metric_payload = _score_amber_generative(
            answers,
            annotations,
            dataset_root=Path(spec.dataset_root or question_path.parent),
        )
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            metrics=metric_payload,
            comparable_to_paper=False,
            comparison_note="local AMBER generative evaluation; automatic and no external judge API",
        )
        metric_dir = mkdir(Path(run_root) / "models" / model.model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        dump_json(
            metric_dir / f"{self.name}.meta.json",
            {
                "annotation_file": spec.annotation_file,
                "question_file": spec.question_file,
                "num_examples": len(answers),
                "protocol": "AMBER generative automatic metric",
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
