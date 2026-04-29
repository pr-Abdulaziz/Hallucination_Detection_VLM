from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.paper.prompts import (
    build_api_critic_feedback_prompt,
    build_ddg_annotation_prompt,
    build_feedback_revision_prompt,
    build_paper_rewrite_prompt,
    build_vcr_annotation_prompt,
    severity_rubric_text,
)
from fg_pipeline.paper.run_stage1_faif import main as paper_stage1_main
from fg_pipeline.paper.run_stage2_detector_dataset import main as paper_stage2_main
from fg_pipeline.paper.run_stage4_rewrite import main as paper_stage4_main


def _released_row(*, row_id: int, gpt_value: str, text: str = "A test response.") -> dict:
    return {
        "id": row_id,
        "image": "vg/images/test.jpg",
        "conversations": [
            {"from": "human", "value": f"<image>\nDescription to Assess:\n{text}"},
            {"from": "gpt", "value": gpt_value},
        ],
    }


def _hallucinated_payload() -> str:
    return (
        "Tags:\n"
        "<object>\n"
        "1 . The gold cross is not visible.\n"
        "Scores:\n"
        "<object>\n"
        "1 . gold cross: Major (3 points): It is not present in the image."
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PaperPromptTests(unittest.TestCase):
    def test_appendix_prompts_render_required_fields(self) -> None:
        ddg = build_ddg_annotation_prompt(regions='[{"phrase": "dog"}]', description="A cat.")
        vcr = build_vcr_annotation_prompt(reasoning="Question: What is shown? Answer: A cat.")
        rewrite = build_paper_rewrite_prompt(
            {
                "question": "What is in the image?",
                "original_response": "A gold cross is on the tower.",
                "detected_critiques": [
                    {
                        "hallucination_type": "object",
                        "severity_label": "major",
                        "severity_score": 3,
                        "evidence_text": "gold cross",
                        "rationale": "The gold cross is not visible.",
                    }
                ],
            }
        )

        self.assertIn("Image Regions Information", ddg)
        self.assertIn("Description:", ddg)
        self.assertIn("Complex reasoning", vcr)
        self.assertIn("Detected hallucination tags and reasons", rewrite)
        self.assertIn("The tags may not always be correct", rewrite)
        self.assertIn("Major (3 points)", severity_rubric_text())

    def test_api_critic_and_revision_prompts_separate_feedback_from_final_rewrite(self) -> None:
        record = {
            "question": "What is in the image?",
            "original_response": "A gold cross is on the tower.",
            "detected_critiques": [
                {
                    "hallucination_type": "object",
                    "severity_label": "major",
                    "severity_score": 3,
                    "evidence_text": "gold cross",
                    "rationale": "The gold cross is not visible.",
                }
            ],
            "initial_rewrite_response": "A tower is visible.",
            "api_feedback": "[gemini]\nNO CRITICAL ISSUES.",
        }
        feedback_prompt = build_api_critic_feedback_prompt(record, "A tower is visible.")
        revision_prompt = build_feedback_revision_prompt(record)

        self.assertIn("feedback only", feedback_prompt)
        self.assertIn("Do not write the final answer", feedback_prompt)
        self.assertIn("Initial rewritten response", feedback_prompt)
        self.assertIn("Critic feedback", revision_prompt)
        self.assertIn("Output only the final revised answer", revision_prompt)


class PaperStage1And2Tests(unittest.TestCase):
    def test_stage1_and_stage2_outputs_counts_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "released.jsonl"
            stage1_out = tmp_dir / "d_faif.jsonl"
            stage1_stats = tmp_dir / "stage1_stats.json"
            stage2_out = tmp_dir / "detector_train.json"
            stage2_stats = tmp_dir / "stage2_stats.json"
            write_jsonl(
                input_path,
                [
                    _released_row(row_id=1, gpt_value=_hallucinated_payload()),
                    _released_row(row_id=2, gpt_value="NO HALLUCINATION"),
                ],
            )

            self.assertEqual(
                paper_stage1_main(
                    [
                        "--input", str(input_path),
                        "--output", str(stage1_out),
                        "--stats-out", str(stage1_stats),
                    ]
                ),
                0,
            )
            rows = _read_jsonl(stage1_out)
            stats = json.loads(stage1_stats.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats["hallucinated_rows"], 1)
            self.assertEqual(stats["non_hallucinated_rows"], 1)
            self.assertEqual(rows[0]["metadata"]["annotation_source"], "released_annotations")

            self.assertEqual(
                paper_stage2_main(
                    [
                        "--input", str(stage1_out),
                        "--output", str(stage2_out),
                        "--stats-out", str(stage2_stats),
                        "--seed", "7",
                    ]
                ),
                0,
            )
            examples = json.loads(stage2_out.read_text(encoding="utf-8"))
            split_stats = json.loads(stage2_stats.read_text(encoding="utf-8"))
            targets = [example["conversations"][1]["value"] for example in examples]
            self.assertEqual(len(examples), 2)
            self.assertIn("NO HALLUCINATION", targets)
            self.assertTrue(any("Tags:" in target and "Scores:" in target for target in targets))
            self.assertEqual(split_stats["selected_hallucinated_rows"], 1)
            self.assertEqual(split_stats["selected_non_hallucinated_rows"], 1)


class PaperStage4Tests(unittest.TestCase):
    def test_stage4_accepts_stage1_d_faif_records_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image = tmp_dir / "image.jpg"
            image.write_bytes(b"placeholder")
            stage1_records = tmp_dir / "d_faif.jsonl"
            rewrites = tmp_dir / "rewrite_records.jsonl"
            prefs = tmp_dir / "preference_pairs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(
                stage1_records,
                [
                    {
                        "id": 1,
                        "image": str(image),
                        "question": "What is shown?",
                        "response_text": "A gold cross is on the tower.",
                        "is_hallucinated": True,
                        "critiques": [
                            {
                                "hallucination_type": "object",
                                "severity_label": "major",
                                "severity_score": 3,
                                "evidence_text": "gold cross",
                                "rationale": "The gold cross is not visible.",
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "image": str(image),
                        "question": "What is shown?",
                        "response_text": "A tower is visible.",
                        "is_hallucinated": False,
                        "critiques": [],
                    },
                ],
            )

            rc = paper_stage4_main(
                [
                    "--input", str(stage1_records),
                    "--output", str(rewrites),
                    "--preferences-out", str(prefs),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                    "--image-root", str(tmp_dir),
                ]
            )
            self.assertEqual(rc, 0)
            pref_rows = _read_jsonl(prefs)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(len(pref_rows), 1)
            self.assertEqual(pref_rows[0]["rejected"], "A gold cross is on the tower.")
            self.assertEqual(pref_rows[0]["rejected_score"], 3.0)
            self.assertEqual(stats["input_rows"], 2)
            self.assertEqual(stats["predicted_non_hallucinated_skipped"], 1)

    def test_stage4_builds_preference_and_filters_identical_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image = tmp_dir / "image.jpg"
            image.write_bytes(b"placeholder")
            detections = tmp_dir / "detections.jsonl"
            rewrites = tmp_dir / "rewrite_records.jsonl"
            prefs = tmp_dir / "preference_pairs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(
                detections,
                [
                    {
                        "id": 1,
                        "image": str(image),
                        "question": "What is shown?",
                        "original_response": "A gold cross is on the tower.",
                        "is_hallucinated_pred": True,
                        "detected_critiques": [
                            {
                                "hallucination_type": "object",
                                "severity_label": "major",
                                "severity_score": 3,
                                "evidence_text": "gold cross",
                                "rationale": "The gold cross is not visible.",
                            }
                        ],
                        "response_severity_score": 3.0,
                    }
                ],
            )

            rc = paper_stage4_main(
                [
                    "--input", str(detections),
                    "--output", str(rewrites),
                    "--preferences-out", str(prefs),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                    "--image-root", str(tmp_dir),
                ]
            )
            self.assertEqual(rc, 0)
            pref_rows = _read_jsonl(prefs)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(len(pref_rows), 1)
            self.assertEqual(pref_rows[0]["rejected"], "A gold cross is on the tower.")
            self.assertEqual(pref_rows[0]["rejected_score"], 3.0)
            self.assertEqual(stats["preference_pairs_emitted"], 1)


if __name__ == "__main__":
    unittest.main()
