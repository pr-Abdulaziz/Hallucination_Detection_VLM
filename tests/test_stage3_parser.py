from __future__ import annotations

import json
import unittest
from pathlib import Path

from fg_pipeline.confidence.parser import parse_detection_response
from fg_pipeline.confidence.run_detect import build_detection_record
from fg_pipeline.confidence.scoring import BootstrapScorer, get_scorer

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fg_pipeline" / "data" / "smoke_detection.jsonl"


def _load_fixture() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


class ParseDetectionResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _load_fixture()

    def _response(self, row_id: int) -> str:
        return self.rows[row_id]["conversations"][1]["value"]

    def test_no_hallucination_returns_empty(self) -> None:
        self.assertEqual(parse_detection_response(self._response(0)), [])

    def test_empty_or_none_returns_empty(self) -> None:
        self.assertEqual(parse_detection_response(""), [])
        self.assertEqual(parse_detection_response(None), [])

    def test_single_object_major(self) -> None:
        signals = parse_detection_response(self._response(8500))
        self.assertEqual(len(signals), 1)
        s = signals[0]
        self.assertEqual(s["hallucination_type"], "object")
        self.assertEqual(s["severity"], 3)
        self.assertEqual(s["severity_label"], "major")
        self.assertEqual(s["span"], "Sidewalk")
        self.assertTrue(s["rationale"])
        self.assertIn("dance routine", s["tag_text"])

    def test_multi_signal_object_plus_attribute(self) -> None:
        signals = parse_detection_response(self._response(8501))
        self.assertEqual(len(signals), 2)
        self.assertEqual([s["hallucination_type"] for s in signals], ["object", "attribute"])
        self.assertEqual([s["severity"] for s in signals], [3, 2])
        self.assertEqual([s["sentence_index"] for s in signals], [0, 1])
        for s in signals:
            self.assertTrue(s["rationale"])

    def test_multi_signal_object_plus_relationship_unnumbered(self) -> None:
        signals = parse_detection_response(self._response(8517))
        self.assertEqual(len(signals), 2)
        self.assertEqual([s["hallucination_type"] for s in signals], ["object", "relationship"])
        self.assertEqual([s["severity"] for s in signals], [2, 2])

    def test_rationale_nonempty_for_all_hallucinated_signals(self) -> None:
        for row_id in (8500, 8501, 8517):
            for s in parse_detection_response(self._response(row_id)):
                self.assertTrue(s["rationale"], f"empty rationale for row {row_id}")


class BuildDetectionRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _load_fixture()
        cls.scorer = BootstrapScorer()

    def test_candidate_response_is_yhat(self) -> None:
        """candidate_response must hold yhat (the description being assessed)."""
        for row_id, row in self.rows.items():
            raw_human = row["conversations"][0]["value"]
            expected = raw_human.replace("<image>\nDescription to Assess:\n", "", 1)
            record = build_detection_record(row, self.scorer).to_dict()
            self.assertEqual(record["candidate_response"], expected, f"row {row_id}")
            self.assertFalse(record["candidate_response"].startswith("<image>"))
            self.assertFalse(record["candidate_response"].startswith("Description to Assess:"))

    def test_raw_detection_is_teacher_annotation(self) -> None:
        """raw_detection must hold the GPT-4 structured annotation."""
        for row_id, row in self.rows.items():
            expected = row["conversations"][1]["value"]
            record = build_detection_record(row, self.scorer).to_dict()
            self.assertEqual(record["raw_detection"], expected, f"row {row_id}")

    def test_candidate_response_and_raw_detection_differ_for_hallucinated_rows(self) -> None:
        for row_id in (8500, 8501, 8517):
            record = build_detection_record(self.rows[row_id], self.scorer).to_dict()
            self.assertNotEqual(
                record["candidate_response"], record["raw_detection"], f"row {row_id}"
            )

    def test_prompt_is_canonical_instruction(self) -> None:
        record = build_detection_record(self.rows[0], self.scorer).to_dict()
        self.assertEqual(record["prompt"], "Describe this image in detail.")
        self.assertEqual(
            record["metadata"]["stage1_instruction_source"], "canonical_describe_prompt"
        )

    def test_no_hallucination_has_no_signals(self) -> None:
        record = build_detection_record(self.rows[0], self.scorer).to_dict()
        self.assertEqual(record["signals"], [])

    def test_multi_signal_row_preserves_all_signals(self) -> None:
        record = build_detection_record(self.rows[8501], self.scorer).to_dict()
        self.assertEqual(len(record["signals"]), 2)

    def test_bootstrap_metadata_tags_placeholder(self) -> None:
        record = build_detection_record(self.rows[8500], self.scorer).to_dict()
        signal = record["signals"][0]
        self.assertEqual(signal["confidence"], 1.0)
        self.assertEqual(signal["metadata"]["scorer"], "bootstrap")
        self.assertTrue(signal["metadata"]["is_placeholder"])

    def test_image_path_preserved(self) -> None:
        record = build_detection_record(self.rows[8517], self.scorer).to_dict()
        self.assertEqual(record["image"], "vg/images/2320679.jpg")


class ScorerRegistryTests(unittest.TestCase):
    def test_bootstrap_scorer_is_registered(self) -> None:
        scorer = get_scorer("bootstrap")
        self.assertEqual(scorer.name, "bootstrap")

    def test_unknown_scorer_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_scorer("does-not-exist")


if __name__ == "__main__":
    unittest.main()
