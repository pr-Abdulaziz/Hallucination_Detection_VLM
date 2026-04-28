from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.paper.run_released_pref_stage3_validate import (
    _Stats,
    _build_validation_prompt,
    _iter_rows,
    _prompt_version,
    _validate_row,
)
from fg_pipeline.paper.run_released_pref_stage4_repair import main as repair_main


class _FakeJudge:
    def __init__(self, family: str, approved: bool, reason: str = "ok") -> None:
        self.family = family
        self._approved = approved
        self._reason = reason

    def judge(self, row: dict) -> dict:
        return {
            "family": self.family,
            "approved": self._approved,
            "reason": self._reason,
            "raw_output": json.dumps({"approved": self._approved, "reason": self._reason}),
        }


def _preference_row(row_id: int = 1) -> dict:
    return {
        "id": row_id,
        "image": f"{row_id}.jpg",
        "question": "What is shown?",
        "chosen": "The image shows people walking on a rainy street.",
        "rejected": "The image shows people standing on a sunny beach.",
        "rejected_score": 2.0,
        "rejected_tag_text": "object hallucination: sunny beach; severity moderate",
        "metadata": {"source": "released_preference"},
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ReleasedPreferenceValidationTests(unittest.TestCase):
    def test_two_shot_prompt_adds_calibration_examples(self) -> None:
        prompt = _build_validation_prompt(_preference_row(), prompt_mode="two_shot")

        self.assertIn("Example 1:", prompt)
        self.assertIn("Example 2:", prompt)
        self.assertIn("Target pair to judge:", prompt)
        self.assertEqual(_prompt_version("two_shot"), "released_pref_validation_2shot_v1")

    def test_either_rule_accepts_when_one_judge_approves(self) -> None:
        row = _preference_row()
        preference, audit = _validate_row(
            row,
            [_FakeJudge("gemini", True), _FakeJudge("openai", False, "unsupported detail")],
            decision_rule="either",
            strict=False,
            image_root=Path("."),
        )

        self.assertTrue(audit["approved"])
        self.assertTrue(preference["validation_approved"])
        self.assertEqual(preference["metadata"]["source_stage"], "released_pref_stage3_validation")
        self.assertEqual(len(preference["metadata"]["api_votes"]), 2)

    def test_both_rule_rejects_when_one_judge_rejects(self) -> None:
        _, audit = _validate_row(
            _preference_row(),
            [_FakeJudge("gemini", True), _FakeJudge("openai", False, "unsupported detail")],
            decision_rule="both",
            strict=False,
            image_root=Path("."),
        )

        self.assertFalse(audit["approved"])

    def test_iter_rows_splits_accepted_and_rejected_preferences(self) -> None:
        stats = _Stats(input_path="prefs.jsonl", api_judge="gemini_openai", decision_rule="either")
        accepted, rejected, audits = _iter_rows(
            [_preference_row(1), _preference_row(2)],
            judges=[_FakeJudge("gemini", False), _FakeJudge("openai", False)],
            stats=stats,
            decision_rule="either",
            strict=False,
            image_root=Path("."),
            limit=None,
            total=2,
        )

        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(audits), 2)
        self.assertEqual(stats.to_dict()["rejected_rows"], 2)


class ReleasedPreferenceRepairTests(unittest.TestCase):
    def test_template_repair_merges_accepted_and_repaired_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            accepted = tmp_dir / "accepted.jsonl"
            rejected = tmp_dir / "rejected.jsonl"
            repair_out = tmp_dir / "repair_records.jsonl"
            repaired_out = tmp_dir / "repaired_preferences.jsonl"
            final_out = tmp_dir / "final_preferences.jsonl"
            stats_out = tmp_dir / "stats.json"

            write_jsonl(accepted, [_preference_row(1)])
            write_jsonl(rejected, [_preference_row(2)])

            rc = repair_main(
                [
                    "--rejected-input",
                    str(rejected),
                    "--accepted-input",
                    str(accepted),
                    "--repair-out",
                    str(repair_out),
                    "--repaired-preferences-out",
                    str(repaired_out),
                    "--final-preferences-out",
                    str(final_out),
                    "--stats-out",
                    str(stats_out),
                    "--backend",
                    "template",
                ]
            )

            self.assertEqual(rc, 0)
            repaired_rows = _read_jsonl(repaired_out)
            final_rows = _read_jsonl(final_out)
            stats = json.loads(stats_out.read_text(encoding="utf-8"))

            self.assertEqual(len(repaired_rows), 1)
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(stats["accepted_rows"], 1)
            self.assertEqual(stats["repaired_rows"], 1)
            self.assertEqual(repaired_rows[0]["metadata"]["source_stage"], "released_pref_stage4_repair")
            self.assertNotEqual(repaired_rows[0]["chosen"], repaired_rows[0]["rejected"])


if __name__ == "__main__":
    unittest.main()
