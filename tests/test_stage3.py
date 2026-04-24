"""Stage 3 unit and smoke tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fg_pipeline.stage3.backends as stage3_backends
import fg_pipeline.stage3.prompts as stage3_prompts
import fg_pipeline.stage3.run_stage3 as stage3_run
from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.stage1.schemas import CritiqueItem
from fg_pipeline.stage2.schemas import Stage2Record
from fg_pipeline.stage3 import (
    APPROVALS_REQUIRED,
    GEMINI_LLAVA_TWO_VOTE_POLICY_VERSION,
    GEMINI_TWO_VOTE_POLICY_VERSION,
    GeminiLlavaTwoVoteBackend,
    GeminiTwoVoteBackend,
    HeuristicVerificationBackend,
    Stage3Record,
    VOTE_COUNT,
    VoteDecision,
    get_backend,
)
from fg_pipeline.stage3.backends import VerificationError


_CONFIDENCE_FRAGMENTS = (
    "confidence",
    "calibration",
    "threshold",
    "tau",
    "crc",
    "cv_crc",
    "probability",
)


def _assert_no_confidence_keys(testcase: unittest.TestCase, obj) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = key.lower()
            for fragment in _CONFIDENCE_FRAGMENTS:
                testcase.assertNotIn(fragment, lowered, f"unexpected key {key!r}")
            _assert_no_confidence_keys(testcase, value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_confidence_keys(testcase, item)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_critique(
    *,
    idx: int = 1,
    evidence: str | None = "gold cross",
    severity_score: int | None = 3,
) -> dict:
    return CritiqueItem(
        index=idx,
        hallucination_type="object",
        severity_label="major" if severity_score else "unknown",
        severity_score=severity_score,
        rationale="The claim introduces unsupported content",
        evidence_text=evidence,
        source_tag_text=None,
        source_score_text=None,
    ).to_dict()


def _make_stage2_record(
    *,
    row_id: int = 1,
    original: str = "The image shows a gold cross on the clock tower.",
    rewrite: str = "The image shows the clock tower.",
    critiques: list[dict] | None = None,
) -> Stage2Record:
    return Stage2Record(
        id=row_id,
        image="vg/images/test.jpg",
        question="What is in the image?",
        original_response=original,
        rewrite_response=rewrite,
        critiques=critiques or [_make_critique()],
        metadata={"source_stage": "stage2_rewrite", "backend": "template", "prompt_version": "v1"},
    )


class SeverityAggregationTests(unittest.TestCase):
    def test_mean_of_known_scores(self) -> None:
        critiques = [
            _make_critique(idx=1, severity_score=3),
            _make_critique(idx=2, severity_score=2),
            _make_critique(idx=3, severity_score=1),
        ]
        self.assertAlmostEqual(stage3_run._aggregate_severity(critiques), 2.0)

    def test_unknown_only_defaults_to_one(self) -> None:
        critiques = [_make_critique(severity_score=None)]
        self.assertAlmostEqual(stage3_run._aggregate_severity(critiques), 1.0)


class HeuristicBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_vote_one_approves_when_evidence_removed(self) -> None:
        record = _make_stage2_record().to_dict()
        vote = self.backend.vote(record, vote_index=1)
        self.assertTrue(vote.approved)
        self.assertEqual(vote.criterion, "hallucination_removal")

    def test_vote_one_rejects_when_evidence_remains(self) -> None:
        record = _make_stage2_record(
            rewrite="The image shows a gold cross on the clock tower. [corrected]"
        ).to_dict()
        vote = self.backend.vote(record, vote_index=1)
        self.assertFalse(vote.approved)

    def test_vote_two_rejects_corrected_marker(self) -> None:
        record = _make_stage2_record(
            rewrite="The image shows the clock tower. [corrected]"
        ).to_dict()
        vote = self.backend.vote(record, vote_index=2)
        self.assertFalse(vote.approved)

    def test_vote_three_approves_good_rewrite(self) -> None:
        record = _make_stage2_record().to_dict()
        vote = self.backend.vote(record, vote_index=3)
        self.assertTrue(vote.approved)


class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_two_of_three_votes_keeps_pair(self) -> None:
        good = _make_stage2_record().to_dict()
        audit_rows, pref_rows, stats = stage3_run._process_rows(
            self.backend,
            [good],
            strict=False,
            limit=None,
        )
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(len(pref_rows), 1)
        self.assertTrue(audit_rows[0]["passed_majority"])
        self.assertGreaterEqual(audit_rows[0]["approvals"], APPROVALS_REQUIRED)
        self.assertEqual(stats.preference_pairs_emitted, 1)

    def test_one_of_three_votes_drops_pair(self) -> None:
        bad = _make_stage2_record(
            row_id=2,
            rewrite="The image shows a gold cross on the clock tower. [corrected]",
            critiques=[_make_critique(evidence="gold cross")],
        ).to_dict()
        audit_rows, pref_rows, stats = stage3_run._process_rows(
            self.backend,
            [bad],
            strict=False,
            limit=None,
        )
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(pref_rows, [])
        self.assertFalse(audit_rows[0]["passed_majority"])
        self.assertEqual(stats.dropped_rows, 1)

    def test_preference_row_is_trainer_compatible(self) -> None:
        good = _make_stage2_record().to_dict()
        _, pref_rows, _ = stage3_run._process_rows(
            self.backend,
            [good],
            strict=False,
            limit=None,
        )
        self.assertEqual(len(pref_rows), 1)
        pref = pref_rows[0]
        self.assertEqual(pref["id"], 1)
        self.assertEqual(pref["question"], "What is in the image?")
        self.assertEqual(pref["chosen"], "The image shows the clock tower.")
        self.assertEqual(pref["rejected"], "The image shows a gold cross on the clock tower.")
        self.assertAlmostEqual(pref["rejected_score"], 3.0)
        self.assertEqual(pref["image"], "vg/images/test.jpg")

    def test_no_confidence_fields_in_audit_or_preferences(self) -> None:
        good = _make_stage2_record().to_dict()
        audit_rows, pref_rows, _ = stage3_run._process_rows(
            self.backend,
            [good],
            strict=False,
            limit=None,
        )
        for row in audit_rows + pref_rows:
            _assert_no_confidence_keys(self, row)


class BackendRegistryTests(unittest.TestCase):
    def test_get_backend_returns_heuristic(self) -> None:
        backend = get_backend("heuristic")
        self.assertIsInstance(backend, HeuristicVerificationBackend)

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("not_a_backend")


class JudgeParsingTests(unittest.TestCase):
    def test_extracts_fenced_json(self) -> None:
        payload = stage3_backends._extract_json_object(
            '```json\n{"approved": true, "reason": "ok"}\n```'
        )
        self.assertTrue(payload["approved"])
        self.assertEqual(payload["reason"], "ok")

    def test_extracts_json_surrounded_by_text(self) -> None:
        payload = stage3_backends._extract_json_object(
            'Here is my decision:\n{"approved": false, "reason": "not preserved"}\nThanks.'
        )
        self.assertFalse(payload["approved"])
        self.assertEqual(payload["reason"], "not preserved")

    def test_fallback_parses_key_value_decision(self) -> None:
        payload = stage3_backends._extract_json_object(
            "approved: true\nreason: removes the unsupported claim"
        )
        self.assertTrue(payload["approved"])
        self.assertIn("removes", payload["reason"])

    def test_fallback_parses_plain_no(self) -> None:
        payload = stage3_backends._extract_json_object(
            "No, the rewrite deletes correct visual detail."
        )
        self.assertFalse(payload["approved"])

    def test_unparseable_response_mentions_raw_snippet(self) -> None:
        with self.assertRaisesRegex(ValueError, "could not parse judge response"):
            stage3_backends._extract_json_object("unclear maybe perhaps")


class PromptTests(unittest.TestCase):
    def test_prompt_avoids_decision_and_reason_placeholders(self) -> None:
        prompt = stage3_prompts.build_vote_prompt(
            _make_stage2_record().to_dict(),
            "hallucination_removal",
        )
        final_line = prompt.strip().splitlines()[-1]
        self.assertEqual(final_line, "Return only the JSON object now.")
        self.assertNotIn('"approved": false', prompt)
        self.assertNotIn("short reason", prompt)
        self.assertIn("do not use placeholder text", prompt)


class GeminiLlavaTwoVoteBackendTests(unittest.TestCase):
    class _FakeRuntime:
        def __init__(self, responses: dict[str, str]) -> None:
            self.responses = responses
            self.calls: list[str] = []

        def judge(self, record, criterion: str) -> str:
            self.calls.append(criterion)
            return self.responses[criterion]

    @staticmethod
    def _response(approved: bool, reason: str) -> str:
        return json.dumps({"approved": approved, "reason": reason})

    def _backend_for_outcomes(self, vote1: bool, vote2: bool) -> GeminiLlavaTwoVoteBackend:
        return GeminiLlavaTwoVoteBackend(
            llava_model_path="models/llava",
            gemini_runtime=self._FakeRuntime(
                {"hallucination_removal": self._response(vote1, "gemini-one")}
            ),
            llava_runtime=self._FakeRuntime(
                {"content_preservation": self._response(vote2, "llava-two")}
            ),
        )

    def test_two_vote_backend_runs_only_gemini_and_llava_votes(self) -> None:
        backend = self._backend_for_outcomes(True, True)
        audit_rows, pref_rows, stats = stage3_run._process_rows(
            backend,
            [_make_stage2_record().to_dict()],
            strict=False,
            limit=None,
        )
        self.assertEqual(len(audit_rows[0]["votes"]), 2)
        self.assertEqual([vote["model_family"] for vote in audit_rows[0]["votes"]], ["gemini", "llava"])
        self.assertTrue(audit_rows[0]["passed_majority"])
        self.assertEqual(len(pref_rows), 1)
        self.assertEqual(stats.vote_policy_version, GEMINI_LLAVA_TWO_VOTE_POLICY_VERSION)
        self.assertEqual(stats.vote_count, 2)
        self.assertEqual(stats.approvals_required, 2)
        self.assertEqual(audit_rows[0]["metadata"]["vote_count"], 2)
        self.assertEqual(audit_rows[0]["metadata"]["approvals_required"], 2)
        self.assertFalse(audit_rows[0]["metadata"]["early_stop_applied"])

    def test_two_vote_backend_requires_both_approvals(self) -> None:
        for vote1, vote2 in [(True, False), (False, True), (False, False)]:
            with self.subTest(vote1=vote1, vote2=vote2):
                backend = self._backend_for_outcomes(vote1, vote2)
                audit_rows, pref_rows, _ = stage3_run._process_rows(
                    backend,
                    [_make_stage2_record().to_dict()],
                    strict=False,
                    limit=None,
                )
                self.assertFalse(audit_rows[0]["passed_majority"])
                self.assertEqual(pref_rows, [])

    def test_two_vote_backend_submits_votes_in_parallel(self) -> None:
        backend = self._backend_for_outcomes(True, True)
        record = _make_stage2_record().to_dict()
        with mock.patch("fg_pipeline.stage3.run_stage3.ThreadPoolExecutor") as executor_cls:
            executor = executor_cls.return_value.__enter__.return_value
            executor.submit.side_effect = [
                mock.Mock(result=mock.Mock(return_value=backend.vote(record, vote_index=1))),
                mock.Mock(result=mock.Mock(return_value=backend.vote(record, vote_index=2))),
            ]

            audit_rows, pref_rows, _ = stage3_run._process_rows(
                backend,
                [record],
                strict=False,
                limit=None,
            )

        self.assertEqual(executor.submit.call_count, 2)
        self.assertEqual([vote["model_family"] for vote in audit_rows[0]["votes"]], ["gemini", "llava"])
        self.assertEqual(len(pref_rows), 1)


class GeminiTwoVoteBackendTests(unittest.TestCase):
    class _FakeRuntime:
        def __init__(self, responses: dict[str, str]) -> None:
            self.responses = responses

        def judge(self, record, criterion: str) -> str:
            return self.responses[criterion]

    @staticmethod
    def _response(approved: bool, reason: str) -> str:
        return json.dumps({"approved": approved, "reason": reason})

    def _backend_for_outcomes(self, vote1: bool, vote2: bool) -> GeminiTwoVoteBackend:
        return GeminiTwoVoteBackend(
            gemini_runtime=self._FakeRuntime(
                {
                    "hallucination_removal": self._response(vote1, "gemini-one"),
                    "content_preservation": self._response(vote2, "gemini-two"),
                }
            )
        )

    def test_gemini_two_vote_backend_requires_both_approvals(self) -> None:
        backend = self._backend_for_outcomes(True, True)
        audit_rows, pref_rows, stats = stage3_run._process_rows(
            backend,
            [_make_stage2_record().to_dict()],
            strict=False,
            limit=None,
        )
        self.assertTrue(audit_rows[0]["passed_majority"])
        self.assertEqual(len(pref_rows), 1)
        self.assertEqual(stats.vote_policy_version, GEMINI_TWO_VOTE_POLICY_VERSION)
        self.assertEqual(stats.vote_count, 2)
        self.assertEqual(stats.approvals_required, 2)
        self.assertEqual(audit_rows[0]["metadata"]["approved_families"], ["gemini"])

    def test_gemini_two_vote_backend_rejects_one_approval(self) -> None:
        backend = self._backend_for_outcomes(True, False)
        audit_rows, pref_rows, _ = stage3_run._process_rows(
            backend,
            [_make_stage2_record().to_dict()],
            strict=False,
            limit=None,
        )
        self.assertFalse(audit_rows[0]["passed_majority"])
        self.assertEqual(pref_rows, [])

    def test_row_workers_rejected_for_local_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "stage2.jsonl"
            write_jsonl(input_path, [_make_stage2_record().to_dict()])
            rc = stage3_run.main(
                [
                    "--input", str(input_path),
                    "--output", str(tmp_dir / "out.jsonl"),
                    "--preferences-out", str(tmp_dir / "prefs.jsonl"),
                    "--stats-out", str(tmp_dir / "stats.json"),
                    "--row-workers", "2",
                ]
            )
            self.assertEqual(rc, 2)


class CLISmokeTests(unittest.TestCase):
    def test_cli_writes_audit_and_preferences(self) -> None:
        rows = [
            _make_stage2_record(row_id=1).to_dict(),
            _make_stage2_record(
                row_id=2,
                rewrite="The image shows a gold cross on the clock tower. [corrected]",
            ).to_dict(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "stage2.jsonl"
            output_path = tmp_dir / "stage3.jsonl"
            prefs_path = tmp_dir / "prefs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(input_path, rows)

            rc = stage3_run.main(
                [
                    "--input", str(input_path),
                    "--output", str(output_path),
                    "--preferences-out", str(prefs_path),
                    "--stats-out", str(stats_path),
                    "--checkpoint-every", "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(prefs_path.exists())
            self.assertTrue(stats_path.exists())

            audit_rows = _read_jsonl(output_path)
            pref_rows = _read_jsonl(prefs_path)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

            self.assertEqual(len(audit_rows), 2)
            self.assertEqual(len(pref_rows), 1)
            self.assertEqual(stats["total_input_rows"], 2)
            self.assertEqual(stats["vote_rows_processed"], 2)
            self.assertEqual(stats["preference_pairs_emitted"], 1)
            self.assertEqual(stats["dropped_rows"], 1)
            self.assertEqual(stats["backend"], "heuristic")
            self.assertEqual(stats["vote_count"], VOTE_COUNT)
            self.assertEqual(stats["approvals_required"], APPROVALS_REQUIRED)
            self.assertEqual(stats["input_path"], str(input_path.resolve()))

    def test_missing_input_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rc = stage3_run.main(
                [
                    "--input", str(tmp_dir / "missing.jsonl"),
                    "--output", str(tmp_dir / "out.jsonl"),
                    "--preferences-out", str(tmp_dir / "prefs.jsonl"),
                    "--stats-out", str(tmp_dir / "stats.json"),
                ]
            )
            self.assertEqual(rc, 2)


class ResumeTests(unittest.TestCase):
    def test_resume_finishes_remaining_rows_without_duplicates(self) -> None:
        rows = [
            _make_stage2_record(row_id=1).to_dict(),
            _make_stage2_record(
                row_id=2,
                rewrite="The image shows a gold cross on the clock tower. [corrected]",
            ).to_dict(),
            _make_stage2_record(
                row_id=3,
                original="A gold cross is visible on the tower.",
                rewrite="The tower is visible.",
            ).to_dict(),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "stage2.jsonl"
            partial_output = tmp_dir / "partial_audit.jsonl"
            partial_prefs = tmp_dir / "partial_prefs.jsonl"
            partial_stats = tmp_dir / "partial_stats.json"
            clean_output = tmp_dir / "clean_audit.jsonl"
            clean_prefs = tmp_dir / "clean_prefs.jsonl"
            clean_stats = tmp_dir / "clean_stats.json"
            write_jsonl(input_path, rows)

            rc = stage3_run.main(
                [
                    "--input", str(input_path),
                    "--output", str(partial_output),
                    "--preferences-out", str(partial_prefs),
                    "--stats-out", str(partial_stats),
                    "--limit", "1",
                    "--checkpoint-every", "1",
                ]
            )
            self.assertEqual(rc, 0)

            rc = stage3_run.main(
                [
                    "--input", str(input_path),
                    "--output", str(partial_output),
                    "--preferences-out", str(partial_prefs),
                    "--stats-out", str(partial_stats),
                    "--resume",
                    "--checkpoint-every", "1",
                ]
            )
            self.assertEqual(rc, 0)

            rc = stage3_run.main(
                [
                    "--input", str(input_path),
                    "--output", str(clean_output),
                    "--preferences-out", str(clean_prefs),
                    "--stats-out", str(clean_stats),
                    "--checkpoint-every", "1",
                ]
            )
            self.assertEqual(rc, 0)

            resumed_audit = _read_jsonl(partial_output)
            resumed_prefs = _read_jsonl(partial_prefs)
            clean_audit_rows = _read_jsonl(clean_output)
            clean_pref_rows = _read_jsonl(clean_prefs)
            resumed_stats = json.loads(partial_stats.read_text(encoding="utf-8"))
            clean_stats_payload = json.loads(clean_stats.read_text(encoding="utf-8"))

            self.assertEqual(resumed_audit, clean_audit_rows)
            self.assertEqual(resumed_prefs, clean_pref_rows)
            self.assertEqual(
                len({row["id"] for row in resumed_audit}),
                len(resumed_audit),
            )
            self.assertEqual(
                len({row["id"] for row in resumed_prefs}),
                len(resumed_prefs),
            )
            self.assertEqual(resumed_stats, clean_stats_payload)


class StrictModeTests(unittest.TestCase):
    def test_invalid_row_raises_in_strict_mode(self) -> None:
        invalid = _make_stage2_record(rewrite="").to_dict()
        with self.assertRaises(VerificationError):
            stage3_run._process_rows(
                HeuristicVerificationBackend(),
                [invalid],
                strict=True,
                limit=None,
            )


class SchemaRoundTripTests(unittest.TestCase):
    def test_vote_decision_round_trip(self) -> None:
        vote = VoteDecision(vote_index=1, criterion="overall_preference", approved=True, reason="good")
        self.assertEqual(vote.to_dict()["criterion"], "overall_preference")

    def test_stage3_record_round_trip(self) -> None:
        record = Stage3Record(
            id=7,
            image="vg/images/x.jpg",
            question="What is in the image?",
            original_response="The sky is green.",
            rewrite_response="The sky is blue.",
            critiques=[_make_critique()],
            votes=[VoteDecision(vote_index=1, criterion="overall_preference", approved=True, reason="good")],
            approvals=3,
            rejections=0,
            passed_majority=True,
            response_severity_score=2.0,
            chosen="The sky is blue.",
            rejected="The sky is green.",
            metadata={"source_stage": "stage3_verification"},
        )
        data = record.to_dict()
        self.assertEqual(data["id"], 7)
        self.assertEqual(data["approvals"], 3)
        self.assertTrue(data["passed_majority"])
        _assert_no_confidence_keys(self, data)


if __name__ == "__main__":
    unittest.main()
