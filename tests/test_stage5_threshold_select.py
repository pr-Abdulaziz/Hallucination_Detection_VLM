from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.verification.backends import HeuristicVerificationBackend
from fg_pipeline.verification.run_verify import _resolve_min_pair_confidence
from fg_pipeline.verification.threshold_selection import (
    build_pair_candidates,
    select_crc_threshold,
    select_cv_crc_threshold,
)


def _sig(conf: float, *, span: str | None = "a blue dog", severity: int = 3) -> dict:
    return {
        "sentence_index": 0,
        "hallucination_type": "object",
        "severity": severity,
        "confidence": conf,
        "metadata": {"span": span} if span else {},
    }


def _row(
    sample_id: str,
    conf: float,
    *,
    good: bool = True,
) -> dict:
    source = "a red car and a blue dog in a park"
    rewritten = (
        "a red car and a [removed] in a park"
        if good
        else "a red car and a blue dog nearby in a park"
    )
    return {
        "sample_id": sample_id,
        "image": "VG_100K/1.jpg",
        "prompt": "Describe this image in detail.",
        "source_response": source,
        "rewritten_response": rewritten,
        "filtered_signals": [_sig(conf)],
        "metadata": {
            "rewrite_status": "generated",
            "rewrite_backend": "template",
        },
    }


class ThresholdSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_build_pair_candidates_uses_verifier_target(self) -> None:
        rows = [_row("good", 0.9, good=True), _row("bad", 0.8, good=False)]
        candidates = build_pair_candidates(rows, self.backend)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[0].is_good)
        self.assertFalse(candidates[1].is_good)

    def test_crc_selects_threshold_that_drops_bad_high_risk_tail(self) -> None:
        rows = [
            _row("g1", 0.9, good=True),
            _row("g2", 0.8, good=True),
            _row("b1", 0.2, good=False),
        ]
        candidates = build_pair_candidates(rows, self.backend)
        report = select_crc_threshold(candidates, alpha=0.4, min_accepted=1)
        self.assertTrue(report["valid"])
        self.assertGreaterEqual(report["threshold"], 0.2)
        self.assertLessEqual(report["upper_bound"], 0.4)

    def test_cv_crc_returns_selected_tau_c(self) -> None:
        rows = [
            _row("g1", 0.9, good=True),
            _row("g2", 0.85, good=True),
            _row("g3", 0.8, good=True),
            _row("b1", 0.1, good=False),
            _row("b2", 0.05, good=False),
        ]
        candidates = build_pair_candidates(rows, self.backend)
        report = select_cv_crc_threshold(
            candidates,
            alpha=0.4,
            num_folds=3,
            min_accepted=1,
        )
        self.assertEqual(report["method"], "cv_crc")
        self.assertIn("selected_tau_c", report)
        self.assertGreaterEqual(report["selected_tau_c"], 0.0)

    def test_run_verify_can_load_selected_tau_c_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "tau_c.json"
            report_path.write_text(json.dumps({"selected_tau_c": 0.123}), encoding="utf-8")
            args = argparse.Namespace(
                min_pair_confidence=0.0,
                threshold_report=str(report_path),
            )
            self.assertAlmostEqual(_resolve_min_pair_confidence(args), 0.123)


if __name__ == "__main__":
    unittest.main()
