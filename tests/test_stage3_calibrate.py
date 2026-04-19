"""Tests for fg_pipeline.confidence.run_calibrate — distribution + τ suggestions."""

from __future__ import annotations

import unittest

from fg_pipeline.confidence.run_calibrate import calibrate, format_report


def _signal(conf: float, h_type: str = "object", severity: int = 3, **meta) -> dict:
    return {
        "hallucination_type": h_type,
        "severity": severity,
        "confidence": conf,
        "metadata": {"is_placeholder": False, **meta},
    }


def _record(signals: list[dict]) -> dict:
    return {"signals": signals}


class CalibrateTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        report = calibrate([])
        self.assertEqual(report["totals"]["rows"], 0)
        self.assertEqual(report["totals"]["real_signals"], 0)
        self.assertEqual(report["overall"]["count"], 0)

    def test_placeholder_only_is_excluded(self) -> None:
        records = [
            _record([
                {
                    "hallucination_type": "object",
                    "severity": 3,
                    "confidence": 1.0,
                    "metadata": {"is_placeholder": True},
                }
            ])
        ]
        report = calibrate(records)
        self.assertEqual(report["totals"]["real_signals"], 0)
        self.assertEqual(report["totals"]["placeholder_signals"], 1)
        self.assertEqual(report["overall"]["count"], 0)

    def test_error_signals_excluded(self) -> None:
        records = [
            _record([
                {
                    "hallucination_type": "object",
                    "severity": 3,
                    "confidence": 0.0,
                    "metadata": {"is_placeholder": False, "error": "image_missing: x"},
                },
                _signal(0.7),
            ])
        ]
        report = calibrate(records)
        self.assertEqual(report["totals"]["real_signals"], 1)
        self.assertEqual(report["totals"]["error_signals"], 1)

    def test_overall_stats(self) -> None:
        records = [_record([_signal(0.1), _signal(0.5), _signal(0.9)])]
        report = calibrate(records)
        self.assertEqual(report["totals"]["real_signals"], 3)
        self.assertAlmostEqual(report["overall"]["mean"], 0.5, places=6)
        self.assertAlmostEqual(report["overall"]["median"], 0.5, places=6)
        self.assertEqual(report["overall"]["min"], 0.1)
        self.assertEqual(report["overall"]["max"], 0.9)

    def test_per_type_and_severity_bucketed(self) -> None:
        records = [
            _record([
                _signal(0.2, h_type="object", severity=3),
                _signal(0.4, h_type="object", severity=2),
                _signal(0.8, h_type="attribute", severity=1),
            ])
        ]
        report = calibrate(records)
        self.assertEqual(report["per_type"]["object"]["count"], 2)
        self.assertEqual(report["per_type"]["attribute"]["count"], 1)
        self.assertEqual(report["per_severity"]["3"]["count"], 1)
        self.assertEqual(report["per_severity"]["2"]["count"], 1)
        self.assertEqual(report["per_severity"]["1"]["count"], 1)

    def test_deciles_are_monotonic(self) -> None:
        records = [_record([_signal(v / 100) for v in range(1, 101)])]
        report = calibrate(records)
        values = list(report["deciles"].values())
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], 0.01, places=6)
        self.assertAlmostEqual(values[-1], 1.00, places=6)

    def test_suggested_tau_monotonic_and_bounded(self) -> None:
        records = [_record([_signal(v / 100) for v in range(1, 101)])]
        report = calibrate(records)
        tau_vals = list(report["suggested_tau"].values())
        self.assertEqual(tau_vals, sorted(tau_vals))
        self.assertGreaterEqual(tau_vals[0], 0.01)
        self.assertLessEqual(tau_vals[-1], 1.0)


class FormatReportTests(unittest.TestCase):
    def test_all_placeholder_prints_warning(self) -> None:
        records = [
            _record([
                {
                    "hallucination_type": "object",
                    "severity": 3,
                    "confidence": 1.0,
                    "metadata": {"is_placeholder": True},
                }
            ])
        ]
        text = format_report(calibrate(records))
        self.assertIn("WARNING", text)
        self.assertIn("SCORER=log_prob", text)
        self.assertNotIn("Suggested tau", text)

    def test_real_signals_include_tau_block(self) -> None:
        records = [_record([_signal(v / 10) for v in range(1, 11)])]
        text = format_report(calibrate(records))
        self.assertIn("Suggested tau", text)
        self.assertIn("Deciles", text)
        self.assertIn("Per hallucination type", text)
        self.assertIn("Per severity", text)


if __name__ == "__main__":
    unittest.main()
