"""Tests for fg_pipeline.confidence.run_calibrate."""

from __future__ import annotations

import unittest

from fg_pipeline.confidence.calibration import (
    apply_temperature_to_records,
    build_group_threshold_policy,
    fit_temperature,
)
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


def _triplet_signal(
    conf: float,
    *,
    severity_label: str,
    companion_log_probs: dict[str, float],
    h_type: str = "object",
    severity: int = 3,
) -> dict:
    return {
        "hallucination_type": h_type,
        "severity": severity,
        "confidence": conf,
        "metadata": {
            "is_placeholder": False,
            "severity_label": severity_label,
            "companion_log_probs": companion_log_probs,
        },
    }


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

    def test_temperature_scaling_is_applied_when_triplets_exist(self) -> None:
        records = [
            _record(
                [
                    _triplet_signal(
                        0.2,
                        severity_label="Major",
                        companion_log_probs={
                            "Minor": -2.5,
                            "Moderate": -1.5,
                            "Major": -0.1,
                        },
                    ),
                    _triplet_signal(
                        0.2,
                        severity_label="Minor",
                        companion_log_probs={
                            "Minor": -0.2,
                            "Moderate": -1.4,
                            "Major": -2.0,
                        },
                        h_type="attribute",
                        severity=1,
                    ),
                ]
            )
        ]
        report = calibrate(records)
        self.assertEqual(report["temperature_scaling"]["status"], "applied")
        self.assertGreater(report["temperature_scaling"]["num_examples"], 0)

    def test_group_threshold_policy_is_emitted(self) -> None:
        records = [
            _record(
                [
                    _signal(0.2, h_type="object", severity=3),
                    _signal(0.4, h_type="object", severity=3),
                    _signal(0.8, h_type="attribute", severity=1),
                ]
            )
        ]
        report = calibrate(records, group_quantile=0.5, min_group_count=1, shrinkage=1.0)
        policy = report["group_threshold_policy"]
        self.assertIn("global_threshold", policy)
        self.assertIn("object|3", policy["by_group"])
        self.assertIn("attribute|1", policy["by_group"])


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


class CalibrationHelpersTests(unittest.TestCase):
    def test_fit_temperature_returns_scalar(self) -> None:
        fit = fit_temperature(
            [
                ([-2.0, -1.0, -0.2], 2),
                ([-0.1, -1.2, -2.4], 0),
                ([-1.5, -0.3, -2.0], 1),
            ]
        )
        self.assertIsNotNone(fit)
        self.assertGreater(fit.temperature, 0.0)

    def test_apply_temperature_to_records_updates_signal_confidence(self) -> None:
        records = [
            _record(
                [
                    _triplet_signal(
                        0.2,
                        severity_label="Major",
                        companion_log_probs={
                            "Minor": -2.0,
                            "Moderate": -1.0,
                            "Major": -0.1,
                        },
                    )
                ]
            )
        ]
        updated = apply_temperature_to_records(records, 0.7)
        meta = updated[0]["signals"][0]["metadata"]
        self.assertEqual(meta["calibration_status"], "applied")
        self.assertIn("raw_confidence", meta)
        self.assertNotEqual(updated[0]["signals"][0]["confidence"], 0.2)

    def test_group_threshold_policy_shrinks_to_global(self) -> None:
        policy = build_group_threshold_policy(
            [
                _signal(0.1, h_type="object", severity=3),
                _signal(0.2, h_type="object", severity=3),
                _signal(0.9, h_type="attribute", severity=1),
            ],
            group_quantile=0.5,
            min_group_count=2,
            shrinkage=10.0,
        )
        self.assertLess(policy["by_group"]["attribute|1"]["threshold"], 0.9)
        self.assertGreaterEqual(
            policy["by_group"]["attribute|1"]["threshold"],
            policy["global_threshold"],
        )


if __name__ == "__main__":
    unittest.main()
