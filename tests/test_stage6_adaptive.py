from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch

from fg_pipeline.adaptive_dpo import normalize_preference_item, resolve_image_path
from fg_pipeline.adaptive_dpo.adaptive_loss import adaptive_example_weight
from fg_pipeline.adaptive_dpo.trainer_adaptive import AdaptiveLlavaDPOTrainer
from fg_pipeline.confidence.run_detect import build_detection_record
from fg_pipeline.confidence.scoring import BootstrapScorer
from fg_pipeline.rewrite.backends import TemplateRewriteBackend
from fg_pipeline.rewrite.run_rewrite import build_rewrite_record
from fg_pipeline.verification.backends import HeuristicVerificationBackend
from fg_pipeline.verification.run_verify import evaluate_pair

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fg_pipeline"
    / "data"
    / "smoke_detection.jsonl"
)


def _load_fixture_row(row_id: int) -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["id"]) == row_id:
                return row
    raise KeyError(row_id)


class NormalizePreferenceItemTests(unittest.TestCase):
    def test_preserves_stage5_image_and_adaptive_fields(self) -> None:
        item = {
            "id": "s1",
            "image": "vg/images/2397397.jpg",
            "question": "Describe this image in detail.",
            "chosen": "chosen",
            "rejected": "rejected",
            "chosen_score": 1.0,
            "rejected_score": 2.5,
            "pair_confidence": 0.8,
            "severity_weight": 2.5,
            "adaptive_weight": 2.0,
        }
        sample = normalize_preference_item(item)
        self.assertEqual(sample["image"], item["image"])
        self.assertAlmostEqual(sample["pair_confidence"], 0.8)
        self.assertAlmostEqual(sample["severity_weight"], 2.5)
        self.assertAlmostEqual(sample["adaptive_weight"], 2.0)

    def test_legacy_rows_remain_backward_compatible(self) -> None:
        item = {
            "id": 7,
            "question": "Describe this image in detail.",
            "chosen": "chosen",
            "rejected": "rejected",
            "rejected_score": 1.3,
        }
        sample = normalize_preference_item(item)
        self.assertEqual(sample["id"], 7)
        self.assertNotIn("pair_confidence", sample)
        self.assertNotIn("severity_weight", sample)
        self.assertNotIn("adaptive_weight", sample)


class ResolveImagePathTests(unittest.TestCase):
    def test_prefers_explicit_stage5_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "vg" / "images" / "2397397.jpg"
            explicit.parent.mkdir(parents=True, exist_ok=True)
            explicit.write_bytes(b"fake")

            fallback = root / "8500.jpg"
            fallback.write_bytes(b"fallback")

            resolved = resolve_image_path("vg/images/2397397.jpg", root, fallback_id=8500)
            self.assertEqual(resolved, explicit)

    def test_falls_back_to_legacy_id_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = root / "12.jpg"
            fallback.write_bytes(b"fallback")

            resolved = resolve_image_path(None, root, fallback_id=12)
            self.assertEqual(resolved, fallback)


class AdaptiveReductionTests(unittest.TestCase):
    def test_reduce_weighted_losses_uses_explicit_adaptive_weights(self) -> None:
        trainer = AdaptiveLlavaDPOTrainer.__new__(AdaptiveLlavaDPOTrainer)
        losses = torch.tensor([1.0, 3.0], dtype=torch.float32)
        weights = torch.tensor([0.5, 1.5], dtype=torch.float32)
        reduced = trainer.reduce_weighted_losses(losses, adaptive_weights=weights)
        self.assertAlmostEqual(reduced.item(), 2.5, places=6)

    def test_reduce_weighted_losses_can_recompute_from_fields(self) -> None:
        trainer = AdaptiveLlavaDPOTrainer.__new__(AdaptiveLlavaDPOTrainer)
        losses = torch.tensor([1.0, 3.0], dtype=torch.float32)
        pair_confidences = torch.tensor([0.5, 0.8], dtype=torch.float32)
        severity_weights = torch.tensor([2.0, 3.0], dtype=torch.float32)
        expected_weights = [
            adaptive_example_weight(0.5, 2.0),
            adaptive_example_weight(0.8, 3.0),
        ]
        expected = (
            losses[0] * expected_weights[0] + losses[1] * expected_weights[1]
        ) / sum(expected_weights)
        reduced = trainer.reduce_weighted_losses(
            losses,
            pair_confidences=pair_confidences,
            severity_weights=severity_weights,
        )
        self.assertAlmostEqual(reduced.item(), float(expected), places=6)


class StageBoundaryCompatibilityTests(unittest.TestCase):
    def test_stage3_to_stage6_schema_handoff_is_consistent(self) -> None:
        row = _load_fixture_row(8500)

        detection = build_detection_record(row, BootstrapScorer())
        rewrite = build_rewrite_record(
            detection.to_dict(),
            TemplateRewriteBackend(),
            confidence_threshold=0.5,
        )
        preference, reason = evaluate_pair(
            rewrite.to_dict(),
            HeuristicVerificationBackend(),
            min_pair_confidence=0.5,
        )

        self.assertEqual(reason, "")
        self.assertIsNotNone(preference)
        normalized = normalize_preference_item(preference.to_dict())

        self.assertEqual(normalized["id"], detection.sample_id)
        self.assertEqual(normalized["image"], detection.image)
        self.assertEqual(normalized["question"], detection.prompt)
        self.assertEqual(normalized["chosen"], preference.chosen)
        self.assertEqual(normalized["rejected"], preference.rejected)
        self.assertIn("pair_confidence", normalized)
        self.assertIn("severity_weight", normalized)
        self.assertIn("adaptive_weight", normalized)


if __name__ == "__main__":
    unittest.main()
