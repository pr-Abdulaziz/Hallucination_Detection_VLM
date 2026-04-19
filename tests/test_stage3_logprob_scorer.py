"""Mock-based tests for LogProbScorer — CPU-runnable.

The real scorer needs a GPU VLM; these tests verify control flow, error
handling, and path resolution via targeted monkey-patches without loading
torch, transformers, or the vendored LLaVA code. The end-to-end happy path
is validated by the vastai GPU run, not here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from fg_pipeline.confidence.scorers_logprob import SEVERITY_LABELS, LogProbScorer


class LogProbScorerConstructionTests(unittest.TestCase):
    def test_defaults(self) -> None:
        scorer = LogProbScorer(model_path="/fake/model")
        self.assertEqual(scorer.name, "log_prob")
        self.assertEqual(scorer.model_path, "/fake/model")
        self.assertEqual(scorer.device, "auto")
        self.assertEqual(scorer.temperature, 1.0)
        self.assertEqual(scorer.conv_template_name, "llava_v1")
        self.assertIsNotNone(scorer.image_root)
        self.assertFalse(scorer._loaded)
        self.assertEqual(scorer._severity_first_token_id, {})

    def test_custom_args(self) -> None:
        scorer = LogProbScorer(
            model_path="/m",
            device="cuda:1",
            temperature=0.7,
            image_root="/data/imgs",
            conv_template_name="llava_v0",
        )
        self.assertEqual(scorer.device, "cuda:1")
        self.assertEqual(scorer.temperature, 0.7)
        self.assertEqual(scorer.image_root, "/data/imgs")
        self.assertEqual(scorer.conv_template_name, "llava_v0")

    def test_severity_labels_are_canonical(self) -> None:
        self.assertEqual(SEVERITY_LABELS, ("Minor", "Moderate", "Major"))


class LogProbScorerErrorPathTests(unittest.TestCase):
    """Error paths exit before torch/llava imports — pure CPU tests."""

    def setUp(self) -> None:
        self.scorer = LogProbScorer(model_path="/fake")

    def _prime_loaded(self) -> None:
        """Simulate a successful lazy load (bypasses real model)."""
        self.scorer._loaded = True
        self.scorer._severity_first_token_id = {"Minor": 1, "Moderate": 2, "Major": 3}

    def test_load_failure_returns_placeholder_false_with_error(self) -> None:
        with mock.patch.object(
            LogProbScorer, "_ensure_loaded", side_effect=RuntimeError("boom")
        ):
            confidence, meta = self.scorer.score(
                {"severity_label": "major", "hallucination_type": "object", "span": "x"},
                {"image": "some.jpg", "candidate_response": "..."},
            )
        self.assertEqual(confidence, 0.0)
        self.assertFalse(meta["is_placeholder"])
        self.assertEqual(meta["scorer"], "log_prob")
        self.assertTrue(meta["error"].startswith("load_failed"))
        self.assertIn("boom", meta["error"])

    def test_unknown_severity_label(self) -> None:
        self._prime_loaded()
        confidence, meta = self.scorer.score(
            {"severity_label": "severe", "hallucination_type": "object", "span": "x"},
            {"image": "some.jpg", "candidate_response": "..."},
        )
        self.assertEqual(confidence, 0.0)
        self.assertFalse(meta["is_placeholder"])
        self.assertTrue(meta["error"].startswith("unknown_severity_label"))

    def test_empty_severity_label_is_unknown(self) -> None:
        self._prime_loaded()
        _, meta = self.scorer.score(
            {"severity_label": "", "hallucination_type": "object", "span": "x"},
            {"image": "some.jpg", "candidate_response": "..."},
        )
        self.assertTrue(meta["error"].startswith("unknown_severity_label"))

    def test_image_missing_returns_error(self) -> None:
        self._prime_loaded()
        confidence, meta = self.scorer.score(
            {"severity_label": "major", "hallucination_type": "object", "span": "x"},
            {"image": "does/not/exist.jpg", "candidate_response": "..."},
        )
        self.assertEqual(confidence, 0.0)
        self.assertFalse(meta["is_placeholder"])
        self.assertTrue(meta["error"].startswith("image_missing"))

    def test_image_missing_when_key_absent(self) -> None:
        self._prime_loaded()
        _, meta = self.scorer.score(
            {"severity_label": "minor", "hallucination_type": "object", "span": "x"},
            {"candidate_response": "..."},
        )
        self.assertTrue(meta["error"].startswith("image_missing"))


class ResolveImageTests(unittest.TestCase):
    """_resolve_image: path handling + caching without the real model."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.rel_path = "sample.jpg"
        img = Image.new("RGB", (4, 4), color=(10, 20, 30))
        img.save(self.root / self.rel_path)
        self.scorer = LogProbScorer(model_path="/fake", image_root=str(self.root))

    def test_resolves_relative_to_image_root(self) -> None:
        image = self.scorer._resolve_image(self.rel_path)
        self.assertEqual(image.size, (4, 4))
        self.assertEqual(image.mode, "RGB")

    def test_caches_resolved_image(self) -> None:
        first = self.scorer._resolve_image(self.rel_path)
        second = self.scorer._resolve_image(self.rel_path)
        self.assertIs(first, second)

    def test_missing_path_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.scorer._resolve_image("missing.jpg")

    def test_none_path_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.scorer._resolve_image(None)

    def test_absolute_path_bypasses_image_root(self) -> None:
        outside = Path(self._tmpdir.name) / "outside.jpg"
        Image.new("RGB", (2, 2)).save(outside)
        scorer = LogProbScorer(model_path="/fake", image_root="/does/not/exist")
        image = scorer._resolve_image(str(outside))
        self.assertEqual(image.size, (2, 2))


class ScorerRegistryIntegrationTests(unittest.TestCase):
    """The factory must expose log_prob without importing torch eagerly."""

    def test_log_prob_resolvable_via_factory(self) -> None:
        from fg_pipeline.confidence.scoring import get_scorer

        scorer = get_scorer("log_prob", model_path="/fake", device="cpu")
        self.assertEqual(scorer.name, "log_prob")
        self.assertEqual(scorer.model_path, "/fake")
        self.assertEqual(scorer.device, "cpu")
        self.assertFalse(scorer._loaded)


if __name__ == "__main__":
    unittest.main()
