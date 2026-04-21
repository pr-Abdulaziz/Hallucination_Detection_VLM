"""Stage 4 rewrite tests — filtering, skip policy, template backend, metadata."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from fg_pipeline.rewrite.backends import (
    LLaVARewriteBackend,
    TemplateRewriteBackend,
    get_backend,
)
from fg_pipeline.rewrite.prompts import build_rewrite_prompt
from fg_pipeline.rewrite.run_rewrite import (
    _coerce_signals,
    build_rewrite_record,
    filter_signals,
    generate_records,
)
from fg_pipeline.schemas import SentenceSignal


def _sig(
    idx: int,
    conf: float,
    *,
    span: str | None = None,
    h_type: str = "object",
    severity: int = 2,
    rationale: str | None = None,
) -> dict:
    return {
        "sentence_index": idx,
        "hallucination_type": h_type,
        "severity": severity,
        "confidence": conf,
        "rationale": rationale,
        "raw_label": None,
        "metadata": {"span": span} if span else {},
    }


def _row(
    sample_id: str = "s1",
    candidate: str = "a red car and a blue dog in a park",
    signals: list[dict] | None = None,
    image: str | None = None,
    prompt: str = "Describe this image in detail.",
) -> dict:
    return {
        "sample_id": sample_id,
        "image": image,
        "prompt": prompt,
        "candidate_response": candidate,
        "signals": signals or [],
    }


class FilterSignalsTests(unittest.TestCase):
    def test_strict_greater_than_threshold(self) -> None:
        signals = _coerce_signals([_sig(0, 0.2), _sig(1, 0.5), _sig(2, 0.9)])
        kept = filter_signals(signals, 0.5)
        self.assertEqual([s.sentence_index for s in kept], [2])

    def test_threshold_zero_keeps_all_positive(self) -> None:
        signals = _coerce_signals([_sig(0, 0.01), _sig(1, 1.0)])
        kept = filter_signals(signals, 0.0)
        self.assertEqual(len(kept), 2)

    def test_preserves_order_and_metadata(self) -> None:
        signals = _coerce_signals(
            [_sig(0, 0.6, span="alpha"), _sig(1, 0.7, span="beta")]
        )
        kept = filter_signals(signals, 0.5)
        self.assertEqual([s.sentence_index for s in kept], [0, 1])
        self.assertEqual(kept[0].metadata.get("span"), "alpha")
        self.assertEqual(kept[1].metadata.get("span"), "beta")

    def test_group_threshold_policy_overrides_global_threshold(self) -> None:
        signals = _coerce_signals(
            [
                _sig(0, 0.55, h_type="object", severity=3),
                _sig(1, 0.55, h_type="attribute", severity=1),
            ]
        )
        policy = {
            "global_threshold": 0.9,
            "by_group": {
                "object|3": {"threshold": 0.6},
                "attribute|1": {"threshold": 0.5},
            },
        }
        kept = filter_signals(signals, 0.9, threshold_policy=policy)
        self.assertEqual([s.sentence_index for s in kept], [1])


class SkipPolicyTests(unittest.TestCase):
    def test_no_signals_is_skipped(self) -> None:
        record = build_rewrite_record(
            _row(signals=[]), TemplateRewriteBackend(), 0.5
        )
        self.assertEqual(record.source_response, record.rewritten_response)
        self.assertEqual(record.filtered_signals, [])
        self.assertEqual(
            record.metadata["rewrite_status"], "skipped_no_reliable_signals"
        )
        self.assertEqual(record.metadata["num_input_signals"], 0)
        self.assertEqual(record.metadata["num_filtered_signals"], 0)
        self.assertEqual(record.metadata["rewrite_backend"], "template")

    def test_all_below_threshold_is_skipped(self) -> None:
        row = _row(signals=[_sig(0, 0.1, span="a red car"), _sig(1, 0.2)])
        record = build_rewrite_record(row, TemplateRewriteBackend(), 0.5)
        self.assertEqual(
            record.metadata["rewrite_status"], "skipped_no_reliable_signals"
        )
        self.assertEqual(record.metadata["num_input_signals"], 2)
        self.assertEqual(record.metadata["num_filtered_signals"], 0)
        self.assertEqual(record.source_response, record.rewritten_response)


class TemplateBackendTests(unittest.TestCase):
    def test_removes_flagged_span_and_tags_smoke_only(self) -> None:
        row = _row(
            candidate="a red car and a blue dog in a park",
            signals=[_sig(0, 0.9, span="a blue dog")],
        )
        record = build_rewrite_record(row, TemplateRewriteBackend(), 0.5)
        self.assertNotEqual(record.source_response, record.rewritten_response)
        self.assertIn("[removed]", record.rewritten_response)
        self.assertEqual(
            record.metadata["rewrite_backend"], "template"
        )
        self.assertEqual(
            record.metadata["rewrite_status"], "generated_smoke_only"
        )
        self.assertEqual(record.metadata["removed_spans"], ["a blue dog"])
        self.assertEqual(record.metadata["num_filtered_signals"], 1)
        self.assertEqual(record.metadata["num_input_signals"], 1)
        self.assertEqual(
            record.metadata["confidence_threshold"], 0.5
        )

    def test_no_matching_span_still_differs(self) -> None:
        # Template must not produce chosen==rejected when rewrite ran; if no
        # span matches, it appends a conservative caveat instead.
        row = _row(
            candidate="a red car and a blue dog in a park",
            signals=[_sig(0, 0.9, span="a purple cat")],
        )
        record = build_rewrite_record(row, TemplateRewriteBackend(), 0.5)
        self.assertNotEqual(record.source_response, record.rewritten_response)
        self.assertEqual(
            record.metadata["rewrite_status"], "generated_smoke_only"
        )
        self.assertEqual(record.metadata["removed_spans"], [])

    def test_preserves_filtered_signals_exactly(self) -> None:
        row = _row(
            signals=[
                _sig(0, 0.2, span="low"),
                _sig(1, 0.8, span="a blue dog", rationale="not a dog"),
                _sig(2, 0.9, span="park"),
            ]
        )
        record = build_rewrite_record(row, TemplateRewriteBackend(), 0.5)
        self.assertEqual(len(record.filtered_signals), 2)
        indices = [s.sentence_index for s in record.filtered_signals]
        confidences = [s.confidence for s in record.filtered_signals]
        self.assertEqual(indices, [1, 2])
        # c^j carried forward unchanged.
        self.assertAlmostEqual(confidences[0], 0.8)
        self.assertAlmostEqual(confidences[1], 0.9)
        self.assertEqual(
            record.filtered_signals[0].rationale, "not a dog"
        )

    def test_top_level_stage5_fields_present(self) -> None:
        row = _row(
            sample_id="id-7",
            image="VG_100K/1.jpg",
            prompt="Describe this image.",
            candidate="a cat on a mat",
            signals=[_sig(0, 0.9, span="a cat")],
        )
        record = build_rewrite_record(row, TemplateRewriteBackend(), 0.5)
        payload = record.to_dict()
        for field in (
            "sample_id",
            "image",
            "prompt",
            "source_response",
            "rewritten_response",
            "filtered_signals",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["sample_id"], "id-7")
        self.assertEqual(payload["image"], "VG_100K/1.jpg")


class GenerateRecordsTests(unittest.TestCase):
    def test_mixed_rows_with_limit(self) -> None:
        rows = [
            _row("a", signals=[]),  # skipped
            _row("b", candidate="a dog", signals=[_sig(0, 0.9, span="a dog")]),
            _row("c", candidate="a cat", signals=[_sig(0, 0.9, span="a cat")]),
        ]
        out = generate_records(
            rows, TemplateRewriteBackend(), confidence_threshold=0.5, limit=2
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0]["metadata"]["rewrite_status"], "skipped_no_reliable_signals"
        )
        self.assertEqual(
            out[1]["metadata"]["rewrite_status"], "generated_smoke_only"
        )

    def test_group_threshold_policy_is_traced_in_metadata(self) -> None:
        rows = [_row("b", candidate="a dog", signals=[_sig(0, 0.9, span="a dog")])]
        policy = {"global_threshold": 0.4, "by_group": {"object|2": {"threshold": 0.4}}}
        out = generate_records(
            rows,
            TemplateRewriteBackend(),
            confidence_threshold=0.1,
            threshold_policy=policy,
        )
        self.assertEqual(out[0]["metadata"]["threshold_policy"], "group_conditional")


class BackendRegistryTests(unittest.TestCase):
    def test_template_is_registered(self) -> None:
        backend = get_backend("template")
        self.assertEqual(backend.name, "template")

    def test_llava_whitelist_filters_unknown_kwargs(self) -> None:
        # Unknown kwargs must be dropped before construction; required
        # kwargs still reach the class.
        backend = get_backend(
            "llava", model_path="/fake/path", unknown_flag="x"
        )
        self.assertEqual(backend.name, "llava")
        self.assertEqual(backend.model_path, "/fake/path")

    def test_unknown_backend_errors(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("does_not_exist")


class LLaVALoaderTests(unittest.TestCase):
    def test_load_uses_stage3_loader_device_contract(self) -> None:
        captured: dict = {}

        class _FakeModel:
            def __init__(self) -> None:
                self.eval_called = False

            def eval(self) -> None:
                self.eval_called = True

        fake_model = _FakeModel()

        fake_scorers = types.ModuleType("fg_pipeline.confidence.scorers_logprob")
        fake_scorers._patch_autoconfig_register = lambda: None
        fake_scorers._ensure_vendored_llava_on_path = lambda: None

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)

        fake_builder = types.ModuleType("llava.model.builder")

        def fake_load_pretrained_model(model_path, model_base, model_name, **kwargs):
            captured["model_path"] = model_path
            captured["model_base"] = model_base
            captured["model_name"] = model_name
            captured.update(kwargs)
            return "tok", fake_model, "proc", None

        fake_builder.load_pretrained_model = fake_load_pretrained_model

        fake_mm_utils = types.ModuleType("llava.mm_utils")
        fake_mm_utils.get_model_name_from_path = lambda path: "fake-llava"

        with mock.patch.dict(
            sys.modules,
            {
                "fg_pipeline.confidence.scorers_logprob": fake_scorers,
                "torch": fake_torch,
                "llava.model.builder": fake_builder,
                "llava.mm_utils": fake_mm_utils,
            },
        ):
            backend = LLaVARewriteBackend(model_path="/fake/path", device="auto")
            backend._load()

        self.assertEqual(captured["model_path"], "/fake/path")
        self.assertIsNone(captured["model_base"])
        self.assertEqual(captured["model_name"], "fake-llava")
        self.assertEqual(captured["device_map"], "auto")
        self.assertEqual(captured["device"], "cuda")
        self.assertEqual(captured["load_8bit"], False)
        self.assertEqual(captured["load_4bit"], False)
        self.assertTrue(fake_model.eval_called)


class PromptTests(unittest.TestCase):
    def test_includes_span_rationale_and_confidence(self) -> None:
        signals = [
            SentenceSignal(
                sentence_index=0,
                hallucination_type="object",
                severity=3,
                confidence=0.82,
                rationale="no dog in image",
                metadata={"span": "a blue dog"},
            )
        ]
        text = build_rewrite_prompt("Describe.", "a cat and a blue dog", signals)
        self.assertIn("span=\"a blue dog\"", text)
        self.assertIn("rationale=no dog in image", text)
        self.assertIn("confidence=0.820", text)
        self.assertIn("Original response:", text)
        self.assertIn("Rewritten response:", text)

    def test_empty_signals_reports_no_flags(self) -> None:
        text = build_rewrite_prompt("p", "r", [])
        self.assertIn("no flagged hallucinations", text)


if __name__ == "__main__":
    unittest.main()
