"""Stage 5 verification tests — heuristic verifier, keep/drop rules, metadata."""

from __future__ import annotations

import unittest

from fg_pipeline.verification.backends import (
    HeuristicVerificationBackend,
    VerificationResult,
    get_backend,
)
from fg_pipeline.verification.run_verify import evaluate_pair, generate_records


def _sig(
    idx: int,
    conf: float,
    *,
    span: str | None = None,
    h_type: str = "object",
    severity: int = 3,
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
    *,
    sample_id: str = "s1",
    image: str | None = "VG_100K/1.jpg",
    prompt: str = "Describe this image in detail.",
    source: str = "a red car and a blue dog in a park",
    rewritten: str = "a red car and a [removed] in a park",
    filtered_signals: list[dict] | None = None,
    rewrite_status: str = "generated_smoke_only",
    rewrite_backend: str = "template",
) -> dict:
    return {
        "sample_id": sample_id,
        "image": image,
        "prompt": prompt,
        "source_response": source,
        "rewritten_response": rewritten,
        "filtered_signals": filtered_signals or [],
        "metadata": {
            "rewrite_status": rewrite_status,
            "rewrite_backend": rewrite_backend,
        },
    }


class HeuristicVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def _verify(self, row: dict):
        from fg_pipeline.verification.run_verify import _coerce_signals

        signals = _coerce_signals(row["filtered_signals"])
        return self.backend.verify(
            row["source_response"], row["rewritten_response"], signals, {}
        )

    def test_empty_rewrite_fails(self) -> None:
        row = _row(rewritten="")
        result = self._verify(row)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "empty_rewrite")

    def test_rewrite_equals_source_fails(self) -> None:
        row = _row(source="a cat on a mat", rewritten="a cat on a mat")
        result = self._verify(row)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "rewrite_equals_source")

    def test_too_short_fails(self) -> None:
        row = _row(rewritten="short")
        result = self._verify(row)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "rewrite_too_short")

    def test_degenerate_repeat_fails(self) -> None:
        row = _row(rewritten="word word word word word")
        result = self._verify(row)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "rewrite_degenerate")

    def test_passes_when_span_removed(self) -> None:
        row = _row(
            source="a red car and a blue dog in a park",
            rewritten="a red car and a [removed] in a park",
            filtered_signals=[_sig(0, 0.9, span="a blue dog")],
        )
        result = self._verify(row)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "passed_span_check")
        self.assertEqual(result.num_verified_signals, 1)
        self.assertEqual(result.metadata["removed_spans"], ["a blue dog"])

    def test_fails_when_no_flagged_span_removed(self) -> None:
        # Rewrite differs from source but still contains every flagged span.
        row = _row(
            source="a red car and a blue dog in a park",
            rewritten="a red car and a blue dog nearby in a park",
            filtered_signals=[_sig(0, 0.9, span="a blue dog")],
        )
        result = self._verify(row)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "no_flagged_span_removed")

    def test_passes_without_spans_when_text_changed(self) -> None:
        row = _row(
            source="a cat on a mat sleeping",
            rewritten="a cat on a mat resting quietly",
            filtered_signals=[_sig(0, 0.9)],  # no span
        )
        result = self._verify(row)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "passed_without_spans")
        self.assertEqual(result.num_verified_signals, 0)


class EvaluatePairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_no_filtered_signals_drops(self) -> None:
        row = _row(filtered_signals=[])
        record, reason = evaluate_pair(row, self.backend, 0.0)
        self.assertIsNone(record)
        self.assertEqual(reason, "no_filtered_signals")

    def test_below_threshold_drops(self) -> None:
        row = _row(
            filtered_signals=[_sig(0, 0.3, span="a blue dog")],
        )
        record, reason = evaluate_pair(row, self.backend, 0.5)
        self.assertIsNone(record)
        self.assertEqual(reason, "below_pair_confidence_threshold")

    def test_threshold_is_strictly_greater(self) -> None:
        # pair_confidence == τ_c must be rejected (strict >).
        row = _row(
            filtered_signals=[_sig(0, 0.5, span="a blue dog")],
        )
        record, reason = evaluate_pair(row, self.backend, 0.5)
        self.assertIsNone(record)
        self.assertEqual(reason, "below_pair_confidence_threshold")

    def test_empty_rewrite_drops_before_verifier(self) -> None:
        row = _row(
            rewritten="",
            filtered_signals=[_sig(0, 0.9, span="a blue dog")],
        )
        record, reason = evaluate_pair(row, self.backend, 0.0)
        self.assertIsNone(record)
        self.assertEqual(reason, "empty_rewrite")

    def test_equal_rewrite_drops(self) -> None:
        # Stage 4 "skipped" rows look like this.
        row = _row(
            source="a cat on a mat",
            rewritten="a cat on a mat",
            filtered_signals=[_sig(0, 0.9, span="a cat")],
            rewrite_status="skipped_no_reliable_signals",
        )
        record, reason = evaluate_pair(row, self.backend, 0.0)
        self.assertIsNone(record)
        self.assertEqual(reason, "rewrite_equals_source")

    def test_verifier_no_span_removed_surfaces_reason(self) -> None:
        row = _row(
            source="a red car and a blue dog in a park",
            rewritten="a red car and a blue dog relaxing in a park",
            filtered_signals=[_sig(0, 0.9, span="a blue dog")],
        )
        record, reason = evaluate_pair(row, self.backend, 0.5)
        self.assertIsNone(record)
        self.assertEqual(reason, "verifier:no_flagged_span_removed")

    def test_improved_pair_is_kept(self) -> None:
        row = _row(
            filtered_signals=[
                _sig(0, 0.8, span="a blue dog", severity=3),
                _sig(1, 0.9, span="a blue dog", severity=2),
            ],
        )
        record, reason = evaluate_pair(row, self.backend, 0.5)
        self.assertIsNotNone(record)
        self.assertEqual(reason, "")
        # chosen = rewritten, rejected = source, question = prompt.
        self.assertEqual(record.chosen, row["rewritten_response"])
        self.assertEqual(record.rejected, row["source_response"])
        self.assertEqual(record.question, row["prompt"])
        self.assertEqual(record.id, row["sample_id"])
        self.assertEqual(record.image, row["image"])

    def test_pair_confidence_equals_average_of_carried_cj(self) -> None:
        row = _row(
            filtered_signals=[
                _sig(0, 0.6, span="a blue dog"),
                _sig(1, 0.9, span="a blue dog"),
            ],
        )
        record, _ = evaluate_pair(row, self.backend, 0.0)
        self.assertIsNotNone(record)
        self.assertAlmostEqual(record.pair_confidence, 0.75, places=6)

    def test_metadata_records_verification_decision(self) -> None:
        row = _row(
            filtered_signals=[_sig(0, 0.9, span="a blue dog", severity=3)],
        )
        record, _ = evaluate_pair(row, self.backend, 0.5)
        self.assertIsNotNone(record)
        meta = record.metadata
        self.assertEqual(meta["verification_backend"], "heuristic")
        self.assertEqual(meta["verification_status"], "kept")
        self.assertEqual(meta["verification_reason"], "passed_span_check")
        self.assertEqual(meta["pair_confidence_threshold"], 0.5)
        self.assertEqual(meta["num_filtered_signals"], 1)
        self.assertEqual(meta["num_verified_signals"], 1)
        # Stage 4 traceability.
        self.assertEqual(meta["source_rewrite_backend"], "template")
        self.assertEqual(meta["source_rewrite_status"], "generated_smoke_only")


class GenerateRecordsTests(unittest.TestCase):
    def test_mixed_batch_summary_counts(self) -> None:
        rows = [
            _row(sample_id="keep", filtered_signals=[_sig(0, 0.9, span="a blue dog")]),
            _row(sample_id="no_sig", filtered_signals=[]),
            _row(
                sample_id="low_conf",
                filtered_signals=[_sig(0, 0.2, span="a blue dog")],
            ),
            _row(
                sample_id="equal",
                source="a cat on a mat",
                rewritten="a cat on a mat",
                filtered_signals=[_sig(0, 0.9, span="a cat")],
            ),
            _row(
                sample_id="no_removed",
                source="a red car and a blue dog in a park",
                rewritten="a red car and a blue dog relaxing in a park",
                filtered_signals=[_sig(0, 0.9, span="a blue dog")],
            ),
        ]
        kept, reasons = generate_records(
            rows, HeuristicVerificationBackend(), min_pair_confidence=0.5
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], "keep")
        self.assertEqual(reasons["kept"], 1)
        self.assertEqual(reasons["no_filtered_signals"], 1)
        self.assertEqual(reasons["below_pair_confidence_threshold"], 1)
        self.assertEqual(reasons["rewrite_equals_source"], 1)
        self.assertEqual(reasons["verifier:no_flagged_span_removed"], 1)

    def test_limit_is_respected(self) -> None:
        rows = [
            _row(sample_id=f"s{i}", filtered_signals=[_sig(0, 0.9, span="a blue dog")])
            for i in range(5)
        ]
        kept, reasons = generate_records(
            rows, HeuristicVerificationBackend(), min_pair_confidence=0.5, limit=2
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(sum(reasons.values()), 2)


class BackendRegistryTests(unittest.TestCase):
    def test_heuristic_is_registered(self) -> None:
        backend = get_backend("heuristic")
        self.assertEqual(backend.name, "heuristic")

    def test_unknown_backend_errors(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("does_not_exist")

    def test_kwargs_whitelist_filters_unknown(self) -> None:
        backend = get_backend("heuristic", min_rewrite_chars=16, unknown_flag="x")
        self.assertEqual(backend.min_rewrite_chars, 16)


class VerificationResultTests(unittest.TestCase):
    def test_defaults(self) -> None:
        result = VerificationResult(True, "ok")
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.num_verified_signals, 0)
        self.assertEqual(result.metadata, {})


if __name__ == "__main__":
    unittest.main()
