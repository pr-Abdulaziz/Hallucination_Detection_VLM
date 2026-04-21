from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable

from fg_pipeline.confidence.scoring import average_confidence
from fg_pipeline.verification.backends import VerificationBackend
from fg_pipeline.verification.run_verify import evaluate_pair


@dataclass(frozen=True)
class PairCandidate:
    sample_id: str
    pair_confidence: float
    is_good: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_pair_candidates(
    rows: Iterable[dict],
    backend: VerificationBackend,
    limit: int | None = None,
) -> list[PairCandidate]:
    rows_list = list(rows)
    if limit is not None:
        rows_list = rows_list[:limit]

    candidates: list[PairCandidate] = []
    for row in rows_list:
        filtered = row.get("filtered_signals", []) or []
        pair_confidence = average_confidence(
            [
                type(
                    "_Signal",
                    (),
                    {"confidence": float(signal.get("confidence", 0.0))},
                )()
                for signal in filtered
            ]
        )
        record, reason = evaluate_pair(row, backend, min_pair_confidence=-1.0)
        candidates.append(
            PairCandidate(
                sample_id=str(row.get("sample_id", "")),
                pair_confidence=pair_confidence,
                is_good=record is not None,
                reason=reason or "kept",
            )
        )
    return candidates


def evaluate_threshold(candidates: Iterable[PairCandidate], threshold: float) -> dict:
    candidates_list = list(candidates)
    accepted = [
        candidate for candidate in candidates_list if candidate.pair_confidence > threshold
    ]
    accepted_count = len(accepted)
    bad_count = sum(1 for candidate in accepted if not candidate.is_good)
    upper_bound = 0.0 if accepted_count == 0 else (bad_count + 1) / (accepted_count + 1)
    empirical_risk = 0.0 if accepted_count == 0 else bad_count / accepted_count
    total_count = len(candidates_list)
    coverage = 0.0 if total_count == 0 else accepted_count / total_count
    return {
        "threshold": threshold,
        "accepted": accepted_count,
        "bad": bad_count,
        "empirical_risk": empirical_risk,
        "upper_bound": upper_bound,
        "coverage": coverage,
    }


def select_crc_threshold(
    candidates: Iterable[PairCandidate],
    *,
    alpha: float = 0.10,
    min_accepted: int = 1,
) -> dict:
    candidates_list = list(candidates)
    candidate_thresholds = sorted({0.0, *[candidate.pair_confidence for candidate in candidates_list]})
    fallback = {
        "threshold": max(candidate_thresholds, default=0.0),
        "accepted": 0,
        "bad": 0,
        "empirical_risk": 0.0,
        "upper_bound": 0.0,
        "coverage": 0.0,
        "valid": False,
    }

    for threshold in candidate_thresholds:
        stats = evaluate_threshold(candidates_list, threshold)
        if stats["accepted"] < min_accepted:
            continue
        if stats["upper_bound"] <= alpha:
            return {
                **stats,
                "alpha": alpha,
                "min_accepted": min_accepted,
                "valid": True,
            }
    return {
        **fallback,
        "alpha": alpha,
        "min_accepted": min_accepted,
    }


def _split_folds(candidates: list[PairCandidate], num_folds: int) -> list[list[PairCandidate]]:
    folds = [[] for _ in range(num_folds)]
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.pair_confidence, candidate.sample_id),
    )
    for idx, candidate in enumerate(ordered):
        folds[idx % num_folds].append(candidate)
    return folds


def select_cv_crc_threshold(
    candidates: Iterable[PairCandidate],
    *,
    alpha: float = 0.10,
    num_folds: int = 5,
    min_accepted: int = 1,
) -> dict:
    candidates_list = list(candidates)
    if num_folds < 2 or len(candidates_list) < num_folds:
        full = select_crc_threshold(
            candidates_list, alpha=alpha, min_accepted=min_accepted
        )
        return {
            "method": "crc",
            "selected_tau_c": full["threshold"],
            "full_data": full,
            "folds": [],
        }

    folds = _split_folds(candidates_list, num_folds)
    fold_reports: list[dict] = []
    fold_thresholds: list[float] = []
    for idx in range(num_folds):
        holdout = folds[idx]
        calibration = [
            candidate
            for fold_idx, fold in enumerate(folds)
            if fold_idx != idx
            for candidate in fold
        ]
        selected = select_crc_threshold(
            calibration,
            alpha=alpha,
            min_accepted=min_accepted,
        )
        fold_thresholds.append(float(selected["threshold"]))
        holdout_eval = evaluate_threshold(holdout, float(selected["threshold"]))
        fold_reports.append(
            {
                "fold_index": idx,
                "selected_threshold": float(selected["threshold"]),
                "calibration": selected,
                "holdout": holdout_eval,
            }
        )

    selected_tau_c = float(median(fold_thresholds))
    return {
        "method": "cv_crc",
        "selected_tau_c": selected_tau_c,
        "alpha": alpha,
        "num_folds": num_folds,
        "min_accepted": min_accepted,
        "folds": fold_reports,
        "full_data": evaluate_threshold(candidates_list, selected_tau_c),
        "full_data_crc": select_crc_threshold(
            candidates_list, alpha=alpha, min_accepted=min_accepted
        ),
    }
