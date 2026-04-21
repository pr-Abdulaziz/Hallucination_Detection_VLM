from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping

SEVERITY_LABELS = ("Minor", "Moderate", "Major")


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    num_examples: int
    hit_boundary: bool


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    """Keep confidence values inside a numerically safe probability range."""

    return max(eps, min(1.0 - eps, value))


def temperature_scale(probability: float, temperature: float = 1.0) -> float:
    """Simple binary temperature scaling helper for calibrated confidence."""

    probability = clamp_probability(probability)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logit = math.log(probability / (1.0 - probability))
    scaled = 1.0 / (1.0 + math.exp(-(logit / temperature)))
    return clamp_probability(scaled)


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [value / total for value in exps]


def _severity_index(label: str | None) -> int | None:
    if not label:
        return None
    normalized = label.capitalize()
    try:
        return SEVERITY_LABELS.index(normalized)
    except ValueError:
        return None


def extract_triplet_log_probs(signal: Mapping) -> tuple[list[float], int] | None:
    """Return the three severity log-probs and the gold severity index, if present."""

    meta = signal.get("metadata") or {}
    companion = meta.get("companion_log_probs")
    label_idx = _severity_index(meta.get("severity_label"))
    if not isinstance(companion, Mapping) or label_idx is None:
        return None
    try:
        values = [float(companion[label]) for label in SEVERITY_LABELS]
    except (KeyError, TypeError, ValueError):
        return None
    return values, label_idx


def scaled_true_label_probability(
    log_probs: list[float], label_index: int, temperature: float
) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = _softmax([value / temperature for value in log_probs])
    return clamp_probability(float(scaled[label_index]))


def _negative_log_likelihood(
    rows: Iterable[tuple[list[float], int]], temperature: float
) -> float:
    losses = []
    for log_probs, label_index in rows:
        prob = scaled_true_label_probability(log_probs, label_index, temperature)
        losses.append(-math.log(prob))
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def fit_temperature(
    rows: Iterable[tuple[list[float], int]],
    *,
    min_temperature: float = 0.25,
    max_temperature: float = 20.0,
    grid_size: int = 41,
    refinement_rounds: int = 5,
) -> TemperatureFit | None:
    """Fit a single scalar temperature on stored severity triplets.

    The scorer stores companion log-probs for the three severity labels. Those
    are sufficient for post-hoc temperature scaling over the severity triplet
    without re-running the VLM.
    """

    rows_list = list(rows)
    if not rows_list:
        return None
    if min_temperature <= 0 or max_temperature <= 0:
        raise ValueError("temperature search range must be positive")
    if min_temperature >= max_temperature:
        raise ValueError("min_temperature must be smaller than max_temperature")

    low = math.log(min_temperature)
    high = math.log(max_temperature)
    best_temperature = 1.0
    best_loss = _negative_log_likelihood(rows_list, best_temperature)

    for _ in range(max(1, refinement_rounds)):
        step = (high - low) / max(1, grid_size - 1)
        candidates = [math.exp(low + step * idx) for idx in range(grid_size)]
        losses = [
            _negative_log_likelihood(rows_list, candidate) for candidate in candidates
        ]
        best_idx = min(range(len(candidates)), key=lambda idx: losses[idx])
        best_temperature = candidates[best_idx]
        best_loss = losses[best_idx]

        left_idx = max(0, best_idx - 1)
        right_idx = min(len(candidates) - 1, best_idx + 1)
        low = math.log(candidates[left_idx])
        high = math.log(candidates[right_idx])

    return TemperatureFit(
        temperature=best_temperature,
        nll_before=_negative_log_likelihood(rows_list, 1.0),
        nll_after=best_loss,
        num_examples=len(rows_list),
        hit_boundary=math.isclose(best_temperature, min_temperature)
        or math.isclose(best_temperature, max_temperature),
    )


def calibrate_signal_confidence(
    signal: Mapping,
    temperature: float,
) -> tuple[float, dict[str, object]]:
    """Return calibrated confidence + calibration metadata for one signal."""

    triplet = extract_triplet_log_probs(signal)
    if triplet is None:
        return float(signal.get("confidence", 0.0)), {
            "calibration_status": "unavailable",
        }

    log_probs, label_index = triplet
    calibrated = scaled_true_label_probability(log_probs, label_index, temperature)
    return calibrated, {
        "calibration_status": "applied",
        "calibration_method": "severity_triplet_temperature_scaling",
        "calibration_temperature": temperature,
        "raw_confidence": float(signal.get("confidence", 0.0)),
        "calibrated_confidence": calibrated,
    }


def apply_temperature_to_records(
    records: Iterable[Mapping],
    temperature: float,
) -> list[dict]:
    calibrated_records: list[dict] = []
    for record in records:
        updated = deepcopy(dict(record))
        signals = []
        for signal in updated.get("signals", []):
            calibrated, meta_update = calibrate_signal_confidence(signal, temperature)
            signal = deepcopy(signal)
            signal["confidence"] = calibrated
            metadata = dict(signal.get("metadata") or {})
            metadata.update(meta_update)
            signal["metadata"] = metadata
            signals.append(signal)
        updated["signals"] = signals
        record_meta = dict(updated.get("metadata") or {})
        record_meta["temperature_calibration"] = {
            "temperature": temperature,
            "applied_to_signals": sum(
                1
                for signal in signals
                if (signal.get("metadata") or {}).get("calibration_status") == "applied"
            ),
        }
        updated["metadata"] = record_meta
        calibrated_records.append(updated)
    return calibrated_records


def threshold_group_key(hallucination_type: str | None, severity: int | str | None) -> str:
    return f"{hallucination_type or 'unknown'}|{severity if severity is not None else 'unknown'}"


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]


def build_group_threshold_policy(
    signals: Iterable[Mapping],
    *,
    group_quantile: float = 0.25,
    min_group_count: int = 25,
    shrinkage: float = 50.0,
) -> dict[str, object]:
    """Build tau_{type,severity} with shrinkage toward a global fallback."""

    if not 0.0 <= group_quantile <= 1.0:
        raise ValueError("group_quantile must be between 0 and 1")

    usable: list[dict] = []
    for signal in signals:
        meta = signal.get("metadata") or {}
        if meta.get("is_placeholder") or meta.get("error"):
            continue
        usable.append(
            {
                "confidence": float(signal.get("confidence", 0.0)),
                "group": threshold_group_key(
                    signal.get("hallucination_type"), signal.get("severity")
                ),
            }
        )

    global_values = sorted(item["confidence"] for item in usable)
    global_threshold = quantile(global_values, group_quantile)

    grouped: dict[str, list[float]] = {}
    for item in usable:
        grouped.setdefault(item["group"], []).append(item["confidence"])

    by_group: dict[str, dict[str, float | int | bool]] = {}
    for group, values in sorted(grouped.items()):
        values = sorted(values)
        group_threshold = quantile(values, group_quantile)
        count = len(values)
        weight = count / (count + max(shrinkage, 1e-6))
        blended = (weight * group_threshold) + ((1.0 - weight) * global_threshold)
        by_group[group] = {
            "count": count,
            "group_threshold": group_threshold,
            "threshold": blended,
            "global_fallback": global_threshold,
            "shrinkage_weight": weight,
            "insufficient_support": count < min_group_count,
        }

    return {
        "group_quantile": group_quantile,
        "min_group_count": min_group_count,
        "shrinkage": shrinkage,
        "global_threshold": global_threshold,
        "by_group": by_group,
        "group_count_median": median(
            [item["count"] for item in by_group.values()]
        )
        if by_group
        else 0,
    }


def lookup_group_threshold(
    policy: Mapping[str, object],
    hallucination_type: str | None,
    severity: int | str | None,
    *,
    default: float = 0.0,
) -> float:
    if not policy:
        return default
    by_group = policy.get("by_group")
    if not isinstance(by_group, Mapping):
        return float(policy.get("global_threshold", default))
    group = threshold_group_key(hallucination_type, severity)
    entry = by_group.get(group)
    if isinstance(entry, Mapping) and "threshold" in entry:
        return float(entry["threshold"])
    return float(policy.get("global_threshold", default))
