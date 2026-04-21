"""Stage 3 calibration and threshold policy selection.

This module upgrades the old descriptive calibration report into a usable
selection step:

- fit a scalar temperature on Stage 3's stored severity triplets,
- optionally write a calibrated ``D_det`` copy,
- build group-conditional Stage 4 thresholds ``tau_{type,severity}``,
- summarize the calibrated confidence distribution.

The calibration remains only as trustworthy as its target. When the downstream
Stage 5 verifier is heuristic, threshold guarantees are relative to that
heuristic unless a manually audited subset or stronger verifier is used.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from typing import Iterable, Mapping

from fg_pipeline.confidence.calibration import (
    apply_temperature_to_records,
    build_group_threshold_policy,
    extract_triplet_log_probs,
    fit_temperature,
    quantile,
)
from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl, write_jsonl

_TAU_QUANTILES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _is_real_signal(signal: Mapping) -> bool:
    meta = signal.get("metadata") or {}
    return not meta.get("is_placeholder") and not meta.get("error")


def _clone_records(records: Iterable[Mapping]) -> list[dict]:
    return [deepcopy(dict(record)) for record in records]


def _real_signals(records: Iterable[Mapping]) -> list[dict]:
    signals: list[dict] = []
    for record in records:
        for signal in record.get("signals", []):
            if _is_real_signal(signal):
                signals.append(signal)
    return signals


def calibrate(
    records: Iterable[dict],
    *,
    group_quantile: float = 0.25,
    min_group_count: int = 25,
    shrinkage: float = 50.0,
    min_temperature: float = 0.25,
    max_temperature: float = 20.0,
) -> dict:
    records_list = _clone_records(records)

    total_rows = len(records_list)
    total_signals = sum(len(record.get("signals", [])) for record in records_list)
    placeholder_count = 0
    error_count = 0
    raw_real_signals: list[dict] = []
    fit_rows: list[tuple[list[float], int]] = []
    for record in records_list:
        for signal in record.get("signals", []):
            meta = signal.get("metadata") or {}
            if meta.get("is_placeholder"):
                placeholder_count += 1
                continue
            if meta.get("error"):
                error_count += 1
                continue
            raw_real_signals.append(signal)
            triplet = extract_triplet_log_probs(signal)
            if triplet is not None:
                fit_rows.append(triplet)

    temperature_fit = fit_temperature(
        fit_rows,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    if temperature_fit is None:
        calibrated_records = records_list
        applied_temperature = 1.0
        temperature_status = "unavailable"
    else:
        calibrated_records = apply_temperature_to_records(
            records_list, temperature_fit.temperature
        )
        applied_temperature = temperature_fit.temperature
        temperature_status = "applied"

    calibrated_real_signals = _real_signals(calibrated_records)
    raw_values = [float(signal.get("confidence", 0.0)) for signal in raw_real_signals]
    calibrated_values = [
        float(signal.get("confidence", 0.0)) for signal in calibrated_real_signals
    ]

    per_type: dict[str, list[float]] = defaultdict(list)
    per_severity: dict[str, list[float]] = defaultdict(list)
    for signal in calibrated_real_signals:
        per_type[signal.get("hallucination_type") or "unknown"].append(
            float(signal.get("confidence", 0.0))
        )
        per_severity[str(signal.get("severity"))].append(
            float(signal.get("confidence", 0.0))
        )

    sorted_values = sorted(calibrated_values)
    deciles = {
        f"p{q * 10:02.0f}": quantile(sorted_values, q / 10) for q in range(0, 11)
    }
    suggested_tau = {
        f"q{int(q * 100):02d}": quantile(sorted_values, q) for q in _TAU_QUANTILES
    }
    group_policy = build_group_threshold_policy(
        calibrated_real_signals,
        group_quantile=group_quantile,
        min_group_count=min_group_count,
        shrinkage=shrinkage,
    )

    report = {
        "totals": {
            "rows": total_rows,
            "signals": total_signals,
            "real_signals": len(raw_real_signals),
            "placeholder_signals": placeholder_count,
            "error_signals": error_count,
        },
        "temperature_scaling": {
            "status": temperature_status,
            "temperature": applied_temperature,
            "num_examples": 0 if temperature_fit is None else temperature_fit.num_examples,
            "nll_before": None if temperature_fit is None else temperature_fit.nll_before,
            "nll_after": None if temperature_fit is None else temperature_fit.nll_after,
            "hit_search_boundary": (
                False if temperature_fit is None else temperature_fit.hit_boundary
            ),
        },
        "overall_raw": _stats(raw_values),
        "overall": _stats(calibrated_values),
        "deciles": deciles,
        "suggested_tau": suggested_tau,
        "group_threshold_policy": group_policy,
        "per_type": {key: _stats(values) for key, values in sorted(per_type.items())},
        "per_severity": {
            key: _stats(values) for key, values in sorted(per_severity.items())
        },
    }
    return report


def format_report(report: dict) -> str:
    lines: list[str] = []
    totals = report["totals"]
    lines.append(f"Rows: {totals['rows']}  Signals: {totals['signals']}")
    lines.append(
        f"  real={totals['real_signals']} placeholder={totals['placeholder_signals']} "
        f"error={totals['error_signals']}"
    )

    if totals["real_signals"] == 0:
        lines.append("")
        lines.append("WARNING: no real confidence values available.")
        if totals["placeholder_signals"]:
            lines.append(
                "  All signals come from the bootstrap scorer "
                "(metadata.is_placeholder=True)."
            )
            lines.append(
                "  Re-run Stage 3 with SCORER=log_prob before calibrating tau / tau_c."
            )
        return "\n".join(lines)

    temperature = report["temperature_scaling"]
    lines.append(
        "Temperature scaling: "
        f"status={temperature['status']} "
        f"T={temperature['temperature']:.4f} "
        f"fit_examples={temperature['num_examples']}"
    )
    if temperature.get("hit_search_boundary"):
        lines.append(
            "  NOTE: fitted temperature hit the search boundary; widen the range if "
            "you want a less clipped post-hoc fit."
        )
    if temperature["nll_before"] is not None:
        lines.append(
            "  NLL: "
            f"before={temperature['nll_before']:.6f} "
            f"after={temperature['nll_after']:.6f}"
        )

    raw = report["overall_raw"]
    lines.append(
        f"Overall raw confidence:       n={raw['count']} mean={raw['mean']:.4f} "
        f"median={raw['median']:.4f} min={raw['min']:.4f} max={raw['max']:.4f}"
    )

    overall = report["overall"]
    lines.append(
        f"Overall calibrated confidence: n={overall['count']} "
        f"mean={overall['mean']:.4f} median={overall['median']:.4f} "
        f"min={overall['min']:.4f} max={overall['max']:.4f}"
    )

    lines.append("Deciles (p00..p100):")
    for key, value in report["deciles"].items():
        lines.append(f"  {key}: {value:.4f}")

    lines.append("Suggested tau / tau_c candidates:")
    for key, value in report["suggested_tau"].items():
        lines.append(f"  {key}: {value:.4f}")

    policy = report["group_threshold_policy"]
    lines.append(
        "Group-conditional tau_{type,severity}: "
        f"q={policy['group_quantile']:.2f} "
        f"global_fallback={policy['global_threshold']:.4f} "
        f"groups={len(policy['by_group'])}"
    )
    for group, entry in sorted(policy["by_group"].items())[:12]:
        lines.append(
            f"  {group}: tau={entry['threshold']:.4f} "
            f"count={entry['count']} "
            f"group_q={entry['group_threshold']:.4f} "
            f"weight={entry['shrinkage_weight']:.3f}"
        )

    lines.append("Per hallucination type:")
    for key, stats in sorted(report["per_type"].items()):
        lines.append(
            f"  {key}: n={stats['count']} mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}"
        )

    lines.append("Per severity:")
    for key, stats in sorted(report["per_severity"].items()):
        lines.append(
            f"  HS={key}: n={stats['count']} mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Stage 3 temperature scaling, summarize calibrated confidence, "
            "and emit a Stage 4 threshold policy."
        )
    )
    parser.add_argument("--input", required=True, help="Path to D_det.jsonl")
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write the JSON calibration report.",
    )
    parser.add_argument(
        "--output-calibrated",
        default=None,
        help="Optional path to write a calibrated D_det copy.",
    )
    parser.add_argument(
        "--group-quantile",
        type=float,
        default=0.25,
        help="Quantile used for group-conditional Stage 4 thresholds.",
    )
    parser.add_argument(
        "--group-min-count",
        type=int,
        default=25,
        help="Support level below which groups are flagged as low-support.",
    )
    parser.add_argument(
        "--group-shrinkage",
        type=float,
        default=50.0,
        help="Shrinkage strength toward the global fallback threshold.",
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=0.25,
        help="Minimum temperature searched during calibration.",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=20.0,
        help="Maximum temperature searched during calibration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(read_jsonl(args.input))
    report = calibrate(
        records,
        group_quantile=args.group_quantile,
        min_group_count=args.group_min_count,
        shrinkage=args.group_shrinkage,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )
    print(format_report(report))

    if args.report:
        out = ensure_parent_dir(args.report)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote calibration report to {args.report}")

    if args.output_calibrated:
        temperature = report["temperature_scaling"]["temperature"]
        if report["temperature_scaling"]["status"] == "applied":
            calibrated_records = apply_temperature_to_records(records, temperature)
        else:
            calibrated_records = records
        write_jsonl(args.output_calibrated, calibrated_records)
        print(f"Wrote calibrated detection records to {args.output_calibrated}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
