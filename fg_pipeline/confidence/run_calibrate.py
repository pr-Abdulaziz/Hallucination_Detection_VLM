"""Batch 2c — Confidence distribution + threshold calibration for D_det.jsonl.

Reads the per-signal confidence values from a D_det.jsonl file, summarizes
the distribution (overall, per-type, per-severity), and proposes candidate
threshold values for Stage 4 (τ, signal-level drop) and Stage 5 (τ_c,
verify-gate). This is a read-only diagnostic — it does not rewrite D_det.

Placeholder signals (bootstrap scorer, ``metadata.is_placeholder == True``)
and error signals (``metadata.error`` present) are excluded from the
distribution. Running this on an all-placeholder file prints a warning and
exits without threshold suggestions.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Iterable

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl

# τ candidates: low-quantile cutoffs that would drop the weakest signals.
_TAU_QUANTILES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def calibrate(records: Iterable[dict]) -> dict:
    real_confidences: list[float] = []
    placeholder_count = 0
    error_count = 0
    per_type: dict[str, list[float]] = defaultdict(list)
    per_severity: dict[str, list[float]] = defaultdict(list)

    total_rows = 0
    total_signals = 0
    for record in records:
        total_rows += 1
        for signal in record.get("signals", []):
            total_signals += 1
            meta = signal.get("metadata") or {}
            if meta.get("is_placeholder"):
                placeholder_count += 1
                continue
            if meta.get("error"):
                error_count += 1
                continue
            conf = float(signal.get("confidence", 0.0))
            real_confidences.append(conf)
            per_type[signal.get("hallucination_type") or "unknown"].append(conf)
            per_severity[str(signal.get("severity"))].append(conf)

    sorted_conf = sorted(real_confidences)
    deciles = {
        f"p{q * 10:02.0f}": _quantile(sorted_conf, q / 10) for q in range(0, 11)
    }
    suggested_tau = {
        f"q{int(q * 100):02d}": _quantile(sorted_conf, q) for q in _TAU_QUANTILES
    }

    return {
        "totals": {
            "rows": total_rows,
            "signals": total_signals,
            "real_signals": len(real_confidences),
            "placeholder_signals": placeholder_count,
            "error_signals": error_count,
        },
        "overall": _stats(real_confidences),
        "deciles": deciles,
        "suggested_tau": suggested_tau,
        "per_type": {k: _stats(v) for k, v in per_type.items()},
        "per_severity": {k: _stats(v) for k, v in per_severity.items()},
    }


def format_report(report: dict) -> str:
    lines: list[str] = []
    t = report["totals"]
    lines.append(f"Rows: {t['rows']}  Signals: {t['signals']}")
    lines.append(
        f"  real={t['real_signals']} placeholder={t['placeholder_signals']} "
        f"error={t['error_signals']}"
    )

    if t["real_signals"] == 0:
        lines.append("")
        lines.append("WARNING: no real confidence values available.")
        if t["placeholder_signals"]:
            lines.append(
                "  All signals come from the bootstrap scorer "
                "(metadata.is_placeholder=True)."
            )
            lines.append(
                "  Re-run Stage 3 with SCORER=log_prob before calibrating tau / tau_c."
            )
        return "\n".join(lines)

    o = report["overall"]
    lines.append(
        f"Overall confidence:  n={o['count']} mean={o['mean']:.4f} "
        f"median={o['median']:.4f} min={o['min']:.4f} max={o['max']:.4f}"
    )

    lines.append("Deciles (p00..p100):")
    for k, v in report["deciles"].items():
        lines.append(f"  {k}: {v:.4f}")

    lines.append("Suggested tau / tau_c candidates (drop signals with c^j < value):")
    for k, v in report["suggested_tau"].items():
        lines.append(f"  {k}: {v:.4f}")

    lines.append("Per hallucination type:")
    for k, s in sorted(report["per_type"].items()):
        lines.append(
            f"  {k}: n={s['count']} mean={s['mean']:.4f} "
            f"median={s['median']:.4f} min={s['min']:.4f} max={s['max']:.4f}"
        )

    lines.append("Per severity:")
    for sev in ("1", "2", "3"):
        s = report["per_severity"].get(sev)
        if not s or s["count"] == 0:
            continue
        lines.append(
            f"  HS={sev}: n={s['count']} mean={s['mean']:.4f} "
            f"median={s['median']:.4f} min={s['min']:.4f} max={s['max']:.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-signal confidence in a D_det.jsonl file and propose "
            "candidate τ / τ_c thresholds for Stages 4 and 5."
        )
    )
    parser.add_argument("--input", required=True, help="Path to D_det.jsonl")
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = calibrate(read_jsonl(args.input))
    print(format_report(report))
    if args.report:
        out = ensure_parent_dir(args.report)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote calibration report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
