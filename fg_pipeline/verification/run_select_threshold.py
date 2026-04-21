from __future__ import annotations

import argparse
import json

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl
from fg_pipeline.verification.backends import get_backend
from fg_pipeline.verification.threshold_selection import (
    build_pair_candidates,
    select_crc_threshold,
    select_cv_crc_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Stage 5 tau_c with CRC/CV-CRC using the verifier as the "
            "pair-quality target."
        )
    )
    parser.add_argument("--input", required=True, help="Path to D_rewrite.jsonl")
    parser.add_argument(
        "--output-report",
        required=True,
        help="Path to write the tau_c selection report JSON.",
    )
    parser.add_argument(
        "--backend",
        default="heuristic",
        help="Verification backend used to define the pair-quality target.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Target upper-bound risk for retained pairs.",
    )
    parser.add_argument(
        "--method",
        choices=("crc", "cv_crc"),
        default="cv_crc",
        help="Threshold-selection method.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for CV-CRC.",
    )
    parser.add_argument(
        "--min-accepted",
        type=int,
        default=100,
        help="Minimum accepted rows required for a threshold to be considered valid.",
    )
    parser.add_argument(
        "--min-rewrite-chars",
        type=int,
        default=8,
        help="Minimum rewrite length forwarded to the verification backend.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = get_backend(args.backend, min_rewrite_chars=args.min_rewrite_chars)
    candidates = build_pair_candidates(
        read_jsonl(args.input),
        backend,
        limit=args.limit,
    )
    if args.method == "cv_crc":
        report = select_cv_crc_threshold(
            candidates,
            alpha=args.alpha,
            num_folds=args.folds,
            min_accepted=args.min_accepted,
        )
    else:
        full = select_crc_threshold(
            candidates,
            alpha=args.alpha,
            min_accepted=args.min_accepted,
        )
        report = {
            "method": "crc",
            "selected_tau_c": full["threshold"],
            "full_data": full,
            "folds": [],
        }

    report.update(
        {
            "input": args.input,
            "num_candidates": len(candidates),
            "backend": backend.name,
            "alpha": args.alpha,
            "min_accepted": args.min_accepted,
            "consideration": (
                "Risk guarantees are only as good as the verification target. "
                "With the heuristic verifier, tau_c is controlled relative to "
                "that heuristic rather than absolute ground truth."
            ),
        }
    )
    out = ensure_parent_dir(args.output_report)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"Selected tau_c={report['selected_tau_c']:.6f} "
        f"via {report['method']} from {len(candidates)} candidate rows"
    )
    print(f"Wrote tau_c report to {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
