from __future__ import annotations

import argparse
from pathlib import Path

from fg_pipeline.eval.benchmarks import BENCHMARK_REGISTRY
from fg_pipeline.eval.reporting import build_general_report, build_paper_comparison, write_comparison_bundle
from fg_pipeline.eval.schemas import BenchmarkSpec, MetricArtifact, ModelSpec
from fg_pipeline.eval.utils import (
    discover_stage_paths,
    getenv_openai_judge_model,
    load_model_specs,
    mkdir,
    summarize_stage3,
    summarize_stage4,
)


_STRICT_PAPER_DEFAULT = "mhalubench,mfhallubench,pope_adv"
_SUPPLEMENTAL_DEFAULT = "object_halbench,amber,mmhal_bench,llava_bench_wild,hss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project-owned strict/supplemental evaluation.")
    parser.add_argument("--run-name", required=True, help="Stable output run name under output/eval.")
    parser.add_argument("--models-json", required=True, help="Path to a JSON manifest of ModelSpec rows.")
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark names. Defaults depend on --paper-core / --supplemental.",
    )
    parser.add_argument("--paper-core", action="store_true", help="Run the strict paper-comparison suite.")
    parser.add_argument("--supplemental", action="store_true", help="Run supplemental local/proxy benchmarks.")
    parser.add_argument("--general", action="store_true", help="Run stage-internal general evaluation.")
    parser.add_argument("--output-root", default="output/eval", help="Evaluation output root.")
    parser.add_argument("--image-root-override", default=None, help="Override benchmark image roots.")
    parser.add_argument("--dataset-root-override", default=None, help="Override benchmark dataset roots.")
    parser.add_argument("--skip-missing-datasets", action="store_true", help="Skip missing benchmark assets instead of failing.")
    parser.add_argument("--openai-judge-model", default=None, help="Judge model for MMHal-Bench and LLaVA-Bench supplemental scoring.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def _selected_benchmarks(args: argparse.Namespace) -> list[str]:
    raw = [item.strip() for item in (args.benchmarks or "").split(",") if item.strip()]
    if raw:
        return raw
    selected: list[str] = []
    if args.paper_core:
        selected.extend(_STRICT_PAPER_DEFAULT.split(","))
    if args.supplemental:
        selected.extend(_SUPPLEMENTAL_DEFAULT.split(","))
    if not selected and not args.general:
        selected.extend(_STRICT_PAPER_DEFAULT.split(","))
    deduped: list[str] = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _load_models(path: str) -> list[ModelSpec]:
    return [ModelSpec.from_dict(item) for item in load_model_specs(path)]


def _validate_strict_manifest(models: list[ModelSpec]) -> None:
    if not models:
        return
    max_new_tokens = {model.max_new_tokens for model in models}
    for model in models:
        if model.temperature != 0.0:
            raise SystemExit(
                f"Strict paper comparison requires temperature=0.0, got {model.temperature} for {model.model_id}"
            )
        if model.num_beams != 1:
            raise SystemExit(
                f"Strict paper comparison requires num_beams=1, got {model.num_beams} for {model.model_id}"
            )
        if model.conv_mode != "vicuna_v1":
            raise SystemExit(
                f"Strict paper comparison requires conv_mode='vicuna_v1', got {model.conv_mode!r} for {model.model_id}"
            )
    if len(max_new_tokens) > 1:
        raise SystemExit(
            "Strict paper comparison requires a single shared max_new_tokens across the manifest"
        )


def _benchmark_spec(name: str, args: argparse.Namespace) -> BenchmarkSpec:
    if name == "hss":
        return BenchmarkSpec(name="hss", enabled=True, judge_required=True, split="default")
    try:
        adapter = BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted({*BENCHMARK_REGISTRY.keys(), "hss"}))
        raise SystemExit(f"Unknown benchmark {name!r}; available: {available}") from exc
    return adapter.build_spec(
        dataset_root_override=args.dataset_root_override,
        image_root_override=args.image_root_override,
    )


def _stage_metrics() -> dict[str, dict]:
    discovered = discover_stage_paths()
    stage_metrics: dict[str, dict] = {}
    if "stage3_dir" in discovered:
        stage_metrics["stage3"] = summarize_stage3(discovered["stage3_dir"])
    if "stage4_dir" in discovered:
        stage_metrics["stage4"] = summarize_stage4(discovered["stage4_dir"])
    return stage_metrics


def _run_hss(
    models: list[ModelSpec],
    output_root: Path,
) -> list[MetricArtifact]:
    raise RuntimeError(
        "HSS local judged evaluation is supplemental only and requires a local judge backend implementation"
    )


def _models_for_benchmark(benchmark_name: str, models: list[ModelSpec]) -> list[ModelSpec]:
    if benchmark_name in {"mhalubench", "mfhallubench"} and models:
        return [models[min(1, len(models) - 1)]]
    return models


def main() -> int:
    args = parse_args()
    models = _load_models(args.models_json)
    selected = _selected_benchmarks(args)
    output_root = mkdir(Path(args.output_root) / args.run_name)
    openai_judge_model = args.openai_judge_model or getenv_openai_judge_model()

    if not args.paper_core and not args.supplemental and not args.general:
        args.paper_core = True

    if args.paper_core:
        _validate_strict_manifest(models)

    metric_artifacts: list[MetricArtifact] = []
    availability: dict[str, dict] = {}

    for benchmark_name in selected:
        spec = _benchmark_spec(benchmark_name, args)
        if benchmark_name == "hss":
            try:
                metric_artifacts.extend(_run_hss(models, output_root))
                availability[benchmark_name] = {"status": "ok", "note": "local judged"}
            except Exception as exc:
                if args.skip_missing_datasets or args.supplemental:
                    availability[benchmark_name] = {"status": "skipped", "note": str(exc)}
                    continue
                raise
            continue

        adapter = BENCHMARK_REGISTRY[benchmark_name]
        if spec.judge_required and not openai_judge_model:
            note = "benchmark requires --openai-judge-model or OPENAI_JUDGE_MODEL"
            if args.skip_missing_datasets or args.supplemental:
                availability[benchmark_name] = {"status": "skipped", "note": note}
                continue
            raise SystemExit(f"Benchmark {benchmark_name!r} requires a local judge backend: {note}")

        try:
            if adapter.requires_model:
                for model in _models_for_benchmark(benchmark_name, models):
                    _, metric_artifact, _ = adapter.evaluate(
                        model,
                        spec,
                        run_root=str(output_root),
                        limit=args.limit,
                        openai_judge_model=openai_judge_model,
                    )
                    metric_artifacts.append(metric_artifact)
            else:
                _, metric_artifact, _ = adapter.evaluate(
                    None,
                    spec,
                    run_root=str(output_root),
                    limit=args.limit,
                    openai_judge_model=openai_judge_model,
                )
                metric_artifacts.append(metric_artifact)
            availability[benchmark_name] = {"status": "ok", "note": None}
        except Exception as exc:
            if args.skip_missing_datasets:
                availability[benchmark_name] = {"status": "skipped", "note": str(exc)}
                continue
            raise

    stage_metrics = _stage_metrics() if args.general else {}
    general_report = build_general_report(stage_metrics, metric_artifacts)
    paper_rows = build_paper_comparison(metric_artifacts, models) if (args.paper_core or args.supplemental) else []
    write_comparison_bundle(
        output_root,
        models=models,
        availability=availability,
        paper_rows=paper_rows,
        general_report=general_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
