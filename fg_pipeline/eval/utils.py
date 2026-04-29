from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    return REPO_ROOT


def dump_json(path: str | Path, payload: Any) -> Path:
    out = ensure_parent_dir(path)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_specs(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("models manifest must be a JSON list")
    return [dict(item) for item in payload]


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def quantize_float(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def ece_score(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    bucket_totals = [0] * bins
    bucket_conf = [0.0] * bins
    bucket_acc = [0.0] * bins
    for label, prob in zip(labels, probabilities):
        idx = min(bins - 1, max(0, int(prob * bins)))
        bucket_totals[idx] += 1
        bucket_conf[idx] += prob
        bucket_acc[idx] += float(label)
    total = len(labels)
    if total == 0:
        return None
    error = 0.0
    for count, conf_sum, acc_sum in zip(bucket_totals, bucket_conf, bucket_acc):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        error += (count / total) * abs(avg_conf - avg_acc)
    return float(error)


def binary_classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float] | None = None,
) -> dict[str, float | None]:
    total = len(labels)
    if total == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "nll": None,
            "ece": None,
        }

    tp = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 0)
    tn = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = safe_div(tp + tn, total)

    auroc = None
    auprc = None
    brier = None
    nll = None
    ece = None
    if scores is not None and len(set(labels)) > 1:
        clipped = [min(1.0 - 1e-6, max(1e-6, float(score))) for score in scores]
        auroc = float(roc_auc_score(labels, clipped))
        auprc = float(average_precision_score(labels, clipped))
        brier = float(brier_score_loss(labels, clipped))
        nll = float(log_loss(labels, clipped))
        ece = ece_score(labels, clipped)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
    }


def macro_f1_from_confusion(confusion: Mapping[str, Mapping[str, int]]) -> float | None:
    labels = sorted({*confusion.keys(), *[k for row in confusion.values() for k in row]})
    if not labels:
        return None
    f1s: list[float] = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in labels if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision is None or recall is None or (precision + recall) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len((text or "").split())


def default_dataset_root(dataset_root_override: str | None, name: str) -> str:
    if dataset_root_override:
        return str(Path(dataset_root_override) / name)
    return str(repo_root() / "playground" / "data" / "eval" / name)


def resolve_existing(*candidates: str | Path | None) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def discover_stage_paths() -> dict[str, str]:
    """Locate optional artifacts the evaluator can summarize.

    Discovers current project-pipeline artifacts when present.
    """

    output_root = repo_root() / "output"
    candidates = {
        "stage3_dir": [
            output_root / "fghd" / "stage3",
        ],
        "stage4_dir": [
            output_root / "fghd" / "stage5_llava_margin",
            output_root / "fghd" / "stage4_llava",
            output_root / "hsa_dpo_llava",
            output_root / "fghd" / "adaptive_dpo",
        ],
        "preference_pairs": [
            output_root / "fghd" / "stage4" / "final_preference_pairs.jsonl",
            output_root / "fghd" / "stage3" / "preference_pairs.jsonl",
            repo_root() / "hsa_dpo" / "data" / "hsa_dpo_preference_llava1dot5.jsonl",
        ],
    }
    discovered: dict[str, str] = {}
    for key, values in candidates.items():
        path = resolve_existing(*values)
        if path:
            discovered[key] = path
    return discovered


def summarize_stage3(stage3_dir: str | Path) -> dict[str, Any]:
    stage3_path = Path(stage3_dir)
    stats_path = resolve_existing(stage3_path / "stats.json")
    result: dict[str, Any] = {"stage3_dir": str(stage3_path)}
    if not stats_path:
        return result
    payload = load_json(stats_path)
    for key in (
        "total_input_rows",
        "total_rows",
        "vote_rows_processed",
        "preference_pairs_emitted",
        "accepted_rows",
        "rejected_rows",
        "dropped_rows",
        "backend",
        "api_judge",
        "decision_rule",
        "prompt_mode",
        "gemini_model",
        "openai_model",
        "vote_count",
        "approvals_required",
        "workers",
    ):
        if key in payload:
            result[key] = payload[key]
    return result


def summarize_stage4(stage4_dir: str | Path) -> dict[str, Any]:
    stage4_path = Path(stage4_dir)
    trainer_state = resolve_existing(
        stage4_path / "trainer_state.json",
        *stage4_path.rglob("trainer_state.json"),
    )
    result: dict[str, Any] = {"stage4_dir": str(stage4_path)}
    if not trainer_state:
        return result
    payload = load_json(trainer_state)
    log_history = payload.get("log_history", []) or []
    train_logs = [entry for entry in log_history if "loss" in entry]
    eval_logs = [entry for entry in log_history if any(key.startswith("eval_") for key in entry)]
    if train_logs:
        result["final_train_loss"] = train_logs[-1].get("loss")
    if eval_logs:
        latest = eval_logs[-1]
        for key in ("eval_rewards/accuracies", "eval_rewards/margins"):
            if key in latest:
                result[key] = latest[key]
    return result


def summarize_stage6(stage6_dir: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for older callers."""
    return summarize_stage4(stage6_dir)


def getenv_openai_judge_model(default: str | None = None) -> str | None:
    return os.environ.get("OPENAI_JUDGE_MODEL", default)
