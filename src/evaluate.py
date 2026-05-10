from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from .artifacts import ensure_dirs, get_or_create_split_indices
from .config import (
    CLASS_NAMES,
    ENSEMBLE_CONFIG_PATH,
    IFOREST_CONFIG_PATH,
    REPORTS_DIR,
    SCALER_PATH,
    XGBOOST_MODEL_PATH,
)
from .data import load_ecg5000
from .inception_adapter import load_inception_predictor
from .inference import (
    load_isolation_artifacts,
    load_scaler,
    load_xgboost_model,
    predict_ensemble_proba,
)
from .metrics import (
    binary_anomaly_metrics,
    load_json,
    multiclass_metrics,
    save_binary_confusion_matrix,
    save_json,
    save_multiclass_confusion_matrix,
)


def main() -> None:
    ensure_dirs()
    X_train_full, y_train_full = load_ecg5000("train")
    split = get_or_create_split_indices(y_train_full)
    X_test, y_test = load_ecg5000("test")
    scaler = load_scaler(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    results: dict[str, dict] = {
        "metadata": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "train_samples": int(len(y_train_full)),
            "validation_samples": int(len(split["val_idx"])),
            "test_samples": int(len(y_test)),
            "test_used_for_model_selection": False,
            "scaler_fit_scope": "inner training split only",
        }
    }

    xgb_proba = None
    if XGBOOST_MODEL_PATH.exists():
        xgb_model = load_xgboost_model()
        xgb_proba = xgb_model.predict_proba(X_test_scaled)
        pred = xgb_proba.argmax(axis=1) + 1
        metrics = multiclass_metrics(y_test, pred)
        results["xgboost_test"] = metrics
        save_json(metrics, REPORTS_DIR / "metrics_xgboost_test.json")
        save_multiclass_confusion_matrix(
            y_test,
            pred,
            REPORTS_DIR / "confusion_matrix_xgboost_test.png",
            "XGBoost test confusion matrix",
        )
    else:
        results["xgboost_test"] = {"status": "missing model"}

    if_decisions = None
    if_config = None
    try:
        if_model, if_config = load_isolation_artifacts()
        if_decisions = if_model.decision_function(X_test_scaled)
        pred_anomaly = if_decisions <= if_config["threshold"]
        metrics = binary_anomaly_metrics(y_test, pred_anomaly)
        metrics["config"] = if_config
        results["isolation_forest_test"] = metrics
        save_json(metrics, REPORTS_DIR / "metrics_isolation_forest_test.json")
        save_binary_confusion_matrix(
            y_test,
            pred_anomaly,
            REPORTS_DIR / "confusion_matrix_isolation_forest_test.png",
            "Isolation Forest test confusion matrix",
        )
    except FileNotFoundError as exc:
        results["isolation_forest_test"] = {"status": str(exc)}

    inception_proba = None
    inception_predictor, inception_status = load_inception_predictor()
    if inception_predictor is None:
        results["inception_test"] = {"status": inception_status}
    else:
        inception_proba = inception_predictor(X_test_scaled)
        pred = inception_proba.argmax(axis=1) + 1
        metrics = multiclass_metrics(y_test, pred)
        results["inception_test"] = metrics
        save_json(metrics, REPORTS_DIR / "metrics_inception_test.json")
        save_multiclass_confusion_matrix(
            y_test,
            pred,
            REPORTS_DIR / "confusion_matrix_inception_test.png",
            "InceptionTime test confusion matrix",
        )

    if ENSEMBLE_CONFIG_PATH.exists():
        try:
            proba, details = predict_ensemble_proba(
                X_test_scaled,
                xgb_proba=xgb_proba,
                inception_proba=inception_proba,
                if_decision=if_decisions,
                if_config=if_config,
            )
            pred = proba.argmax(axis=1) + 1
            metrics = multiclass_metrics(y_test, pred)
            metrics["prediction_details"] = details
            results["ensemble_test"] = metrics
            save_json(metrics, REPORTS_DIR / "metrics_ensemble_test.json")
            save_multiclass_confusion_matrix(
                y_test,
                pred,
                REPORTS_DIR / "confusion_matrix_ensemble_test.png",
                "Entropy-weighted ensemble test confusion matrix",
            )
        except (FileNotFoundError, RuntimeError) as exc:
            results["ensemble_test"] = {"status": str(exc)}
    else:
        results["ensemble_test"] = {"status": "missing ensemble config"}

    save_json(results, REPORTS_DIR / "metrics_all_test.json")
    write_markdown_report(results)
    print(f"Saved report to {REPORTS_DIR / 'model_results.md'}")


def write_markdown_report(results: dict) -> None:
    train_counts = _counts(load_ecg5000("train")[1])
    test_counts = _counts(load_ecg5000("test")[1])
    val_metrics = _load_validation_metrics()

    lines = [
        "# ECG5000 Model Results",
        "",
        "## Scope",
        "",
        "This report documents a reproducible ECG5000 experiment for five-class heartbeat classification. "
        "It is not a clinically validated diagnostic system.",
        "",
        "## Anti-Leakage Controls",
        "",
        "- The official `ECG5000_test.csv` split is reserved for final evaluation.",
        "- The scaler is fitted only on the inner training split created from `ECG5000_train.csv`.",
        "- Validation is used for early stopping, Isolation Forest threshold selection, and ensemble weight selection.",
        "- Test labels are not used for preprocessing, model selection, threshold selection, or ensemble configuration.",
        "- Isolation Forest is trained only on class 1 normal samples from the inner training split.",
        "",
        "## Data Distribution",
        "",
        "| Split | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _count_row("Official train", train_counts),
        _count_row("Official test", test_counts),
        "",
        "## Validation Metrics Used For Selection",
        "",
        _metrics_table(val_metrics),
        "",
        "## Final Test Metrics",
        "",
        _metrics_table(results),
        "",
        "## Ensemble Selection",
        "",
        _ensemble_section(),
        "",
        "## Ensemble Diagnostics",
        "",
        _ensemble_diagnostics_section(results),
        "",
        "## InceptionTime Training Notes",
        "",
        _inception_section(val_metrics),
        "",
        "## Isolation Forest Notes",
        "",
        _isolation_section(results),
        "",
        "## Per-Class Recall On Test",
        "",
        _per_class_recall_table(results),
        "",
        "## Limitations",
        "",
        "- ECG5000 is small for classes 3 and 5, so minority-class metrics can move substantially with a few examples.",
        "- Isolation Forest is anomaly-oriented and does not identify the exact abnormal subtype by itself.",
        "- The formula ensemble requires `models/inception_cpu.pkl` and fastai/tsai for CPU inference.",
        "- Metrics should be interpreted as project evidence, not clinical performance claims.",
        "",
        "## Reproducibility",
        "",
        "```powershell",
        "python -m src.train_xgboost",
        "python -m src.train_isolation_forest",
        "python -m src.train_ensemble",
        "python -m src.evaluate",
        "python app.py",
        "```",
        "",
    ]
    (REPORTS_DIR / "model_results.md").write_text("\n".join(lines), encoding="utf-8")


def _load_validation_metrics() -> dict:
    metrics = {}
    for name, path in {
        "xgboost_validation": REPORTS_DIR / "metrics_xgboost_validation.json",
        "isolation_forest_validation": REPORTS_DIR / "metrics_isolation_forest_validation.json",
        "inception_validation": REPORTS_DIR / "metrics_inception_validation.json",
        "ensemble_validation": REPORTS_DIR / "metrics_ensemble_validation.json",
    }.items():
        if path.exists():
            metrics[name] = load_json(path)
        else:
            metrics[name] = {"status": "not available"}
    return metrics


def _counts(y: np.ndarray) -> dict[int, int]:
    return {label: int((y == label).sum()) for label in CLASS_NAMES}


def _count_row(name: str, counts: dict[int, int]) -> str:
    return (
        f"| {name} | {counts[1]} | {counts[2]} | {counts[3]} | "
        f"{counts[4]} | {counts[5]} | {sum(counts.values())} |"
    )


def _metrics_table(results: dict) -> str:
    rows = ["| Model | Accuracy | Macro-F1 | Balanced accuracy | Status |", "|---|---:|---:|---:|---|"]
    for key, value in results.items():
        if key == "metadata":
            continue
        label = key.replace("_", " ")
        if "accuracy" in value and "macro_f1" in value:
            rows.append(
                f"| {label} | {value['accuracy']:.4f} | {value['macro_f1']:.4f} | "
                f"{value['balanced_accuracy']:.4f} | ok |"
            )
        else:
            rows.append(f"| {label} |  |  |  | {value.get('status', 'not available')} |")
    return "\n".join(rows)


def _isolation_section(results: dict) -> str:
    value = results.get("isolation_forest_test", {})
    if "anomaly_recall" not in value:
        return value.get("status", "Isolation Forest metrics are not available.")
    return (
        f"- Anomaly recall: `{value['anomaly_recall']:.4f}`\n"
        f"- Normal specificity: `{value['normal_specificity']:.4f}`\n"
        f"- Binary macro-F1: `{value['macro_f1']:.4f}`"
    )


def _ensemble_section() -> str:
    if not ENSEMBLE_CONFIG_PATH.exists():
        return "Ensemble configuration is not available."
    config = load_json(ENSEMBLE_CONFIG_PATH)
    if config.get("ensemble_type") == "entropy_weighted_three_model":
        weights = config.get("base_weights", {})
        return (
            f"- Ensemble type: `{config.get('ensemble_type')}`\n"
            f"- Selection metric: `{config.get('selection_metric', 'not recorded')}`\n"
            f"- Base weights: XGBoost `{weights.get('xgboost', 0.0):.2f}`, "
            f"InceptionTime `{weights.get('inception', 0.0):.2f}`, "
            f"Isolation Forest `{weights.get('isolation_forest', 0.0):.2f}`\n"
            f"- Entropy epsilon: `{config.get('epsilon', 'not recorded')}`\n"
            f"- Contribution policy: `{config.get('model_contribution_policy', 'not recorded')}`\n"
            "- Test set used for ensemble selection: `false`"
        )

    sources = config.get("supervised_sources", {})
    xgb_weight = sources.get("xgboost", 0.0)
    inception_weight = sources.get("inception", 0.0)
    gamma = config.get("isolation_gamma", 0.0)
    return (
        f"- Selection metric: `{config.get('selection_metric', 'not recorded')}`\n"
        f"- Supervised weights: XGBoost `{xgb_weight:.2f}`, InceptionTime `{inception_weight:.2f}`\n"
        f"- Isolation Forest anomaly adjustment gamma: `{gamma:.2f}`\n"
        f"- InceptionTime artifact status during selection: `{config.get('inception_status', 'unknown')}`\n"
        "- Test set used for ensemble selection: `false`"
    )


def _ensemble_diagnostics_section(results: dict) -> str:
    details = results.get("ensemble_test", {}).get("prediction_details", {})
    base = details.get("base_weights", {})
    entropy = details.get("mean_entropy", {})
    dynamic = details.get("mean_dynamic_weights", {})
    if not (base and entropy and dynamic):
        return "Ensemble diagnostics are not available."
    rows = [
        "| Diagnostic | XGBoost | InceptionTime | Isolation Forest |",
        "|---|---:|---:|---:|",
        (
            f"| Base weight | {base.get('xgboost', 0.0):.4f} | "
            f"{base.get('inception', 0.0):.4f} | {base.get('isolation_forest', 0.0):.4f} |"
        ),
        (
            f"| Mean test entropy | {entropy.get('xgboost', 0.0):.4f} | "
            f"{entropy.get('inception', 0.0):.4f} | {entropy.get('isolation_forest', 0.0):.4f} |"
        ),
        (
            f"| Mean test dynamic weight | {dynamic.get('xgboost', 0.0):.4f} | "
            f"{dynamic.get('inception', 0.0):.4f} | {dynamic.get('isolation_forest', 0.0):.4f} |"
        ),
    ]
    return "\n".join(rows)


def _inception_section(val_metrics: dict) -> str:
    value = val_metrics.get("inception_validation", {})
    metadata = value.get("metadata", {})
    if not metadata:
        return value.get("status", "InceptionTime validation metadata is not available.")
    return (
        f"- Runtime device: `{metadata.get('device', 'not recorded')}`\n"
        f"- Epochs requested: `{metadata.get('epochs_requested', 'not recorded')}`\n"
        f"- Early stopping patience: `{metadata.get('early_stopping_patience', 'not recorded')}`\n"
        f"- Batch size: `{metadata.get('batch_size', 'not recorded')}`\n"
        f"- Class-weighted loss: `{str(metadata.get('class_weighted_loss', 'not recorded')).lower()}`\n"
        f"- Scaler fit scope: `{metadata.get('scaler_fit_scope', 'not recorded')}`\n"
        f"- Test set used during training: `{str(metadata.get('test_set_used', 'not recorded')).lower()}`"
    )


def _per_class_recall_table(results: dict) -> str:
    rows = ["| Model | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in results.items():
        if "classification_report" not in value:
            continue
        report = value["classification_report"]
        recalls = []
        for class_name in CLASS_NAMES.values():
            recalls.append(report.get(class_name, {}).get("recall"))
        if all(r is None for r in recalls):
            continue
        formatted = ["" if r is None else f"{r:.4f}" for r in recalls]
        rows.append(f"| {key.replace('_', ' ')} | " + " | ".join(formatted) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
