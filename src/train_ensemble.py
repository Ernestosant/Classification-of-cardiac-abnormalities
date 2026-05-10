from __future__ import annotations

import json

import numpy as np

from .artifacts import ensure_dirs, fit_or_load_scaler, get_or_create_split_indices
from .config import ENSEMBLE_CONFIG_PATH, REPORTS_DIR, SEED
from .data import load_ecg5000
from .inception_adapter import load_inception_predictor
from .inference import (
    anomaly_confidence,
    apply_isolation_adjustment,
    load_isolation_artifacts,
    load_xgboost_model,
    normalize_proba,
)
from .metrics import multiclass_metrics, save_json, save_multiclass_confusion_matrix


def main() -> None:
    ensure_dirs()
    X_full, y_full = load_ecg5000("train")
    split = get_or_create_split_indices(y_full)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    scaler = fit_or_load_scaler(X_full, train_idx)
    X_val = scaler.transform(X_full[val_idx])
    y_val = y_full[val_idx]

    xgb_model = load_xgboost_model()
    xgb_proba = xgb_model.predict_proba(X_val)

    inception_predictor, inception_status = load_inception_predictor()
    inception_proba = inception_predictor(X_val) if inception_predictor is not None else None

    try:
        if_model, if_config = load_isolation_artifacts()
        decisions = if_model.decision_function(X_val)
        anomaly_conf = anomaly_confidence(decisions, if_config["threshold"], if_config["scale"])
        gamma_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        isolation_status = "loaded"
    except FileNotFoundError as exc:
        anomaly_conf = np.zeros(len(y_val), dtype=float)
        gamma_values = [0.0]
        isolation_status = (
            f"missing Isolation Forest artifacts: {exc}. "
            "Skipped anomaly adjustment; run python -m src.train_isolation_forest to enable it."
        )

    supervised_candidates = []
    if inception_proba is None:
        supervised_candidates.append(({"xgboost": 1.0}, xgb_proba))
    else:
        for xgb_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            inc_weight = 1.0 - xgb_weight
            proba = normalize_proba(xgb_weight * xgb_proba + inc_weight * inception_proba)
            supervised_candidates.append(({"xgboost": xgb_weight, "inception": inc_weight}, proba))

    best = {
        "macro_f1": -1.0,
        "balanced_accuracy": -1.0,
        "isolation_gamma": None,
        "supervised_sources": None,
        "pred": None,
    }
    for sources, supervised_proba in supervised_candidates:
        for gamma in gamma_values:
            ensemble_proba = apply_isolation_adjustment(supervised_proba, anomaly_conf, gamma)
            pred = ensemble_proba.argmax(axis=1) + 1
            metrics = multiclass_metrics(y_val, pred)
            if (metrics["macro_f1"] > best["macro_f1"]) or (
                np.isclose(metrics["macro_f1"], best["macro_f1"])
                and metrics["balanced_accuracy"] > best["balanced_accuracy"]
            ):
                best.update(
                    {
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "isolation_gamma": gamma,
                        "supervised_sources": sources,
                        "pred": pred,
                    }
                )

    selected_sources = {
        name: weight for name, weight in best["supervised_sources"].items() if float(weight) > 0.0
    }
    config = {
        "seed": SEED,
        "selection_metric": "validation macro_f1, tie-breaker balanced_accuracy",
        "supervised_sources": selected_sources,
        "candidate_supervised_sources": best["supervised_sources"],
        "isolation_gamma": best["isolation_gamma"],
        "isolation_status": isolation_status,
        "inception_status": inception_status,
        "test_set_used_for_selection": False,
    }
    with ENSEMBLE_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics = multiclass_metrics(y_val, best["pred"])
    metrics["ensemble_config"] = config
    save_json(metrics, REPORTS_DIR / "metrics_ensemble_validation.json")
    save_multiclass_confusion_matrix(
        y_val,
        best["pred"],
        REPORTS_DIR / "confusion_matrix_ensemble_validation.png",
        "Ensemble validation confusion matrix",
    )
    print(f"Saved ensemble config to {ENSEMBLE_CONFIG_PATH}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Inception status: {inception_status}")
    print(f"Isolation Forest status: {isolation_status}")


if __name__ == "__main__":
    main()
