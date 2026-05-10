from __future__ import annotations

import json

import numpy as np

from .artifacts import ensure_dirs, fit_or_load_scaler, get_or_create_split_indices
from .config import ENSEMBLE_CONFIG_PATH, REPORTS_DIR, SEED
from .data import load_ecg5000
from .ensemble_formula import (
    MODEL_NAMES,
    abnormal_class_priors,
    entropy_weighted_ensemble,
    fit_isolation_calibration,
    isolation_anomaly_logit,
    isolation_forest_class_proba,
    positive_base_weight_grid,
    summarize_diagnostics,
)
from .inception_adapter import load_inception_predictor
from .inference import load_isolation_artifacts, load_xgboost_model
from .metrics import multiclass_metrics, save_json, save_multiclass_confusion_matrix


EPSILON = 0.05
MIN_BETA = 0.10
WEIGHT_GRID_STEP = 0.05


def main() -> None:
    ensure_dirs()
    X_full, y_full = load_ecg5000("train")
    split = get_or_create_split_indices(y_full)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    scaler = fit_or_load_scaler(X_full, train_idx)
    X_val = scaler.transform(X_full[val_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    xgb_model = load_xgboost_model()
    xgb_proba = xgb_model.predict_proba(X_val)

    inception_predictor, inception_status = load_inception_predictor()
    if inception_predictor is None:
        raise SystemExit(f"InceptionTime is required for the formula ensemble: {inception_status}")
    inception_proba = inception_predictor(X_val)

    if_model, if_config = load_isolation_artifacts()
    if_decisions = if_model.decision_function(X_val)
    if_z = isolation_anomaly_logit(if_decisions, if_config["threshold"], if_config["scale"])
    if_calibration = fit_isolation_calibration(if_z, y_val)
    abnormal_priors = abnormal_class_priors(y_train)
    if_proba = isolation_forest_class_proba(if_decisions, if_config, if_calibration, abnormal_priors)

    probabilities = {
        "xgboost": xgb_proba,
        "inception": inception_proba,
        "isolation_forest": if_proba,
    }

    best = {
        "macro_f1": -1.0,
        "balanced_accuracy": -1.0,
        "base_weights": None,
        "pred": None,
        "proba": None,
        "diagnostics": None,
    }
    for base_weights in positive_base_weight_grid(min_beta=MIN_BETA, step=WEIGHT_GRID_STEP):
        ensemble_proba, diagnostics = entropy_weighted_ensemble(probabilities, base_weights, EPSILON)
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
                    "base_weights": base_weights,
                    "pred": pred,
                    "proba": ensemble_proba,
                    "diagnostics": diagnostics,
                }
            )

    config = {
        "ensemble_type": "entropy_weighted_three_model",
        "seed": SEED,
        "selection_metric": "validation macro_f1, tie-breaker balanced_accuracy",
        "model_contribution_policy": "all models have positive base weights and entropy-adjusted per-sample weights",
        "sources_required": list(MODEL_NAMES),
        "base_weights": best["base_weights"],
        "min_beta": MIN_BETA,
        "weight_grid_step": WEIGHT_GRID_STEP,
        "epsilon": EPSILON,
        "isolation_forest": {
            "calibration": if_calibration,
            "abnormal_class_priors": abnormal_priors,
            "pseudo_probability_rule": "p_if[1]=1-a_if; p_if[c]=a_if*pi_c for c in 2..5",
            "trained_only_on_label": 1,
        },
        "inception_status": inception_status,
        "isolation_status": "loaded",
        "test_set_used_for_selection": False,
    }
    with ENSEMBLE_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics = multiclass_metrics(y_val, best["pred"])
    metrics["ensemble_config"] = config
    metrics["diagnostics"] = summarize_diagnostics(best["diagnostics"])
    save_json(metrics, REPORTS_DIR / "metrics_ensemble_validation.json")
    save_multiclass_confusion_matrix(
        y_val,
        best["pred"],
        REPORTS_DIR / "confusion_matrix_ensemble_validation.png",
        "Entropy-weighted ensemble validation confusion matrix",
    )
    print(f"Saved ensemble config to {ENSEMBLE_CONFIG_PATH}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Base weights: {best['base_weights']}")
    print(f"Inception status: {inception_status}")
    print("Isolation Forest status: loaded")


if __name__ == "__main__":
    main()
