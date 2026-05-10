from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, f1_score

from .artifacts import ensure_dirs, fit_or_load_scaler, get_or_create_split_indices
from .config import IFOREST_CONFIG_PATH, IFOREST_MODEL_PATH, REPORTS_DIR, SEED
from .data import load_ecg5000
from .metrics import binary_anomaly_metrics, save_binary_confusion_matrix, save_json


def main() -> None:
    ensure_dirs()
    X_full, y_full = load_ecg5000("train")
    split = get_or_create_split_indices(y_full)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

    scaler = fit_or_load_scaler(X_full, train_idx)
    X_train = scaler.transform(X_full[train_idx])
    X_val = scaler.transform(X_full[val_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    X_normal = X_train[y_train == 1]
    model = IsolationForest(
        n_estimators=500,
        max_samples="auto",
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_normal)

    decisions = model.decision_function(X_val)
    y_val_anomaly = (y_val != 1).astype(int)
    best = {"threshold": None, "macro_f1": -1.0, "balanced_accuracy": -1.0}
    for threshold in np.quantile(decisions, np.linspace(0.01, 0.99, 99)):
        pred_anomaly = (decisions <= threshold).astype(int)
        macro_f1 = f1_score(y_val_anomaly, pred_anomaly, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(y_val_anomaly, pred_anomaly)
        if (macro_f1 > best["macro_f1"]) or (
            np.isclose(macro_f1, best["macro_f1"]) and balanced > best["balanced_accuracy"]
        ):
            best = {
                "threshold": float(threshold),
                "macro_f1": float(macro_f1),
                "balanced_accuracy": float(balanced),
            }

    threshold = float(best["threshold"])
    pred_anomaly = decisions <= threshold
    metrics = binary_anomaly_metrics(y_val, pred_anomaly)
    metrics["threshold_selection"] = best
    metrics["train_normal_samples"] = int(X_normal.shape[0])

    scale = float(np.std(decisions))
    if scale <= 1e-8:
        scale = 1.0
    config = {
        "threshold": threshold,
        "scale": scale,
        "score_rule": "decision_function <= threshold means anomaly",
        "trained_only_on_label": 1,
        "seed": SEED,
    }

    joblib.dump(model, IFOREST_MODEL_PATH)
    with IFOREST_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    save_json(metrics, REPORTS_DIR / "metrics_isolation_forest_validation.json")
    save_binary_confusion_matrix(
        y_val,
        pred_anomaly,
        REPORTS_DIR / "confusion_matrix_isolation_forest_validation.png",
        "Isolation Forest validation confusion matrix",
    )
    print(f"Saved Isolation Forest model to {IFOREST_MODEL_PATH}")
    print(f"Validation anomaly macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

