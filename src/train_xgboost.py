from __future__ import annotations

import json

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .artifacts import ensure_dirs, fit_or_load_scaler, get_or_create_split_indices
from .config import REPORTS_DIR, SEED, XGBOOST_META_PATH, XGBOOST_MODEL_PATH
from .data import load_ecg5000
from .metrics import multiclass_metrics, save_json, save_multiclass_confusion_matrix


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

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    y_train_zero = y_train - 1
    y_val_zero = y_val - 1

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method="hist",
        device="cpu",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train_zero,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val_zero)],
        verbose=False,
    )

    model.save_model(XGBOOST_MODEL_PATH)
    meta = {
        "seed": SEED,
        "model": "XGBClassifier",
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
        "labels_are_saved_as": "model classes 0-4 correspond to ECG5000 labels 1-5",
    }
    with XGBOOST_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    val_pred = model.predict(X_val) + 1
    metrics = multiclass_metrics(y_val, val_pred)
    metrics["metadata"] = meta
    save_json(metrics, REPORTS_DIR / "metrics_xgboost_validation.json")
    save_multiclass_confusion_matrix(
        y_val,
        val_pred,
        REPORTS_DIR / "confusion_matrix_xgboost_validation.png",
        "XGBoost validation confusion matrix",
    )
    print(f"Saved XGBoost model to {XGBOOST_MODEL_PATH}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

