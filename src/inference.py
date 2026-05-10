from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    CLASS_NAMES,
    ENSEMBLE_CONFIG_PATH,
    IFOREST_CONFIG_PATH,
    IFOREST_MODEL_PATH,
    SCALER_PATH,
    XGBOOST_MODEL_PATH,
)
from .data import read_inference_csv
from .ensemble_formula import (
    MODEL_NAMES,
    entropy_weighted_ensemble,
    isolation_forest_class_proba,
    summarize_diagnostics,
)
from .inception_adapter import load_inception_predictor


def force_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@lru_cache(maxsize=1)
def load_xgboost_model(path: Path = XGBOOST_MODEL_PATH) -> XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Missing XGBoost model: {path}")
    model = XGBClassifier()
    model.load_model(path)
    return model


@lru_cache(maxsize=1)
def load_scaler(path: Path = SCALER_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_isolation_artifacts(
    model_path: Path = IFOREST_MODEL_PATH,
    config_path: Path = IFOREST_CONFIG_PATH,
):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Isolation Forest model: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Isolation Forest config: {config_path}")
    model = joblib.load(model_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


@lru_cache(maxsize=1)
def load_ensemble_config(path: Path = ENSEMBLE_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing ensemble config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_ensemble_proba(
    X_scaled: np.ndarray,
    config_path: Path = ENSEMBLE_CONFIG_PATH,
    xgb_proba: np.ndarray | None = None,
    inception_proba: np.ndarray | None = None,
    if_decision: np.ndarray | None = None,
    if_config: dict | None = None,
    include_diagnostics: bool = False,
) -> tuple[np.ndarray, dict]:
    config = load_ensemble_config(config_path)
    if config.get("ensemble_type") != "entropy_weighted_three_model":
        raise RuntimeError("Unsupported or legacy ensemble config. Run `python -m src.train_ensemble` first.")

    if xgb_proba is None:
        xgb_model = load_xgboost_model()
        xgb_proba = xgb_model.predict_proba(X_scaled)

    if inception_proba is None:
        predictor, reason = load_inception_predictor()
        if predictor is None:
            raise RuntimeError(f"InceptionTime is required for the formula ensemble: {reason}")
        inception_proba = predictor(X_scaled)

    if if_decision is None or if_config is None:
        if_model, if_config = load_isolation_artifacts()
        if_decision = if_model.decision_function(X_scaled)
    if_proba = isolation_forest_class_proba(
        if_decision,
        if_config,
        config["isolation_forest"]["calibration"],
        config["isolation_forest"]["abnormal_class_priors"],
    )

    probabilities = {
        "xgboost": xgb_proba,
        "inception": inception_proba,
        "isolation_forest": if_proba,
    }
    final, diagnostics = entropy_weighted_ensemble(probabilities, config["base_weights"], config["epsilon"])
    details = {
        "ensemble_type": config["ensemble_type"],
        "sources_used": list(MODEL_NAMES),
        "base_weights": config["base_weights"],
        "epsilon": config["epsilon"],
        **summarize_diagnostics(diagnostics),
    }
    if include_diagnostics:
        details["per_sample_entropy"] = diagnostics["entropy"]
        details["per_sample_dynamic_weights"] = diagnostics["dynamic_weights"]
        details["isolation_forest_proba"] = if_proba
    return final, details


def predict_file_to_dataframe(path: str | Path, include_inception: bool | None = None) -> tuple[pd.DataFrame, list[str]]:
    force_cpu_only()
    parsed = read_inference_csv(path)
    scaler = load_scaler()
    X_scaled = scaler.transform(parsed.X)

    xgb_model = load_xgboost_model()
    xgb_proba = xgb_model.predict_proba(X_scaled)
    xgb_pred = xgb_proba.argmax(axis=1) + 1

    inception_predictor, inception_status = load_inception_predictor()
    if inception_predictor is None:
        raise RuntimeError(f"InceptionTime is required for the formula ensemble: {inception_status}")
    inception_proba = inception_predictor(X_scaled)
    inception_pred = inception_proba.argmax(axis=1) + 1

    if_model, if_config = load_isolation_artifacts()
    if_decision = if_model.decision_function(X_scaled)
    is_anomaly = if_decision <= if_config["threshold"]

    ensemble_proba, details = predict_ensemble_proba(
        X_scaled,
        xgb_proba=xgb_proba,
        inception_proba=inception_proba,
        if_decision=if_decision,
        if_config=if_config,
        include_diagnostics=True,
    )
    ensemble_pred = ensemble_proba.argmax(axis=1) + 1
    per_sample_entropy = details["per_sample_entropy"]
    per_sample_weights = details["per_sample_dynamic_weights"]
    if_proba = details["isolation_forest_proba"]

    columns = {
        "id": np.arange(1, len(parsed.X) + 1),
        "ensemble_class": ensemble_pred,
        "ensemble_label": [CLASS_NAMES[int(label)] for label in ensemble_pred],
        "ensemble_confidence": ensemble_proba.max(axis=1),
        "xgboost_class": xgb_pred,
        "xgboost_confidence": xgb_proba.max(axis=1),
        "inception_class": inception_pred,
        "inception_label": [CLASS_NAMES[int(label)] for label in inception_pred],
        "inception_confidence": inception_proba.max(axis=1),
        "isolation_normal_probability": if_proba[:, 0],
        "isolation_anomaly_probability": if_proba[:, 1:].sum(axis=1),
        "entropy_xgboost": per_sample_entropy["xgboost"],
        "entropy_inception": per_sample_entropy["inception"],
        "entropy_isolation_forest": per_sample_entropy["isolation_forest"],
        "dynamic_weight_xgboost": per_sample_weights["xgboost"],
        "dynamic_weight_inception": per_sample_weights["inception"],
        "dynamic_weight_isolation_forest": per_sample_weights["isolation_forest"],
    }

    columns.update(
        {
            "isolation_anomaly": is_anomaly,
            "isolation_score": -if_decision,
        }
    )
    out = pd.DataFrame(columns)
    notes = parsed.notes + [
        f"Formula ensemble sources used: {', '.join(details['sources_used'])}",
        "InceptionTime is part of the final ensemble, so CPU inference can be slower on large CSV files.",
    ]
    return out, notes
