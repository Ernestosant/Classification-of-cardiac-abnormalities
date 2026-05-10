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
    CLASS_VALUES,
    ENSEMBLE_CONFIG_PATH,
    IFOREST_CONFIG_PATH,
    IFOREST_MODEL_PATH,
    SCALER_PATH,
    XGBOOST_MODEL_PATH,
)
from .data import read_inference_csv
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


def anomaly_confidence(decision_scores: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-6)
    logits = (threshold - decision_scores) / scale
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def apply_isolation_adjustment(proba: np.ndarray, anomaly_conf: np.ndarray, gamma: float) -> np.ndarray:
    adjusted = np.asarray(proba, dtype=float).copy()
    gamma = float(gamma)
    if gamma <= 0:
        return normalize_proba(adjusted)

    shift = adjusted[:, 0] * gamma * anomaly_conf
    adjusted[:, 0] = np.maximum(adjusted[:, 0] - shift, 0.0)

    abnormal = adjusted[:, 1:]
    abnormal_sum = abnormal.sum(axis=1, keepdims=True)
    fallback = np.full_like(abnormal, 1.0 / (len(CLASS_VALUES) - 1))
    weights = np.divide(abnormal, abnormal_sum, out=fallback, where=abnormal_sum > 1e-12)
    adjusted[:, 1:] += shift[:, None] * weights
    return normalize_proba(adjusted)


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    row_sum = proba.sum(axis=1, keepdims=True)
    return np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 1e-12)


def predict_ensemble_proba(X_scaled: np.ndarray, config_path: Path = ENSEMBLE_CONFIG_PATH) -> tuple[np.ndarray, dict]:
    config = load_ensemble_config(config_path)

    sources = config["supervised_sources"]
    probas = []
    weights = []
    details = {"sources_used": [], "sources_skipped": []}

    xgb_weight = float(sources.get("xgboost", 0.0))
    if xgb_weight > 0.0:
        xgb_model = load_xgboost_model()
        probas.append(xgb_model.predict_proba(X_scaled))
        weights.append(xgb_weight)
        details["sources_used"].append("xgboost")

    inception_weight = float(sources.get("inception", 0.0))
    if inception_weight > 0.0:
        predictor, reason = load_inception_predictor()
        if predictor is None:
            details["sources_skipped"].append({"inception": reason})
        else:
            probas.append(predictor(X_scaled))
            weights.append(inception_weight)
            details["sources_used"].append("inception")

    if not probas:
        raise RuntimeError("No supervised model was available for ensemble prediction")

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    supervised = sum(weight * proba for weight, proba in zip(weights_arr, probas))
    supervised = normalize_proba(supervised)

    if_model, if_config = load_isolation_artifacts()
    decisions = if_model.decision_function(X_scaled)
    anomaly_conf = anomaly_confidence(decisions, if_config["threshold"], if_config["scale"])
    final = apply_isolation_adjustment(supervised, anomaly_conf, config["isolation_gamma"])
    details["isolation_gamma"] = config["isolation_gamma"]
    return final, details


def predict_file_to_dataframe(path: str | Path, include_inception: bool = False) -> tuple[pd.DataFrame, list[str]]:
    force_cpu_only()
    parsed = read_inference_csv(path)
    scaler = load_scaler()
    X_scaled = scaler.transform(parsed.X)

    xgb_model = load_xgboost_model()
    xgb_proba = xgb_model.predict_proba(X_scaled)
    xgb_pred = xgb_proba.argmax(axis=1) + 1

    if_model, if_config = load_isolation_artifacts()
    if_decision = if_model.decision_function(X_scaled)
    is_anomaly = if_decision <= if_config["threshold"]

    ensemble_proba, details = predict_ensemble_proba(X_scaled)
    ensemble_pred = ensemble_proba.argmax(axis=1) + 1

    columns = {
        "id": np.arange(1, len(parsed.X) + 1),
        "ensemble_class": ensemble_pred,
        "ensemble_label": [CLASS_NAMES[int(label)] for label in ensemble_pred],
        "ensemble_confidence": ensemble_proba.max(axis=1),
        "xgboost_class": xgb_pred,
        "xgboost_confidence": xgb_proba.max(axis=1),
    }

    if include_inception:
        inception_predictor, inception_status = load_inception_predictor()
        if inception_predictor is None:
            inception_note = f"InceptionTime unavailable: {inception_status}"
        else:
            inception_proba = inception_predictor(X_scaled)
            inception_pred = inception_proba.argmax(axis=1) + 1
            columns.update(
                {
                    "inception_class": inception_pred,
                    "inception_label": [CLASS_NAMES[int(label)] for label in inception_pred],
                    "inception_confidence": inception_proba.max(axis=1),
                }
            )
            inception_note = "InceptionTime prediction completed."
    else:
        inception_note = "Skipped separate InceptionTime columns for faster CPU inference."

    columns.update(
        {
            "isolation_anomaly": is_anomaly,
            "isolation_score": -if_decision,
        }
    )
    out = pd.DataFrame(columns)
    notes = parsed.notes + [f"Ensemble sources used: {', '.join(details['sources_used'])}"]
    notes.append(inception_note)
    if details["sources_skipped"]:
        notes.append(f"Skipped sources: {details['sources_skipped']}")
    return out, notes
