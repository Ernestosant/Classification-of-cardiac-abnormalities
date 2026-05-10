from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import CLASS_VALUES


MODEL_NAMES = ("xgboost", "inception", "isolation_forest")


def validate_abnormal_priors(abnormal_priors: dict[str, float]) -> np.ndarray:
    missing = [str(label) for label in CLASS_VALUES[1:] if str(label) not in abnormal_priors]
    if missing:
        raise ValueError(f"Missing abnormal class priors for labels: {missing}")

    try:
        priors = np.asarray([float(abnormal_priors[str(label)]) for label in CLASS_VALUES[1:]], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Abnormal class priors must be finite numeric values") from exc
    if not np.all(np.isfinite(priors)):
        raise ValueError("Abnormal class priors must be finite numeric values")
    if np.any(priors < 0.0):
        raise ValueError("Abnormal class priors must be non-negative")

    prior_sum = float(priors.sum())
    if prior_sum <= 0.0:
        raise ValueError("Abnormal class priors must have a positive sum")
    return priors / prior_sum


def validate_base_weights(base_weights: dict[str, float]) -> dict[str, float]:
    missing = [name for name in MODEL_NAMES if name not in base_weights]
    if missing:
        raise ValueError(f"Missing ensemble base weights for models: {missing}")

    try:
        weights = {name: float(base_weights[name]) for name in MODEL_NAMES}
    except (TypeError, ValueError) as exc:
        raise ValueError("Ensemble base weights must be finite numeric values") from exc
    values = np.asarray(list(weights.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Ensemble base weights must be finite numeric values")
    if np.any(values < 0.0):
        raise ValueError("Ensemble base weights must be non-negative")
    if float(values.sum()) <= 0.0:
        raise ValueError("Ensemble base weights must have a positive sum")
    return weights


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    row_sum = proba.sum(axis=1, keepdims=True)
    return np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 1e-12)


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(normalize_proba(proba), 1e-12, 1.0)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    return entropy / math.log(proba.shape[1])


def isolation_anomaly_logit(decision_scores: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    scale = max(abs(float(scale)), 1e-6)
    return (float(threshold) - np.asarray(decision_scores, dtype=float)) / scale


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_isolation_calibration(z_if: np.ndarray, y_labels: np.ndarray) -> dict[str, Any]:
    y_anomaly = (np.asarray(y_labels) != 1).astype(int)
    z_if = np.asarray(z_if, dtype=float).reshape(-1, 1)
    unique = np.unique(y_anomaly)
    if len(unique) < 2:
        anomaly_rate = float(np.clip(y_anomaly.mean(), 1e-6, 1.0 - 1e-6))
        return {
            "method": "fallback_logit_prior",
            "intercept": float(np.log(anomaly_rate / (1.0 - anomaly_rate))),
            "coefficient": 0.0,
            "input": "(threshold - decision_function) / scale",
            "uses_validation_labels": True,
        }

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(z_if, y_anomaly)
    return {
        "method": "validation_logistic_regression",
        "intercept": float(model.intercept_[0]),
        "coefficient": float(model.coef_[0][0]),
        "input": "(threshold - decision_function) / scale",
        "uses_validation_labels": True,
    }


def calibrated_anomaly_probability(z_if: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    logits = float(calibration["intercept"]) + float(calibration["coefficient"]) * np.asarray(z_if, dtype=float)
    return sigmoid(logits)


def abnormal_class_priors(y_train: np.ndarray) -> dict[str, float]:
    y_train = np.asarray(y_train)
    counts = {label: int((y_train == label).sum()) for label in CLASS_VALUES[1:]}
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot compute abnormal class priors without abnormal training samples")
    return {str(label): counts[label] / total for label in CLASS_VALUES[1:]}


def isolation_forest_class_proba(
    decision_scores: np.ndarray,
    if_config: dict[str, Any],
    calibration: dict[str, Any],
    abnormal_priors: dict[str, float],
) -> np.ndarray:
    z_if = isolation_anomaly_logit(decision_scores, if_config["threshold"], if_config["scale"])
    anomaly_prob = calibrated_anomaly_probability(z_if, calibration)
    priors = validate_abnormal_priors(abnormal_priors)

    proba = np.zeros((len(anomaly_prob), len(CLASS_VALUES)), dtype=float)
    proba[:, 0] = 1.0 - anomaly_prob
    proba[:, 1:] = anomaly_prob[:, None] * priors[None, :]
    return normalize_proba(proba)


def positive_base_weight_grid(min_beta: float = 0.10, step: float = 0.05) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    min_units = int(math.ceil(min_beta / step - 1e-12))
    weights: list[dict[str, float]] = []
    for xgb_units in range(min_units, units - 2 * min_units + 1):
        for inc_units in range(min_units, units - xgb_units - min_units + 1):
            if_units = units - xgb_units - inc_units
            if if_units < min_units:
                continue
            weights.append(
                {
                    "xgboost": round(xgb_units * step, 10),
                    "inception": round(inc_units * step, 10),
                    "isolation_forest": round(if_units * step, 10),
                }
            )
    return weights


def entropy_weighted_ensemble(
    probabilities: dict[str, np.ndarray],
    base_weights: dict[str, float],
    epsilon: float,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    missing = [name for name in MODEL_NAMES if name not in probabilities]
    if missing:
        raise ValueError(f"Missing model probabilities for ensemble: {missing}")
    weights = validate_base_weights(base_weights)
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("Entropy epsilon must be a finite non-negative value")

    normalized = {name: normalize_proba(probabilities[name]) for name in MODEL_NAMES}
    entropies = {name: normalized_entropy(normalized[name]) for name in MODEL_NAMES}
    confidence = {name: 1.0 - entropies[name] for name in MODEL_NAMES}

    numerator = {
        name: weights[name] * (epsilon + confidence[name]) for name in MODEL_NAMES
    }
    denominator = sum(numerator.values())
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        raise ValueError("Entropy-weighted ensemble denominator must be finite and positive")
    dynamic_weights = {name: numerator[name] / denominator for name in MODEL_NAMES}

    final = np.zeros_like(normalized["xgboost"], dtype=float)
    for name in MODEL_NAMES:
        final += dynamic_weights[name][:, None] * normalized[name]

    return normalize_proba(final), {"entropy": entropies, "dynamic_weights": dynamic_weights}


def summarize_diagnostics(diagnostics: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {
        "mean_entropy": {
            name: float(np.mean(values)) for name, values in diagnostics["entropy"].items()
        },
        "mean_dynamic_weights": {
            name: float(np.mean(values)) for name, values in diagnostics["dynamic_weights"].items()
        },
    }
