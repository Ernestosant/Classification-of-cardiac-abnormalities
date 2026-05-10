from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import CLASS_NAMES, CLASS_VALUES


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = CLASS_VALUES
    target_names = [CLASS_NAMES[label] for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": report,
    }


def binary_anomaly_metrics(y_true_labels: np.ndarray, y_pred_anomaly: np.ndarray) -> dict[str, Any]:
    y_true_anomaly = (y_true_labels != 1).astype(int)
    y_pred_anomaly = y_pred_anomaly.astype(int)
    report = classification_report(
        y_true_anomaly,
        y_pred_anomaly,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true_anomaly, y_pred_anomaly)),
        "macro_f1": float(f1_score(y_true_anomaly, y_pred_anomaly, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_anomaly, y_pred_anomaly)),
        "normal_specificity": float(recall_score(y_true_anomaly, y_pred_anomaly, pos_label=0, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true_anomaly, y_pred_anomaly, pos_label=1, zero_division=0)),
        "classification_report": report,
    }


def save_multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_VALUES)
    _save_confusion_matrix(cm, [str(label) for label in CLASS_VALUES], path, title)


def save_binary_confusion_matrix(y_true_labels: np.ndarray, y_pred_anomaly: np.ndarray, path: Path, title: str) -> None:
    y_true_anomaly = (y_true_labels != 1).astype(int)
    cm = confusion_matrix(y_true_anomaly, y_pred_anomaly, labels=[0, 1])
    _save_confusion_matrix(cm, ["normal", "anomaly"], path, title)


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

