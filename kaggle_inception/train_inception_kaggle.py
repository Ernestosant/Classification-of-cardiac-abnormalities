"""Train InceptionTime for ECG5000 on Kaggle GPU.

Outputs are written under /kaggle/working/models and /kaggle/working/reports.
The script fits preprocessing only on the inner training fold and never touches
the official test split.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def install_deps() -> None:
    packages = [
        "fastai>=2.7,<2.9",
        "tsai>=0.4,<0.5",
        "fastprogress>=1.0,<1.1",
        "starlette>=0.40,<0.47",
        "seaborn>=0.13",
        "joblib>=1.3",
        "scikit-learn>=1.4",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", *packages])


install_deps()

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import accuracy
from fastai.optimizer import ranger
from fastai.torch_core import set_seed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tsai.all import CategoryBlock, DataBlock, InceptionTime, ItemGetter, L, TSTensorBlock, itemify


SEED = 42
VALIDATION_SIZE = 0.2
N_TIMESTEPS = 140
EPOCHS = 40
LR = 3e-3
BATCH_SIZE = 128
PATIENCE = 8
CLASS_VALUES = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "Normal",
    2: "PVC",
    3: "Supraventricular premature beat",
    4: "Ectopic beat",
    5: "Unknown abnormal pathology",
}
RAW_BASE = "https://raw.githubusercontent.com/Ernestosant/Classification-of-cardiac-abnormalities/main/dataset"
WORKING_DIR = Path("/kaggle/working")
MODELS_DIR = WORKING_DIR / "models"
REPORTS_DIR = WORKING_DIR / "reports"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ecg5000_train() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f"{RAW_BASE}/ECG5000_train.csv")
    if df.shape[1] != N_TIMESTEPS + 1:
        raise ValueError(f"Expected {N_TIMESTEPS + 1} columns, got {df.shape[1]}")
    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    if set(np.unique(y).tolist()) != set(CLASS_VALUES):
        raise ValueError(f"Unexpected labels: {sorted(np.unique(y).tolist())}")
    if not np.isfinite(X).all():
        raise ValueError("Training data contains NaN or infinite values")
    return X, y


def make_split(y: np.ndarray) -> dict:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    split = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
        "n_samples": len(y),
    }
    joblib.dump(split, MODELS_DIR / "split_indices.joblib")
    return split


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    target_names = [CLASS_NAMES[label] for label in CLASS_VALUES]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_VALUES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=CLASS_VALUES, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_VALUES, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_VALUES,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_VALUES)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_VALUES, yticklabels=CLASS_VALUES)
    plt.title("InceptionTime validation confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix_inception_validation.png", dpi=180)
    plt.close()


def clean_recorder_values(values) -> list[list[float | str]]:
    cleaned = []
    for row in values:
        cleaned_row = []
        for value in list(row):
            try:
                cleaned_row.append(float(value))
            except (TypeError, ValueError):
                cleaned_row.append(str(value))
        cleaned.append(cleaned_row)
    return cleaned


def main() -> None:
    ensure_dirs()
    set_seed(SEED, reproducible=True)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    X_full, y_full = load_ecg5000_train()
    split = make_split(y_full)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

    scaler = MinMaxScaler()
    scaler.fit(X_full[train_idx])
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    X_train = scaler.transform(X_full[train_idx])
    X_val = scaler.transform(X_full[val_idx])
    y_train_numeric = y_full[train_idx]
    y_val_numeric = y_full[val_idx]
    y_train = y_train_numeric.astype(str)
    y_val = y_val_numeric.astype(str)

    X_all = np.concatenate([X_train, X_val], axis=0)
    X_all = X_all.reshape(X_all.shape[0], N_TIMESTEPS, 1).transpose(0, 2, 1)
    y_all = np.concatenate([y_train, y_val], axis=0)
    splits = (
        L(np.arange(0, len(y_train)), use_list=True),
        L(np.arange(len(y_train), len(y_all)), use_list=True),
    )

    dblock = DataBlock(
        blocks=(TSTensorBlock, CategoryBlock),
        getters=[ItemGetter(0), ItemGetter(1)],
        splitter=lambda _: splits,
    )
    dls = dblock.dataloaders(itemify(X_all, y_all), bs=BATCH_SIZE, val_bs=BATCH_SIZE * 2)

    classes = np.asarray(CLASS_VALUES)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_numeric)
    weight_by_label = {str(label): weight for label, weight in zip(classes, class_weights)}
    ordered_weights = torch.tensor(
        [weight_by_label[str(vocab_item)] for vocab_item in dls.vocab],
        dtype=torch.float32,
        device=dls.device,
    )

    net = InceptionTime(dls.dataset[0][0].shape[-2], dls.c)
    learn = Learner(
        dls,
        net,
        loss_func=CrossEntropyLossFlat(weight=ordered_weights),
        metrics=accuracy,
        opt_func=ranger,
    )
    callbacks = [
        EarlyStoppingCallback(monitor="valid_loss", patience=PATIENCE),
        SaveModelCallback(monitor="valid_loss", fname="best_inception_time"),
    ]
    learn.fit_one_cycle(EPOCHS, LR, cbs=callbacks)
    learn.load("best_inception_time")

    preds, targets = learn.get_preds(dl=dls.valid)
    vocab = [int(str(v)) for v in dls.vocab]
    pred_labels = np.asarray([vocab[i] for i in preds.argmax(dim=1).cpu().numpy()])
    true_labels = np.asarray([vocab[i] for i in targets.cpu().numpy()])

    learn.export(MODELS_DIR / "inception_cpu.pkl")
    metrics = multiclass_metrics(true_labels, pred_labels)
    metrics["metadata"] = {
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
        "epochs_requested": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "device": device_name,
        "test_set_used": False,
        "scaler_fit_scope": "inner training split only",
        "class_weighted_loss": True,
        "recorder_values": clean_recorder_values(learn.recorder.values),
    }
    save_json(metrics, REPORTS_DIR / "metrics_inception_validation.json")
    save_confusion_matrix(true_labels, pred_labels)
    save_json(metrics["metadata"], REPORTS_DIR / "inception_training_config.json")

    artifact_root = WORKING_DIR / "inception_artifacts"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    shutil.copytree(MODELS_DIR, artifact_root / "models")
    shutil.copytree(REPORTS_DIR, artifact_root / "reports")
    shutil.make_archive(str(WORKING_DIR / "inception_artifacts"), "zip", artifact_root)

    print(json.dumps({"status": "ok", "macro_f1": metrics["macro_f1"], "device": device_name}, indent=2))


if __name__ == "__main__":
    main()
