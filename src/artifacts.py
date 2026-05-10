from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from .config import MODELS_DIR, REPORTS_DIR, SCALER_PATH, SEED, SPLIT_PATH, VALIDATION_SIZE


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_or_create_split_indices(
    y: np.ndarray,
    split_path: Path = SPLIT_PATH,
    seed: int = SEED,
    validation_size: float = VALIDATION_SIZE,
    force: bool = False,
) -> dict:
    """Create one stratified train/validation split from the official train set."""
    ensure_dirs()
    if split_path.exists() and not force:
        split = joblib.load(split_path)
        if split.get("n_samples") != len(y):
            raise ValueError("Saved split does not match the current training set size")
        return split

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    split = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "seed": seed,
        "validation_size": validation_size,
        "n_samples": len(y),
    }
    joblib.dump(split, split_path)
    return split


def fit_or_load_scaler(
    X_full_train: np.ndarray,
    train_idx: np.ndarray,
    scaler_path: Path = SCALER_PATH,
    force: bool = False,
) -> MinMaxScaler:
    """Fit MinMaxScaler only on the inner training fold, never on validation/test."""
    ensure_dirs()
    if scaler_path.exists() and not force:
        return joblib.load(scaler_path)

    scaler = MinMaxScaler()
    scaler.fit(X_full_train[train_idx])
    joblib.dump(scaler, scaler_path)
    return scaler

