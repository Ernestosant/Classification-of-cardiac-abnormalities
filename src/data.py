from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import CLASS_VALUES, DATA_DIR, N_TIMESTEPS


@dataclass(frozen=True)
class InferenceInput:
    X: np.ndarray
    notes: list[str]


def load_ecg5000(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load the official ECG5000 train or test CSV with labels in column 0."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    path = data_dir / f"ECG5000_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ECG5000 file: {path}")

    df = pd.read_csv(path)
    expected_cols = N_TIMESTEPS + 1
    if df.shape[1] != expected_cols:
        raise ValueError(f"{path.name} must have {expected_cols} columns; got {df.shape[1]}")

    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    _validate_labeled_arrays(X, y, source=path.name)
    return X, y


def _validate_labeled_arrays(X: np.ndarray, y: np.ndarray, source: str) -> None:
    if X.ndim != 2 or X.shape[1] != N_TIMESTEPS:
        raise ValueError(f"{source} features must have shape (n_samples, {N_TIMESTEPS})")
    labels = set(np.unique(y).tolist())
    allowed = set(CLASS_VALUES)
    if not labels.issubset(allowed):
        raise ValueError(f"{source} contains labels outside {sorted(allowed)}: {sorted(labels)}")
    if not np.isfinite(X).all():
        raise ValueError(f"{source} contains NaN or infinite feature values")


def read_inference_csv(path: str | Path) -> InferenceInput:
    """Read an inference CSV with 140 feature columns or 141 columns including labels.

    If a label column is present, it is ignored and reported in notes. No scaler or
    other preprocessing is fitted here; inference must use saved training artifacts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    notes: list[str] = []
    try:
        df = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError("The CSV is empty") from exc
    if df.empty:
        raise ValueError("The CSV is empty")

    if df.shape[1] not in {N_TIMESTEPS, N_TIMESTEPS + 1}:
        try:
            no_header_df = pd.read_csv(path, header=None)
        except EmptyDataError as exc:
            raise ValueError("The CSV is empty") from exc
        if no_header_df.shape[1] in {N_TIMESTEPS, N_TIMESTEPS + 1}:
            df = no_header_df
            notes.append("CSV loaded without header row.")
        else:
            raise ValueError(
                f"Expected {N_TIMESTEPS} feature columns or {N_TIMESTEPS + 1} columns "
                f"with a label column; got {df.shape[1]} columns."
            )

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("The CSV contains non-numeric or missing values")

    arr = numeric.to_numpy(dtype=float)
    if arr.shape[1] == N_TIMESTEPS + 1:
        first_col = arr[:, 0]
        labels = set(np.unique(first_col.astype(int)).tolist())
        if np.all(first_col == first_col.astype(int)) and labels.issubset(set(CLASS_VALUES)):
            notes.append("Detected and ignored the first label column.")
            arr = arr[:, 1:]
        else:
            raise ValueError(
                f"Found {N_TIMESTEPS + 1} columns, but the first column does not look like "
                "ECG5000 labels 1-5."
            )

    if arr.shape[1] != N_TIMESTEPS:
        raise ValueError(f"Expected {N_TIMESTEPS} feature columns; got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError("The CSV contains NaN or infinite feature values")

    return InferenceInput(X=arr, notes=notes)
