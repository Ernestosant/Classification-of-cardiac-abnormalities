from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CLASS_VALUES, INCEPTION_MODEL_PATH, N_TIMESTEPS


@lru_cache(maxsize=1)
def load_inception_predictor(model_path: Path = INCEPTION_MODEL_PATH):
    """Load an exported fastai/tsai learner for CPU-only inference if available."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if not model_path.exists():
        return None, f"Missing InceptionTime artifact at {model_path}"

    try:
        from fastai.learner import load_learner
        from tsai.all import itemify
    except ImportError as exc:
        return None, f"fastai/tsai are not installed: {exc}"

    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            learner = load_learner(model_path, cpu=True)
        except TypeError:
            learner = load_learner(model_path)
    finally:
        pathlib.PosixPath = original_posix_path

    def predict_proba(X_scaled: np.ndarray) -> np.ndarray:
        X_ts = X_scaled.reshape(X_scaled.shape[0], N_TIMESTEPS, 1).transpose(0, 2, 1)
        ids = np.arange(1, X_ts.shape[0] + 1)
        dl = learner.dls.test_dl(itemify(X_ts, ids))
        preds, _ = learner.get_preds(dl=dl)
        probs = preds.detach().cpu().numpy()
        vocab = [int(str(v)) for v in learner.dls.vocab]
        ordered = np.zeros((X_scaled.shape[0], len(CLASS_VALUES)), dtype=float)
        for col_idx, class_value in enumerate(vocab):
            ordered[:, class_value - 1] = probs[:, col_idx]
        return ordered

    return predict_proba, "loaded"
