from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from scripts.run_kaggle_inception import safe_extract
from src.artifacts import fit_or_load_scaler, get_or_create_split_indices
from src.config import ENSEMBLE_CONFIG_PATH, N_TIMESTEPS
from src.data import load_ecg5000, read_inference_csv
from src.inference import predict_ensemble_proba


def test_ecg5000_shapes_and_labels():
    X_train, y_train = load_ecg5000("train")
    X_test, y_test = load_ecg5000("test")

    assert X_train.shape == (7600, N_TIMESTEPS)
    assert X_test.shape == (1900, N_TIMESTEPS)
    assert set(np.unique(y_train)) == {1, 2, 3, 4, 5}
    assert set(np.unique(y_test)) == {1, 2, 3, 4, 5}


def test_inner_split_is_disjoint():
    _, y_train = load_ecg5000("train")
    split = get_or_create_split_indices(y_train)
    train_idx = set(split["train_idx"].tolist())
    val_idx = set(split["val_idx"].tolist())

    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(y_train)


def test_scaler_was_fit_only_on_inner_train_fold():
    X_train, y_train = load_ecg5000("train")
    split = get_or_create_split_indices(y_train)
    scaler = fit_or_load_scaler(X_train, split["train_idx"])
    expected = MinMaxScaler().fit(X_train[split["train_idx"]])

    assert np.allclose(scaler.data_min_, expected.data_min_)
    assert np.allclose(scaler.data_max_, expected.data_max_)


def test_inference_reader_accepts_and_ignores_label_column(tmp_path):
    X_test, y_test = load_ecg5000("test")
    sample = pd.DataFrame(np.column_stack([y_test[:3], X_test[:3]]))
    path = tmp_path / "with_labels.csv"
    sample.to_csv(path, index=False)

    parsed = read_inference_csv(path)

    assert parsed.X.shape == (3, N_TIMESTEPS)
    assert any("label column" in note for note in parsed.notes)


def test_ensemble_config_does_not_use_test_for_selection():
    if not ENSEMBLE_CONFIG_PATH.exists():
        return
    config = json.loads(ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["test_set_used_for_selection"] is False


def test_ensemble_gamma_zero_does_not_load_isolation_forest(tmp_path, monkeypatch):
    config_path = tmp_path / "ensemble_config.json"
    config_path.write_text(
        json.dumps(
            {
                "supervised_sources": {"xgboost": 1.0},
                "isolation_gamma": 0.0,
            }
        ),
        encoding="utf-8",
    )

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("Isolation Forest should not load when gamma is zero")

    monkeypatch.setattr("src.inference.load_isolation_artifacts", fail_if_loaded)
    X_scaled = np.zeros((2, N_TIMESTEPS), dtype=float)
    xgb_proba = np.asarray([[0.8, 0.2, 0.0, 0.0, 0.0], [0.1, 0.7, 0.2, 0.0, 0.0]])

    proba, details = predict_ensemble_proba(X_scaled, config_path=config_path, xgb_proba=xgb_proba)

    assert np.allclose(proba, xgb_proba)
    assert details["isolation_gamma"] == 0.0


def test_kaggle_artifact_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with zipfile.ZipFile(archive) as zf, pytest.raises(SystemExit):
        safe_extract(zf, tmp_path / "extract")

    assert not (tmp_path / "escape.txt").exists()
