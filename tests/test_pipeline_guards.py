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
from src.ensemble_formula import (
    MODEL_NAMES,
    entropy_weighted_ensemble,
    isolation_forest_class_proba,
)
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
    if config.get("ensemble_type") == "entropy_weighted_three_model":
        weights = config["base_weights"]
        assert set(weights) == set(MODEL_NAMES)
        assert all(weights[name] > 0.0 for name in MODEL_NAMES)
        assert np.isclose(sum(weights.values()), 1.0)


def test_formula_ensemble_uses_all_three_models_and_sums_to_one(tmp_path):
    config_path = tmp_path / "ensemble_config.json"
    config_path.write_text(
        json.dumps(
            {
                "ensemble_type": "entropy_weighted_three_model",
                "base_weights": {"xgboost": 0.5, "inception": 0.3, "isolation_forest": 0.2},
                "epsilon": 0.05,
                "isolation_forest": {
                    "calibration": {"intercept": 0.0, "coefficient": 1.0},
                    "abnormal_class_priors": {"2": 0.7, "3": 0.1, "4": 0.15, "5": 0.05},
                },
                "test_set_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )

    X_scaled = np.zeros((2, N_TIMESTEPS), dtype=float)
    xgb_proba = np.asarray([[0.8, 0.2, 0.0, 0.0, 0.0], [0.1, 0.7, 0.2, 0.0, 0.0]])
    inception_proba = np.asarray([[0.6, 0.3, 0.1, 0.0, 0.0], [0.2, 0.6, 0.1, 0.1, 0.0]])
    if_decision = np.asarray([-1.0, 1.0])
    if_config = {"threshold": 0.0, "scale": 1.0}

    proba, details = predict_ensemble_proba(
        X_scaled,
        config_path=config_path,
        xgb_proba=xgb_proba,
        inception_proba=inception_proba,
        if_decision=if_decision,
        if_config=if_config,
        include_diagnostics=True,
    )

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert details["sources_used"] == list(MODEL_NAMES)
    dynamic = details["per_sample_dynamic_weights"]
    assert all(np.all(dynamic[name] > 0.0) for name in MODEL_NAMES)
    assert np.allclose(sum(dynamic[name] for name in MODEL_NAMES), 1.0)


def test_entropy_changes_dynamic_weights():
    probabilities = {
        "xgboost": np.asarray([[0.96, 0.01, 0.01, 0.01, 0.01], [0.2, 0.2, 0.2, 0.2, 0.2]]),
        "inception": np.asarray([[0.2, 0.2, 0.2, 0.2, 0.2], [0.96, 0.01, 0.01, 0.01, 0.01]]),
        "isolation_forest": np.asarray([[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]]),
    }

    _, diagnostics = entropy_weighted_ensemble(
        probabilities,
        {"xgboost": 1 / 3, "inception": 1 / 3, "isolation_forest": 1 / 3},
        epsilon=0.05,
    )

    assert diagnostics["dynamic_weights"]["xgboost"][0] > diagnostics["dynamic_weights"]["xgboost"][1]
    assert diagnostics["dynamic_weights"]["inception"][1] > diagnostics["dynamic_weights"]["inception"][0]


def test_isolation_forest_pseudo_probabilities_sum_to_one():
    proba = isolation_forest_class_proba(
        decision_scores=np.asarray([-2.0, 2.0]),
        if_config={"threshold": 0.0, "scale": 1.0},
        calibration={"intercept": 0.0, "coefficient": 1.0},
        abnormal_priors={"2": 0.7, "3": 0.1, "4": 0.15, "5": 0.05},
    )

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] < proba[1, 0]


def test_isolation_forest_pseudo_probabilities_reject_invalid_priors():
    with pytest.raises(ValueError, match="Missing abnormal class priors"):
        isolation_forest_class_proba(
            decision_scores=np.asarray([0.0]),
            if_config={"threshold": 0.0, "scale": 1.0},
            calibration={"intercept": 0.0, "coefficient": 1.0},
            abnormal_priors={"2": 0.0, "3": 0.0, "4": 0.0},
        )

    with pytest.raises(ValueError, match="positive sum"):
        isolation_forest_class_proba(
            decision_scores=np.asarray([0.0]),
            if_config={"threshold": 0.0, "scale": 1.0},
            calibration={"intercept": 0.0, "coefficient": 1.0},
            abnormal_priors={"2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
        )


def test_entropy_weighted_ensemble_rejects_invalid_base_weights():
    probabilities = {
        "xgboost": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        "inception": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        "isolation_forest": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    }

    with pytest.raises(ValueError, match="Missing ensemble base weights"):
        entropy_weighted_ensemble(probabilities, {"xgboost": 1.0, "inception": 0.0}, epsilon=0.05)

    with pytest.raises(ValueError, match="positive sum"):
        entropy_weighted_ensemble(
            probabilities,
            {"xgboost": 0.0, "inception": 0.0, "isolation_forest": 0.0},
            epsilon=0.05,
        )


def test_kaggle_artifact_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with zipfile.ZipFile(archive) as zf, pytest.raises(SystemExit):
        safe_extract(zf, tmp_path / "extract")

    assert not (tmp_path / "escape.txt").exists()
