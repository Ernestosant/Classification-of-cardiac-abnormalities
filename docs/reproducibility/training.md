# Training

This page lists the exact commands used to train or refresh the current
artifacts.

## Local Models

Train XGBoost:

```powershell
python -m src.train_xgboost
```

Train Isolation Forest:

```powershell
python -m src.train_isolation_forest
```

Select the ensemble on validation:

```powershell
python -m src.train_ensemble
```

Evaluate on the official test split and regenerate reports:

```powershell
python -m src.evaluate
```

## Kaggle GPU InceptionTime Training

The Kaggle workflow is the preferred GPU path for InceptionTime retraining.

```powershell
python scripts\run_kaggle_inception.py --submit --wait --download --copy-artifacts
```

This command submits the Kaggle script, waits for the run to finish, downloads
the output, and copies the exported artifacts into the local `models/` and
`reports/` folders.

After downloading InceptionTime artifacts, rebuild the ensemble and final
report:

```powershell
python -m src.train_ensemble
python -m src.evaluate
```

## Full Reproducible Sequence

```powershell
python -m src.train_xgboost
python -m src.train_isolation_forest
python scripts\run_kaggle_inception.py --submit --wait --download --copy-artifacts
python -m src.train_ensemble
python -m src.evaluate
```

## Expected Outputs

| Command | Main outputs |
|---|---|
| `python -m src.train_xgboost` | `models/xgboost_model.json`, XGBoost validation metrics |
| `python -m src.train_isolation_forest` | `models/isolation_forest.joblib`, anomaly threshold config |
| Kaggle InceptionTime command | `models/inception_cpu.pkl`, InceptionTime validation metrics |
| `python -m src.train_ensemble` | `models/ensemble_config.json` |
| `python -m src.evaluate` | `reports/model_results.md`, test metrics, confusion matrices |

