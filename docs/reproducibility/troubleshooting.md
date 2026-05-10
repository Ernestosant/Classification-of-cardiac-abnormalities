# Troubleshooting

This page collects common issues seen while training and running the project.

## Kaggle Authentication

If Kaggle commands fail with authentication errors:

1. Confirm that a valid `kaggle.json` token exists locally.
2. Confirm that the token is not committed to Git.
3. Run the helper command again:

```powershell
python scripts\run_kaggle_inception.py --submit --wait --download --copy-artifacts
```

The helper checks supported credential locations and uses the first valid
configuration it can authenticate with.

## InceptionTime Loads On Kaggle But Not Windows

FastAI learners exported on Linux can include POSIX paths inside the pickle. The
local adapter in `src.inception_adapter` handles this compatibility issue when
loading `models/inception_cpu.pkl` on Windows.

If loading still fails, confirm that:

```powershell
pip install -r requirements-inception.txt
```

has completed successfully.

## Gradio Or FastAPI Dependency Conflicts

The project pins compatible versions in `requirements.txt` and
`requirements-inception.txt`. If the global Python environment has unrelated
packages that require incompatible versions, create a fresh virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-inception.txt
```

Then validate:

```powershell
python -m pytest -q
python app.py
```

## Unexpectedly Good Metrics

Very high metrics can be suspicious in small benchmark datasets. Re-run:

```powershell
python -m src.evaluate
```

Then inspect `reports/model_results.md` and confirm:

- The scaler fit scope is `inner training split only`.
- Test data was not used for model selection.
- The ensemble configuration was selected on validation.

## Missing Model Artifacts

If inference fails because an artifact is missing, retrain or regenerate the
artifact:

| Missing artifact | Command |
|---|---|
| `models/xgboost_model.json` | `python -m src.train_xgboost` |
| `models/isolation_forest.joblib` | `python -m src.train_isolation_forest` |
| `models/inception_cpu.pkl` | Kaggle InceptionTime command |
| `models/ensemble_config.json` | `python -m src.train_ensemble` |

