# Setup

This project is designed to run locally with CPU-only inference. GPU access is
only needed when retraining InceptionTime.

## Core Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The core dependencies support:

- Data loading and preprocessing.
- XGBoost training.
- Isolation Forest training.
- Ensemble selection.
- Evaluation and report generation.
- CLI and Gradio inference.

## Optional InceptionTime Dependencies

To load or retrain the exported InceptionTime learner locally:

```powershell
pip install -r requirements-inception.txt
```

These dependencies include FastAI, tsai, PyTorch, and compatibility pins used to
avoid conflicts with the Gradio/FastAPI stack.

## Kaggle Credentials

Kaggle GPU training requires a local `kaggle.json` API token. Keep credentials
out of Git. The repository ignores:

```text
.kaggle/
kaggle.json
**/kaggle.json
```

Place the token in one of the locations supported by the helper script, then run
the Kaggle training command documented in [training](training.md).

## Verifying The Environment

Run:

```powershell
python -m pytest -q
python -m src.predict --help
```

The first command checks leakage and pipeline guards. The second confirms that
the batch inference command is available.

