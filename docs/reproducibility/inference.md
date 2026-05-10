# Inference

Inference is CPU-only. The prediction path loads saved artifacts and does not
fit a scaler, tune thresholds, train models, or update artifacts.

## Batch CLI

Run:

```powershell
python -m src.predict --input path\to\beats.csv --output predictions.csv
```

The input CSV must contain either:

| Shape | Meaning |
|---|---|
| 140 columns | Feature-only ECG samples |
| 141 columns | First column is a label and is ignored |

The output CSV includes ensemble predictions, XGBoost predictions, InceptionTime
predictions when available, and Isolation Forest anomaly scores.

## Gradio Interface

Launch:

```powershell
python app.py
```

The interface accepts a CSV file and returns a predictions table plus notes about
label-column detection and model availability.

## CPU-Only Behavior

The inference code sets `CUDA_VISIBLE_DEVICES` to disable GPU use. This applies
to the CLI and the Gradio app. The InceptionTime adapter loads the exported
FastAI/tsai learner on CPU.

## Model Availability

The ensemble uses only models with positive weights in
`models/ensemble_config.json`. The current ensemble uses XGBoost. InceptionTime
is still reported separately in inference outputs when `models/inception_cpu.pkl`
and the optional dependencies are available.

