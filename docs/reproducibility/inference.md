# Inference

Inference is CPU-only. The prediction path loads saved artifacts and does not
fit a scaler, tune thresholds, train models, or update artifacts.

## Batch CLI

Run:

```powershell
python -m src.predict --input examples\sample_input.csv --output predictions.csv
```

The input CSV must contain either:

| Shape | Meaning |
|---|---|
| 140 columns | Feature-only ECG samples |
| 141 columns | First column is a label and is ignored |

The output CSV includes ensemble predictions, XGBoost predictions, InceptionTime
predictions, calibrated Isolation Forest probabilities, entropy diagnostics, and
dynamic ensemble weights.

InceptionTime is part of the final formula ensemble. Full-file inference is
therefore slower than an XGBoost-only path when running on CPU.

## Gradio Interface

Launch:

```powershell
python app.py
```

The interface accepts a CSV file and returns a predictions table plus notes about
label-column detection and model availability.

The app includes `examples/sample_input.csv` as a loadable example. The Gradio
table intentionally shows a compact reviewer summary: `id`, `ensemble_class`,
`ensemble_label`, `ensemble_confidence`, and
`isolation_anomaly_probability`. The batch CLI still writes the full prediction
table.

The app always runs the full three-model ensemble. Large CSV files can take
longer because the InceptionTime learner is evaluated on CPU.

## CPU-Only Behavior

The inference code sets `CUDA_VISIBLE_DEVICES` to disable GPU use. This applies
to the CLI and the Gradio app. The InceptionTime adapter loads the exported
FastAI/tsai learner on CPU.

## Model Availability

The ensemble requires all three model artifacts. The current formula uses
positive base weights for XGBoost, InceptionTime, and Isolation Forest, then
adjusts their per-sample contribution using normalized entropy.
