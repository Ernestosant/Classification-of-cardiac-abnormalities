# Data Preprocessing

The preprocessing pipeline is intentionally small. This reduces the risk of
accidental leakage and keeps inference behavior easy to inspect.

## Input Format

Each ECG5000 row contains one label and 140 signal samples in the dataset files.
For inference, the CLI and Gradio app accept either:

| Accepted CSV shape | Behavior |
|---|---|
| 140 columns | Treated as unlabeled heartbeat samples |
| 141 columns | First column is treated as an ECG5000 label and ignored |

Files with an invalid number of columns or empty content are rejected.

## Scaling

The project uses `MinMaxScaler`. The scaler is fitted only on the inner training
split created from the official training file. The same saved scaler is then
used for validation, test, and inference.

This means:

- Validation data does not influence scaling parameters.
- Test data does not influence scaling parameters.
- Inference never fits or updates preprocessing artifacts.

## Label Handling

Training and evaluation use ECG5000 labels `1` through `5`. Some libraries use
zero-based labels internally, but public outputs remain one-based and are paired
with human-readable class names.

## Inference Validation

The prediction path validates CSV shape before scaling. If a label column is
detected, it is ignored and a note is returned to the user. The output includes:

| Column | Meaning |
|---|---|
| `ensemble_class` | Final selected class from the ensemble |
| `ensemble_label` | Human-readable class name |
| `ensemble_confidence` | Maximum ensemble probability |
| `xgboost_class` | XGBoost class prediction |
| `xgboost_confidence` | Maximum XGBoost probability |
| `inception_class` | InceptionTime prediction when artifact is available |
| `isolation_normal_probability` | Calibrated Isolation Forest normal probability |
| `isolation_anomaly_probability` | Calibrated Isolation Forest anomaly probability |
| `entropy_*` | Normalized entropy for each model probability vector |
| `dynamic_weight_*` | Per-sample entropy-adjusted ensemble weight |
| `isolation_anomaly` | Binary anomaly decision |
| `isolation_score` | Higher values indicate more anomalous samples |
