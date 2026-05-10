# ECG5000 Model Results

## Scope

This report documents a reproducible ECG5000 experiment for five-class heartbeat classification. It is not a clinically validated diagnostic system.

## Anti-Leakage Controls

- The official `ECG5000_test.csv` split is reserved for final evaluation.
- The scaler is fitted only on the inner training split created from `ECG5000_train.csv`.
- Validation is used for early stopping, Isolation Forest threshold selection, and ensemble weight selection.
- Test labels are not used for preprocessing, model selection, threshold selection, or ensemble configuration.
- Isolation Forest is trained only on class 1 normal samples from the inner training split.

## Data Distribution

| Split | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Official train | 4427 | 2683 | 149 | 306 | 35 | 7600 |
| Official test | 1119 | 674 | 33 | 63 | 11 | 1900 |

## Validation Metrics Used For Selection

| Model | Accuracy | Macro-F1 | Balanced accuracy | Status |
|---|---:|---:|---:|---|
| xgboost validation | 0.9855 | 0.8856 | 0.8505 | ok |
| isolation forest validation | 0.9375 | 0.9367 | 0.9445 | ok |
| inception validation | 0.8987 | 0.6118 | 0.7589 | ok |
| ensemble validation | 0.9855 | 0.8909 | 0.8505 | ok |

## Final Test Metrics

| Model | Accuracy | Macro-F1 | Balanced accuracy | Status |
|---|---:|---:|---:|---|
| xgboost test | 0.9863 | 0.9092 | 0.8840 | ok |
| isolation forest test | 0.9363 | 0.9353 | 0.9440 | ok |
| inception test | 0.9021 | 0.6078 | 0.7509 | ok |
| ensemble test | 0.9853 | 0.8971 | 0.8834 | ok |

## Ensemble Selection

- Ensemble type: `entropy_weighted_three_model`
- Selection metric: `validation macro_f1, tie-breaker balanced_accuracy`
- Base weights: XGBoost `0.50`, InceptionTime `0.25`, Isolation Forest `0.25`
- Entropy epsilon: `0.05`
- Contribution policy: `all models have positive base weights and entropy-adjusted per-sample weights`
- Test set used for ensemble selection: `false`

## Ensemble Diagnostics

| Diagnostic | XGBoost | InceptionTime | Isolation Forest |
|---|---:|---:|---:|
| Base weight | 0.5000 | 0.2500 | 0.2500 |
| Mean test entropy | 0.0354 | 0.4672 | 0.3063 |
| Mean test dynamic weight | 0.6112 | 0.1702 | 0.2186 |

## InceptionTime Training Notes

- Runtime device: `Tesla P100-PCIE-16GB`
- Epochs requested: `40`
- Early stopping patience: `8`
- Batch size: `128`
- Class-weighted loss: `true`
- Scaler fit scope: `inner training split only`
- Test set used during training: `false`

## Isolation Forest Notes

- Anomaly recall: `0.9872`
- Normal specificity: `0.9008`
- Binary macro-F1: `0.9353`

## Per-Class Recall On Test

| Model | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|---|---:|---:|---:|---:|---:|
| xgboost test | 0.9946 | 0.9955 | 1.0000 | 0.7937 | 0.6364 |
| inception test | 0.9410 | 0.8798 | 0.8485 | 0.5397 | 0.5455 |
| ensemble test | 0.9946 | 0.9926 | 1.0000 | 0.7937 | 0.6364 |

## Limitations

- ECG5000 is small for classes 3 and 5, so minority-class metrics can move substantially with a few examples.
- Isolation Forest is anomaly-oriented and does not identify the exact abnormal subtype by itself.
- The formula ensemble requires `models/inception_cpu.pkl` and fastai/tsai for CPU inference.
- Metrics should be interpreted as project evidence, not clinical performance claims.

## Reproducibility

```powershell
python -m src.train_xgboost
python -m src.train_isolation_forest
python -m src.train_ensemble
python -m src.evaluate
python app.py
```
