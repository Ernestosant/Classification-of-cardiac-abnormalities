# Model Results

This page summarizes the current experiment results. The generated source
report is [`../../reports/model_results.md`](../../reports/model_results.md),
and machine-readable metrics are stored in
[`../../reports/metrics_all_test.json`](../../reports/metrics_all_test.json).

## Final Test Metrics

| Model | Accuracy | Macro-F1 | Balanced accuracy | Interpretation |
|---|---:|---:|---:|---|
| XGBoost | 0.9863 | 0.9092 | 0.8840 | Best supervised model |
| InceptionTime | 0.9021 | 0.6078 | 0.7509 | Useful separate neural baseline |
| Isolation Forest | 0.9363 | 0.9353 | 0.9440 | Binary anomaly detector |
| Ensemble | 0.9863 | 0.9092 | 0.8840 | Validation selected XGBoost only |

## Per-Class Recall On Test

| Model | Normal | PVC | Supraventricular premature beat | Ectopic beat | Unknown abnormal pathology |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.9946 | 0.9955 | 1.0000 | 0.7937 | 0.6364 |
| InceptionTime | 0.9410 | 0.8798 | 0.8485 | 0.5397 | 0.5455 |
| Ensemble | 0.9946 | 0.9955 | 1.0000 | 0.7937 | 0.6364 |

## Isolation Forest Binary Metrics

| Metric | Value |
|---|---:|
| Anomaly recall | 0.9872 |
| Normal specificity | 0.9008 |
| Binary macro-F1 | 0.9353 |

## Confusion Matrices

| Model | Validation | Test |
|---|---|---|
| XGBoost | [PNG](../../reports/confusion_matrix_xgboost_validation.png) | [PNG](../../reports/confusion_matrix_xgboost_test.png) |
| InceptionTime | [PNG](../../reports/confusion_matrix_inception_validation.png) | [PNG](../../reports/confusion_matrix_inception_test.png) |
| Isolation Forest | [PNG](../../reports/confusion_matrix_isolation_forest_validation.png) | [PNG](../../reports/confusion_matrix_isolation_forest_test.png) |
| Ensemble | [PNG](../../reports/confusion_matrix_ensemble_validation.png) | [PNG](../../reports/confusion_matrix_ensemble_test.png) |

## Interpretation

The strongest final five-class result comes from XGBoost, and the ensemble
matches it because validation selected XGBoost weight `1.00`. InceptionTime is
still valuable as a documented neural baseline, especially because it was
trained under a strict protocol and exported for CPU inference.

Minority-class results should be interpreted carefully. Class 5 has only 11
samples in the official test split, so one or two errors can move recall
substantially.

