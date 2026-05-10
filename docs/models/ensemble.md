# Ensemble

The ensemble combines supervised class probabilities with an optional anomaly
adjustment from Isolation Forest. Its configuration is selected only on the
validation split and then frozen before final test evaluation.

## Candidate Signals

| Signal | Source |
|---|---|
| Supervised probabilities | XGBoost |
| Supervised probabilities | InceptionTime |
| Anomaly confidence | Isolation Forest decision score |

## Selection Rule

The selection metric is validation macro-F1, with balanced accuracy used as a
tie-breaker. The test split is not used during weight selection.

The current selected configuration is:

| Component | Selected value |
|---|---:|
| XGBoost weight | 1.00 |
| InceptionTime weight | 0.00 |
| Isolation Forest adjustment gamma | 0.00 |

The configuration is stored in `models/ensemble_config.json`.

## Why InceptionTime Has Weight Zero

InceptionTime is trained and evaluated, but the validation selection process did
not find a blend that improved macro-F1 over XGBoost alone. The final ensemble
therefore uses XGBoost probabilities only.

This is a deliberate research decision. Reporting InceptionTime separately while
excluding it from the final ensemble avoids overstating performance or tuning to
the test set.

## Results

| Split | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Validation | 0.9855 | 0.8856 | 0.8505 |
| Test | 0.9863 | 0.9092 | 0.8840 |

