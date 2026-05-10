# XGBoost

XGBoost is the strongest supervised model in the current experiment. It treats
the 140 scaled ECG samples as tabular features and predicts one of the five
ECG5000 classes.

## Training Design

| Design choice | Value |
|---|---|
| Objective | `multi:softprob` |
| Model selection data | Validation split only |
| Class imbalance handling | Sample weights from inner training labels |
| Early stopping | Validation-based |
| Final artifact | `models/xgboost_model.json` |

Class weights are computed only from the inner training split. Validation labels
are used for early stopping and metric reporting, not for fitting preprocessing
or class weights.

## Results

| Split | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Validation | 0.9855 | 0.8856 | 0.8505 |
| Test | 0.9863 | 0.9092 | 0.8840 |

## Test Recall By Class

| Class | Recall |
|---|---:|
| Normal | 0.9946 |
| Premature ventricular contraction | 0.9955 |
| Premature supraventricular contraction | 1.0000 |
| Ectopic beat | 0.7937 |
| Unknown abnormal pathology | 0.6364 |

## Interpretation

The model performs strongly overall, but the smallest class remains difficult.
Class 5 has only 11 samples in the official test split, so its recall is highly
sensitive to a small number of examples.

