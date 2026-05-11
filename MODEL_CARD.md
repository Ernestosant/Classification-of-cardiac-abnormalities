# Model Card: ECG5000 Ensemble Classifier

## Model Name

ECG5000 entropy-weighted ensemble classifier.

## Task

Five-class classification of ECG5000 heartbeat signal patterns. Each input row
contains 140 time samples from one heartbeat segment.

## Intended Use

This model is intended for research, education, and portfolio demonstration. It
can be used to reproduce a benchmark ECG classification experiment and to test
batch or Gradio inference on ECG5000-format CSV files.

This project is not clinically validated and must not be used for medical
diagnosis, patient management, triage, or treatment decisions.

## Dataset

The project uses ECG5000:

| Split | Samples |
|---|---:|
| Official train | 7,600 |
| Official test | 1,900 |

The official train split is divided into inner train and validation folds for
preprocessing, training, threshold selection, early stopping, and ensemble
selection. The official test split is reserved for final evaluation only.

## Inputs

Inference accepts CSV files with either:

| Shape | Behavior |
|---|---|
| 140 columns | Treated as feature-only heartbeat samples |
| 141 columns | First column is treated as an ECG5000 label and ignored |

All values must be numeric and finite.

## Outputs

The batch inference path returns:

- Final ensemble class and label.
- Ensemble confidence.
- XGBoost and InceptionTime predictions.
- Calibrated Isolation Forest normal/anomaly probabilities.
- Entropy and dynamic ensemble weight diagnostics.

The Gradio demo shows a compact reviewer-facing subset of the same predictions.

## Training Setup

The final ensemble combines:

| Component | Training or selection policy |
|---|---|
| XGBoost | Class-aware sample weights from inner training labels |
| InceptionTime | Kaggle GPU training with class-weighted loss |
| Isolation Forest | Trained only on normal beats from the inner training fold |
| Ensemble | Positive base weights selected on validation macro-F1 |

The scaler is fitted only on the inner training fold and reused for validation,
test, and inference.

## Metrics

Final metrics on the official ECG5000 test split:

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| XGBoost | 0.9863 | 0.9092 | 0.8840 |
| InceptionTime | 0.9021 | 0.6078 | 0.7509 |
| Isolation Forest | 0.9363 | 0.9353 | 0.9440 |
| Ensemble | 0.9853 | 0.8971 | 0.8834 |

The ensemble's class 5 recall is 0.6364 on 11 official test samples, so that
metric should be interpreted cautiously.

## Limitations

- ECG5000 is a benchmark dataset and is not sufficient for real-world clinical
  validation.
- Minority classes have very small support, especially class 5.
- InceptionTime CPU inference is slower than XGBoost-only inference.
- Isolation Forest provides an anomaly signal, not a specific abnormal subtype.
- The project does not evaluate fairness, device robustness, noise robustness,
  or external generalization across patient populations.

## Ethical And Clinical Disclaimer

This model is a research/educational prototype. It is not clinically validated
and must not be used for medical diagnosis or patient management.
