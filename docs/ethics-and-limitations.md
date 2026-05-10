# Ethics And Limitations

This repository is a research and education project. It is not a clinically
validated diagnostic tool and must not be used for medical decision-making.

## Clinical Limitations

The project uses ECG5000, a benchmark dataset. Benchmark performance does not
establish safety, fairness, robustness, or clinical value in real-world
settings.

Before any clinical use, a system would require:

- Larger and more representative datasets.
- External validation across acquisition devices and patient populations.
- Clinical review of false positives and false negatives.
- Regulatory, privacy, and safety evaluation.
- Prospective testing under realistic workflows.

## Dataset Limitations

The ECG5000 class distribution is imbalanced. The smallest classes have few
examples, especially in the official test split.

| Class | Test support |
|---|---:|
| Normal | 1119 |
| Premature ventricular contraction | 674 |
| Premature supraventricular contraction | 33 |
| Ectopic beat | 63 |
| Unknown abnormal pathology | 11 |

Minority-class metrics can move substantially when only a few predictions
change. Macro-F1 and balanced accuracy are therefore reported alongside
accuracy.

## Modeling Limitations

- XGBoost is strong on this benchmark but may not generalize to new ECG sources.
- InceptionTime was trained with class weighting, but minority precision remains
  limited.
- Isolation Forest detects abnormality as a binary signal and does not classify
  abnormal subtype.
- The final ensemble reflects validation performance, not a theoretical claim
  that one model family is always better.

## Responsible Interpretation

The most important result is not the headline accuracy. The important claim is
that the reported metrics come from a documented protocol that avoids obvious
data leakage and separates model selection from final test evaluation.

