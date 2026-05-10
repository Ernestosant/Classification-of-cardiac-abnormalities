# Isolation Forest

Isolation Forest is used as a normal-vs-anomaly detector. It is not a
five-class classifier by itself.

## Training Design

| Design choice | Value |
|---|---|
| Training data | Only class 1 normal beats from the inner training split |
| Positive anomaly classes | ECG5000 classes 2, 3, 4, and 5 |
| Threshold selection | Validation split only |
| Artifact | `models/isolation_forest.joblib` |
| Config | `models/isolation_forest_config.json` |

Training only on normal beats preserves the anomaly-detection framing. The
model is evaluated as a binary detector and can also provide an auxiliary
anomaly confidence signal for the ensemble.

For the final formula ensemble, this binary signal is calibrated on validation
data and converted into a five-class pseudo-probability. Class 1 receives the
normal probability, and classes 2-5 share the anomaly probability according to
abnormal class priors computed from the inner training split only.

## Decision Rule

The saved configuration records the threshold. A sample is considered anomalous
when:

```text
decision_function(sample) <= threshold
```

For reporting and inference, the project also exposes an `isolation_score` where
higher values indicate more anomalous behavior.

## Results

| Split | Accuracy | Binary macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Validation | 0.9375 | 0.9367 | 0.9445 |
| Test | 0.9363 | 0.9353 | 0.9440 |

## Test Binary Metrics

| Metric | Value |
|---|---:|
| Anomaly recall | 0.9872 |
| Normal specificity | 0.9008 |
| Binary macro-F1 | 0.9353 |

## Interpretation

Isolation Forest is useful for detecting whether a beat is abnormal, especially
when abnormal recall is important. It does not identify the exact abnormal
subtype and should not be interpreted as a replacement for the five-class
models.
