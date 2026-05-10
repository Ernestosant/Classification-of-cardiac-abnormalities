# Ensemble

The final ensemble is an entropy-weighted three-model formula. XGBoost,
InceptionTime, and Isolation Forest all contribute to every final prediction.
The configuration is selected on validation data only and stored in
`models/ensemble_config.json`.

## Probability Sources

| Signal | Probability vector |
|---|---|
| XGBoost | Five-class `predict_proba` output |
| InceptionTime | Five-class FastAI/tsai softmax output |
| Isolation Forest | Calibrated pseudo-probability over the five ECG5000 classes |

Isolation Forest is still trained only as a normal-vs-anomaly detector. To use
it in the five-class ensemble, its decision score is converted to a calibrated
binary anomaly probability and then distributed across abnormal classes using
inner-training abnormal class priors.

The conversion is:

```text
z_if = (threshold - decision_function(x)) / scale
a_if = sigmoid(b0 + b1 * z_if)
p_if[1] = 1 - a_if
p_if[c] = a_if * pi_c, for c in {2,3,4,5}
```

The calibration coefficients `b0` and `b1` are fit on validation labels only.
The abnormal priors `pi_c` are computed from the inner training split only.

## Formula

For each model `m`, compute normalized entropy:

```text
H_m = - sum_c p_m[c] * log(p_m[c]) / log(5)
r_m = 1 - H_m
```

The dynamic per-sample weight is:

```text
w_m(x) = beta_m * (epsilon + r_m(x)) / sum_j beta_j * (epsilon + r_j(x))
```

The final class probability is:

```text
p_final = normalize(
    w_xgb * p_xgb +
    w_inc * p_inc +
    w_if  * p_if
)
```

The selected class is `argmax(p_final)`.

## Selected Configuration

| Parameter | Value |
|---|---:|
| XGBoost base weight | 0.50 |
| InceptionTime base weight | 0.25 |
| Isolation Forest base weight | 0.25 |
| Entropy epsilon | 0.05 |
| Minimum base weight during search | 0.10 |
| Grid step | 0.05 |

All base weights are constrained to be positive. The validation search still
uses macro-F1 as the primary metric, but it cannot remove a model entirely.

## Results

| Split | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Validation | 0.9855 | 0.8909 | 0.8505 |
| Test | 0.9853 | 0.8971 | 0.8834 |

The formula ensemble is slightly below XGBoost-only on the current test macro-F1
but better satisfies the research goal of a transparent multi-model ensemble.
The cost is documented rather than hidden.
