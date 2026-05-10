# InceptionTime

InceptionTime is the neural time-series model in this project. It is trained in
a GPU runtime and exported for local CPU-only inference.

## Training Runtime

The preferred retraining path uses Kaggle GPU through
`scripts/run_kaggle_inception.py` and the script in `kaggle_inception/`.

Current documented training metadata:

| Setting | Value |
|---|---|
| Runtime device | Tesla P100-PCIE-16GB |
| Epochs requested | 40 |
| Early stopping patience | 8 |
| Batch size | 128 |
| Learning rate | 0.003 |
| Class-weighted loss | true |
| Test set used during training | false |

## Data Policy

The Kaggle script downloads or reads the ECG5000 training data, creates the same
stratified inner split, fits the scaler only on the inner training fold, and
uses validation for early stopping and reporting. The official test split is not
used during training.

## Exported Artifacts

| Artifact | Purpose |
|---|---|
| `models/inception_cpu.pkl` | Exported learner used by CPU-only inference |
| `models/best_inception_time.pth` | Best training checkpoint |
| `reports/metrics_inception_validation.json` | Validation metrics from training |
| `reports/confusion_matrix_inception_validation.png` | Validation confusion matrix |

## Results

| Split | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Validation | 0.8987 | 0.6118 | 0.7589 |
| Test | 0.9021 | 0.6078 | 0.7509 |

InceptionTime improved recall on some minority abnormal classes, but at the cost
of lower precision and lower macro-F1 than XGBoost. In the current final
ensemble, InceptionTime contributes through a positive base weight and a
per-sample entropy-adjusted dynamic weight.

## CPU-Only Loading

The inference adapter sets `CUDA_VISIBLE_DEVICES` to disable GPU use and loads
the exported FastAI/tsai learner on CPU. The adapter also handles the Windows
path compatibility issue that can occur when loading a learner exported from a
Linux Kaggle environment.
