# Project Overview

This project studies heartbeat classification on the ECG5000 dataset using a
small but carefully controlled machine learning pipeline. The goal is to
classify each heartbeat segment into one of five classes while documenting the
experimental protocol clearly enough for review and reproduction.

The repository is designed as a research portfolio project rather than a
clinical product. It includes model training scripts, saved artifacts, generated
reports, a simple CPU-only inference path, and a minimal Gradio interface.

## Research Motivation

ECG classification is a useful setting for demonstrating responsible model
development because it combines time-series structure, class imbalance, and a
high-risk domain. A strong project in this space should not only report headline
accuracy; it should show how data leakage was avoided, how minority classes were
handled, and how final results were selected.

This project therefore compares three complementary modeling approaches:

| Approach | Role |
|---|---|
| InceptionTime | Deep time-series classifier trained on GPU |
| XGBoost | Strong tabular baseline on scaled heartbeat samples |
| Isolation Forest | Binary anomaly signal trained only on normal beats |
| Ensemble | Validation-selected combination of available model signals |

## Dataset

ECG5000 contains one heartbeat segment per row. Each row has one label and 140
time samples.

| Label | Meaning |
|---:|---|
| 1 | Normal |
| 2 | Premature ventricular contraction |
| 3 | Premature supraventricular contraction |
| 4 | Ectopic beat |
| 5 | Unknown abnormal pathology |

The repository includes the official train and test CSV files in `dataset/`.
The official test split is reserved for final evaluation only.

## Task Definition

The primary task is five-class classification. The final supervised output is a
class label from 1 to 5, with class names reported in the inference outputs.

Isolation Forest is evaluated as a separate binary task:

| Binary label | ECG5000 classes |
|---|---|
| Normal | Class 1 |
| Anomaly | Classes 2, 3, 4, and 5 |

The anomaly detector is also converted into a calibrated five-class
pseudo-probability for the entropy-weighted ensemble. Its calibration and
contribution are selected on validation data only.

## What This Project Demonstrates

- Reproducible data splitting and artifact management.
- Explicit anti-leakage controls.
- Honest reporting of model strengths and weaknesses.
- GPU-assisted InceptionTime training with CPU-only inference.
- A simple inference interface that loads artifacts without fitting anything.

## Boundaries

This is not a diagnostic system. ECG5000 is a benchmark dataset, not a
deployment validation dataset. The project should be interpreted as evidence of
research engineering discipline, not as evidence of clinical readiness.
