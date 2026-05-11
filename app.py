from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pandas as pd

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - user-facing dependency guard
    raise SystemExit(
        "Gradio is required to run the interface. Install dependencies with: "
        "pip install -r requirements.txt"
    ) from exc

from src.inference import predict_file_to_dataframe


EXAMPLE_INPUT = Path(__file__).resolve().parent / "examples" / "sample_input.csv"
COMPACT_COLUMNS = [
    "id",
    "ensemble_class",
    "ensemble_label",
    "ensemble_confidence",
    "isolation_anomaly_probability",
]


def compact_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the reviewer-facing subset shown in the Gradio demo."""
    if predictions.empty:
        return predictions
    summary = predictions.loc[:, COMPACT_COLUMNS].copy()
    summary["ensemble_confidence"] = summary["ensemble_confidence"].round(4)
    summary["isolation_anomaly_probability"] = summary["isolation_anomaly_probability"].round(4)
    return summary


def predict_csv(file_obj):
    if file_obj is None:
        return pd.DataFrame(), "Load a CSV file first."
    try:
        predictions, notes = predict_file_to_dataframe(file_obj.name)
        return compact_predictions(predictions), "\n".join(notes) if notes else "Prediction completed."
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


examples = [str(EXAMPLE_INPUT)] if EXAMPLE_INPUT.exists() else None

demo = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(label="ECG CSV"),
    outputs=[
        gr.DataFrame(label="Reviewer Summary"),
        gr.Textbox(label="Notes", lines=4),
    ],
    title="ECG5000 Ensemble Inference",
    examples=examples,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
