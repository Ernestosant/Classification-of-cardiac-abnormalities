from __future__ import annotations

import os

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


def predict_csv(file_obj):
    if file_obj is None:
        return pd.DataFrame(), "Load a CSV file first."
    try:
        predictions, notes = predict_file_to_dataframe(file_obj.name)
        return predictions, "\n".join(notes) if notes else "Prediction completed."
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


demo = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(label="ECG CSV"),
    outputs=[
        gr.DataFrame(label="Predictions"),
        gr.Textbox(label="Notes", lines=4),
    ],
    title="ECG5000 Ensemble Inference",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
