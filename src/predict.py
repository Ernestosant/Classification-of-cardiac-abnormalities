from __future__ import annotations

import argparse

from .inference import predict_file_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only ECG5000 ensemble inference on a CSV file.")
    parser.add_argument("--input", required=True, help="CSV with 140 feature columns or 141 columns including label")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    parser.add_argument(
        "--include-inception",
        action="store_true",
        help="Deprecated no-op. InceptionTime is always used by the formula ensemble.",
    )
    args = parser.parse_args()

    predictions, notes = predict_file_to_dataframe(args.input)
    predictions.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    for note in notes:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
