"""Plot validation metrics from a YOLO results CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLO validation metrics over epochs.")
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to the YOLO results.csv file.")
    parser.add_argument("--output-path", type=Path, default=Path("results/validation_metrics.png"))
    parser.add_argument("--title", type=str, default="Validation Metrics over Training Epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df.columns = [column.strip() for column in df.columns]

    required_columns = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV file: {missing_columns}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=3, marker="o", markevery=5)
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=3, marker="s", markevery=5)
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=3, marker="^", markevery=5)
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=3, marker="D", markevery=5)

    plt.title(args.title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Metric plot saved to: {args.output_path}")


if __name__ == "__main__":
    main()
