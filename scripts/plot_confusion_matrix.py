"""Plot a normalized confusion matrix from a CSV file.

The CSV file should contain a square matrix without headers. Class names can be
provided with a text file that includes one class name per line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_class_names(path: Path | None, class_count: int) -> list[str]:
    if path is None:
        return [f"Class {index}" for index in range(class_count)]

    with path.open("r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]

    if len(class_names) != class_count:
        raise ValueError("The number of class names must match the confusion matrix size.")
    return class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a normalized confusion matrix.")
    parser.add_argument("--matrix-path", type=Path, required=True, help="Path to a CSV confusion matrix.")
    parser.add_argument("--class-names-path", type=Path, default=None, help="Optional path to class names.")
    parser.add_argument("--output-path", type=Path, default=Path("results/confusion_matrix_normalized.png"))
    parser.add_argument("--title", type=str, default="Normalized Confusion Matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = np.loadtxt(args.matrix_path, delimiter=",")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The confusion matrix must be square.")

    class_names = read_class_names(args.class_names_path, matrix.shape[0])

    plt.figure(figsize=(14, 12))
    sns.set(font_scale=1.0)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="lightgray",
        cbar=False,
    )
    plt.title(args.title, fontsize=18, pad=30)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {args.output_path}")


if __name__ == "__main__":
    main()
