#!/usr/bin/env python3
"""Plot Precision, Recall, mAP@0.5, and mAP@0.5:0.95 from Ultralytics results.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/validation_metrics.png'))
    parser.add_argument('--title', default='YOLOv11n - Validation Metrics over 50 Epochs')
    args = parser.parse_args()

    df = clean_columns(pd.read_csv(args.results_csv))
    required = ['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in results.csv: {missing}')

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision (P)', linewidth=3, marker='o', markevery=5)
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall (R)', linewidth=3, marker='s', markevery=5)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=3, marker='^', markevery=5)
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=4, marker='D', markevery=5)
    plt.title(args.title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=500, bbox_inches='tight', facecolor='white')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
