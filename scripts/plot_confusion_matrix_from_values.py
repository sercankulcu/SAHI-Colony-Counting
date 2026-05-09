#!/usr/bin/env python3
"""Reproduce the normalized confusion-matrix figure from a CSV matrix file."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CLASS_NAMES = [
    'Actinobacillus equuli', 'Actinobacillus pleuropneumoniae', 'Aeromonas hydrophila',
    'Bacillus cereus', 'Bibersteinia trehalosi', 'Bordetella bronchiseptica',
    'Brucella ovis', 'Clostridium perfringens', 'Corynebacterium pseudotuberculosis',
    'Erysipelothrix rhusiopathiae', 'Escherichia coli', 'Glaesserella parasuis',
    'Klebsiella pneumoniae', 'Listeria monocytogenes', 'Paenibacillus larvae',
    'Pasteurella multocida', 'Proteus mirabilis', 'Pseudomonas aeruginosa',
    'Rhodococcus equi', 'Salmonella enterica', 'Staphylococcus aureus',
    'Staphylococcus hyicus', 'Streptococcus agalactiae', 'Trueperella pyogenes', 'background'
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-csv', type=Path, required=True, help='25x25 normalized matrix CSV without header, or with index/header.')
    parser.add_argument('--output', type=Path, default=Path('outputs/confusion_matrix.png'))
    parser.add_argument('--no-colorbar', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.matrix_csv, header=None)
    cm = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='all').dropna(axis=1, how='all').to_numpy(dtype=float)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f'Matrix must be square. Got {cm.shape}')
    names = DEFAULT_CLASS_NAMES[:cm.shape[0]]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest')
    if not args.no_colorbar:
        fig.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix Normalized', fontsize=18, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=5)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
