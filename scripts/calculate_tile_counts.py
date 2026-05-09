#!/usr/bin/env python3
"""Calculate expected tile counts for several overlap ratios."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def calculate_tile_count(width: int, height: int, tile_size: int, overlap_ratio: float) -> tuple[int, int, int, int]:
    stride = int(tile_size * (1 - overlap_ratio))
    if stride <= 0:
        raise ValueError('Stride must be greater than zero.')
    tiles_x = 1 if width <= tile_size else math.ceil((width - tile_size) / stride) + 1
    tiles_y = 1 if height <= tile_size else math.ceil((height - tile_size) / stride) + 1
    return tiles_x, tiles_y, tiles_x * tiles_y, stride


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlaps', nargs='+', type=float, default=[0.10, 0.15, 0.20, 0.25])
    parser.add_argument('--output-csv', type=Path, default=Path('outputs/tile_count_summary.csv'))
    args = parser.parse_args()

    rows = []
    for image_path in sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        with Image.open(image_path) as img:
            width, height = img.size
        row = {'image_name': image_path.name, 'width': width, 'height': height}
        for overlap in args.overlaps:
            tiles_x, tiles_y, total, stride = calculate_tile_count(width, height, args.tile_size, overlap)
            pct = int(overlap * 100)
            row[f'stride_{pct}%'] = stride
            row[f'tiles_x_{pct}%'] = tiles_x
            row[f'tiles_y_{pct}%'] = tiles_y
            row[f'total_tiles_{pct}%'] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        total_row = {'image_name': 'TOTAL', 'width': '', 'height': ''}
        for overlap in args.overlaps:
            pct = int(overlap * 100)
            total_row[f'stride_{pct}%'] = ''
            total_row[f'tiles_x_{pct}%'] = ''
            total_row[f'tiles_y_{pct}%'] = ''
            total_row[f'total_tiles_{pct}%'] = df[f'total_tiles_{pct}%'].sum()
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df)
    print(f'Saved: {args.output_csv}')


if __name__ == '__main__':
    main()
