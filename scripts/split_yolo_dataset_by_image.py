#!/usr/bin/env python3
"""Split a YOLO dataset at original-image level into train/val/test folders."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_images(images_dir: Path) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f'No images found in {images_dir}')
    return images


def copy_pair(image_path: Path, labels_dir: Path, out_root: Path, split: str) -> None:
    out_img = out_root / 'images' / split / image_path.name
    out_lbl = out_root / 'labels' / split / f'{image_path.stem}.txt'
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_img)
    src_lbl = labels_dir / f'{image_path.stem}.txt'
    if src_lbl.exists():
        shutil.copy2(src_lbl, out_lbl)
    else:
        out_lbl.write_text('', encoding='utf-8')


def split_items(items: list[Path], test_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return {'train': train, 'val': val, 'test': test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True, help='Folder containing original images.')
    parser.add_argument('--labels-dir', type=Path, required=True, help='Folder containing YOLO .txt labels.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output YOLO dataset root.')
    parser.add_argument('--test-ratio', type=float, default=0.20)
    parser.add_argument('--val-ratio', type=float, default=0.16, help='Validation ratio of all original images. Use 0.16 for 64/16/20 split.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    images = list_images(args.images_dir)
    splits = split_items(images, args.test_ratio, args.val_ratio, args.seed)

    for split, split_images in splits.items():
        for image_path in split_images:
            copy_pair(image_path, args.labels_dir, args.output_dir, split)
        print(f'{split}: {len(split_images)} images')

    print(f'Dataset written to: {args.output_dir}')


if __name__ == '__main__':
    main()
