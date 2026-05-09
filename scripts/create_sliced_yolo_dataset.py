#!/usr/bin/env python3
"""Create 640x640 overlapping YOLO tiles while preserving split structure.

The script keeps only tiles that contain at least one object. An object is assigned
into a tile when its bounding-box center falls inside the tile. Bounding boxes are
clipped to the tile and re-normalized to tile coordinates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    classes, boxes = [], []
    if not label_path.exists():
        return classes, boxes
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        classes.append(int(float(parts[0])))
        boxes.append([float(x) for x in parts[1:5]])
    return classes, boxes


def save_yolo_tile(img, boxes, classes, x_min, y_min, x_max, y_max, out_img: Path, out_lbl: Path) -> bool:
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    img_h, img_w = img.shape[:2]

    new_items: list[tuple[int, float, float, float, float]] = []
    for cls, (xc, yc, bw, bh) in zip(classes, boxes):
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if not (x_min <= center_x < x_max and y_min <= center_y < y_max):
            continue

        nx1 = max(x_min, x1) - x_min
        ny1 = max(y_min, y1) - y_min
        nx2 = min(x_max, x2) - x_min
        ny2 = min(y_max, y2) - y_min
        if nx2 <= nx1 or ny2 <= ny1:
            continue

        nbw = nx2 - nx1
        nbh = ny2 - ny1
        nxc = nx1 + nbw / 2
        nyc = ny1 + nbh / 2
        new_items.append((cls, nxc / crop_w, nyc / crop_h, nbw / crop_w, nbh / crop_h))

    if not new_items:
        return False

    crop = img[y_min:y_max, x_min:x_max]
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), crop)
    with out_lbl.open('w', encoding='utf-8') as f:
        for cls, xc, yc, bw, bh in new_items:
            f.write(f'{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
    return True


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def process_split(images_dir: Path, labels_dir: Path, output_dir: Path, split: str, tile_size: int, overlap: float) -> tuple[int, int]:
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError('Overlap is too large; stride must be positive.')

    source_count = 0
    tile_count = 0
    for image_path in tqdm(image_files, desc=f'Slicing {split}'):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f'Warning: failed to read {image_path}')
            continue
        classes, boxes = read_yolo_labels(labels_dir / f'{image_path.stem}.txt')
        if not boxes:
            continue
        h, w = img.shape[:2]
        idx = 0
        for y in tile_positions(h, tile_size, stride):
            for x in tile_positions(w, tile_size, stride):
                out_stem = f'{image_path.stem}_slice_{idx:04d}'
                saved = save_yolo_tile(
                    img, boxes, classes,
                    x, y, min(x + tile_size, w), min(y + tile_size, h),
                    output_dir / 'images' / split / f'{out_stem}.jpg',
                    output_dir / 'labels' / split / f'{out_stem}.txt',
                )
                tile_count += int(saved)
                idx += 1
        source_count += 1
    return source_count, tile_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True, help='Root with images/train,val,test and labels/train,val,test.')
    parser.add_argument('--output-root', type=Path, required=True, help='Output tiled dataset root.')
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    for split in args.splits:
        images_dir = args.input_root / 'images' / split
        labels_dir = args.input_root / 'labels' / split
        if not images_dir.exists():
            print(f'Skip missing split: {split}')
            continue
        source_count, tile_count = process_split(images_dir, labels_dir, args.output_root, split, args.tile_size, args.overlap)
        print(f'{split}: {source_count} source images -> {tile_count} labeled tiles')


if __name__ == '__main__':
    main()
