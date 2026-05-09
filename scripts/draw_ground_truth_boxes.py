#!/usr/bin/env python3
"""Draw YOLO-format ground-truth boxes on an image."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def draw_yolo_groundtruth(image_path: Path, label_path: Path, output_path: Path, thickness: int = 4) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    h, w = img.shape[:2]

    if label_path.exists():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f'Saved: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--label', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/ground_truth.jpg'))
    parser.add_argument('--thickness', type=int, default=4)
    args = parser.parse_args()
    draw_yolo_groundtruth(args.image, args.label, args.output, args.thickness)


if __name__ == '__main__':
    main()
