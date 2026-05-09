#!/usr/bin/env python3
"""Run SAHI inference over a folder for several overlap ratios and record detection counts/timing.

For full mAP evaluation of tiled predictions, export predictions and evaluate with your chosen
COCO/YOLO evaluation pipeline. This script is intended for reproducible timing and count checks.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--model-type', default='yolo11', choices=['yolov5', 'yolov8', 'yolo11'])
    parser.add_argument('--overlaps', nargs='+', type=float, default=[0.10, 0.15, 0.20, 0.25])
    parser.add_argument('--slice-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-csv', type=Path, default=Path('outputs/overlap_sensitivity.csv'))
    args = parser.parse_args()

    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f'No images found in {args.image_dir}')

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    rows = []
    for overlap in args.overlaps:
        for image_path in images:
            start = time.perf_counter()
            result = get_sliced_prediction(
                str(image_path), detection_model,
                slice_height=args.slice_size, slice_width=args.slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            rows.append({
                'image': image_path.name,
                'overlap': overlap,
                'detections': len(result.object_prediction_list),
                'time_ms': round(elapsed_ms, 3),
            })
            print(rows[-1])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'overlap', 'detections', 'time_ms'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {args.output_csv}')


if __name__ == '__main__':
    main()
