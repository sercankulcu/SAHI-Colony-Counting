#!/usr/bin/env python3
"""Create a detection-density heatmap from SAHI tiled predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-type', default='yolo11', choices=['yolov5', 'yolov8', 'yolo11'])
    parser.add_argument('--slice-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--blur', type=int, default=101)
    parser.add_argument('--output', type=Path, default=Path('outputs/heatmap.jpg'))
    args = parser.parse_args()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )
    result = get_sliced_prediction(
        args.image,
        detection_model,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
    )

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)

    for pred in result.object_prediction_list:
        bbox = pred.bbox
        xc = int((bbox.minx + bbox.maxx) / 2)
        yc = int((bbox.miny + bbox.maxy) / 2)
        if 0 <= xc < width and 0 <= yc < height:
            mask[yc, xc] += 1.0

    blur = make_odd(args.blur)
    heatmap = cv2.GaussianBlur(mask, (blur, blur), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    final_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), final_img)
    print(f'Detections: {len(result.object_prediction_list)}')
    print(f'Heatmap saved to: {args.output}')


if __name__ == '__main__':
    main()
