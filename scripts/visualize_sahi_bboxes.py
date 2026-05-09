#!/usr/bin/env python3
"""Draw only SAHI prediction bounding boxes on an image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CLASS_COLORS = {
    0: (0, 255, 255), 1: (255, 255, 0), 2: (0, 255, 0), 3: (255, 0, 255),
    4: (0, 165, 255), 5: (255, 105, 180), 6: (255, 255, 255), 7: (128, 255, 0),
    8: (255, 0, 128), 9: (0, 255, 128), 10: (180, 105, 255), 11: (255, 215, 0),
    12: (0, 128, 255), 13: (255, 192, 203), 14: (50, 205, 50), 15: (255, 165, 0),
    16: (255, 20, 147), 17: (0, 255, 255), 18: (255, 255, 255), 19: (0, 255, 255),
    20: (255, 255, 0), 21: (0, 255, 0), 22: (255, 0, 255), 23: (0, 165, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-type', default='yolo11', choices=['yolov5', 'yolov8', 'yolo11'])
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--slice-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=Path('outputs/sahi_bbox_only.jpg'))
    parser.add_argument('--thickness', type=int, default=5)
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

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        color = CLASS_COLORS.get(int(obj.category.id), (0, 255, 255))
        cv2.rectangle(image, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), color, args.thickness)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    print(f'Detections: {len(result.object_prediction_list)}')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
