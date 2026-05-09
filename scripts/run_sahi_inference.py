#!/usr/bin/env python3
"""Run SAHI tiled inference and optionally export visualization/predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def prediction_to_dict(obj) -> dict:
    bbox = obj.bbox
    return {
        'category_id': int(obj.category.id),
        'category_name': obj.category.name,
        'score': float(obj.score.value),
        'bbox_xyxy': [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, help='YOLO best.pt path.')
    parser.add_argument('--image', required=True, help='Full-resolution image path.')
    parser.add_argument('--model-type', default='yolo11', choices=['yolov5', 'yolov8', 'yolo11'])
    parser.add_argument('--slice-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sahi'))
    parser.add_argument('--name', default='sahi_result')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    result.export_visuals(export_dir=str(args.output_dir), file_name=args.name)
    json_path = args.output_dir / f'{args.name}.json'
    json_path.write_text(json.dumps([prediction_to_dict(o) for o in result.object_prediction_list], indent=2), encoding='utf-8')
    print(f'Detections: {len(result.object_prediction_list)}')
    print(f'Visualization and JSON saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
