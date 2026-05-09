#!/usr/bin/env python3
"""Train a YOLO model with Ultralytics."""
from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yolo11n.pt', help='Model weights, e.g. yolov5n.pt, yolov8n.pt, yolo11n.pt.')
    parser.add_argument('--data', default='configs/data.yaml', help='YOLO data.yaml path.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0', help='CUDA device index or cpu.')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='colony_yolo')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        exist_ok=True,
    )


if __name__ == '__main__':
    main()
