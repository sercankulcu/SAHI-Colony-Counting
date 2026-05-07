"""Train a YOLO detection model.

This script provides a reproducible training entry point for YOLOv5n,
YOLOv8n, YOLOv11n, or other Ultralytics-compatible models.
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO object detection model.")
    parser.add_argument("--weights", type=str, default="yolov11n.pt", help="Pretrained model weights.")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to the YOLO dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--name", type=str, default="yolo_experiment", help="Experiment name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
