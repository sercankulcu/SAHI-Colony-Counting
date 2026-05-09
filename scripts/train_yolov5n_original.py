#!/usr/bin/env python3
"""Convenience command for the resized full-plate YOLOv5n baseline."""
from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO('yolov5n.pt')
    model.train(data='configs/data_original.yaml', epochs=50, imgsz=640, batch=4, device=0, name='yolov5n_original', exist_ok=True)


if __name__ == '__main__':
    main()
