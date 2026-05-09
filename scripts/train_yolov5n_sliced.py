#!/usr/bin/env python3
"""Convenience command for YOLOv5n training on the sliced/tiled dataset."""
from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO('yolov5n.pt')
    model.train(data='configs/data.yaml', epochs=50, imgsz=640, batch=4, device=0, name='yolov5n_tiled', exist_ok=True)


if __name__ == '__main__':
    main()
