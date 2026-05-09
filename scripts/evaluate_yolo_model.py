#!/usr/bin/env python3
"""Evaluate a YOLO checkpoint on val or test split."""
from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='configs/data.yaml')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--iou', type=float, default=0.7)
    args = parser.parse_args()

    model = YOLO(args.weights)
    kwargs = dict(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device, iou=args.iou)
    if args.conf is not None:
        kwargs['conf'] = args.conf
    metrics = model.val(**kwargs)
    print(metrics)


if __name__ == '__main__':
    main()
