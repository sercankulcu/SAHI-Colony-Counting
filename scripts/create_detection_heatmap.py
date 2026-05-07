"""Create a detection-density heatmap from SAHI predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a heatmap from SAHI object detections.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="yolo11", choices=["yolov5", "yolov8", "yolo11"])
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("results/detection_heatmap.jpg"))
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--blur-size", type=int, default=101, help="Gaussian blur kernel size. It must be odd.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.blur_size % 2 == 0:
        raise ValueError("The blur size must be an odd integer.")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    result = get_sliced_prediction(
        args.image_path,
        detection_model,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
    )

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {args.image_path}")

    height, width = image.shape[:2]
    density_mask = np.zeros((height, width), dtype=np.float32)

    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        x_center = int(bbox.minx + (bbox.maxx - bbox.minx) / 2)
        y_center = int(bbox.miny + (bbox.maxy - bbox.miny) / 2)
        if 0 <= x_center < width and 0 <= y_center < height:
            density_mask[y_center, x_center] += 1

    heatmap = cv2.GaussianBlur(density_mask, (args.blur_size, args.blur_size), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_normalized = np.uint8(heatmap_normalized)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_path), overlay)
    print(f"Heatmap saved to: {args.output_path}")
    print(f"Detected objects: {len(result.object_prediction_list)}")


if __name__ == "__main__":
    main()
