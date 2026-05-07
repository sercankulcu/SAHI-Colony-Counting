"""Run SAHI-based tiled inference on a high-resolution image."""

from __future__ import annotations

import argparse
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAHI tiled inference.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained YOLO model weights.")
    parser.add_argument("--model-type", type=str, default="yolo11", choices=["yolov5", "yolov8", "yolo11"], help="SAHI model type.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sahi_predictions"), help="Output directory.")
    parser.add_argument("--slice-size", type=int, default=640, help="Slice height and width.")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio.")
    parser.add_argument("--confidence", type=float, default=0.50, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    result.export_visuals(export_dir=str(args.output_dir), file_name="sahi_prediction")
    print(f"Prediction completed. Detected objects: {len(result.object_prediction_list)}")
    print(f"Visual output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
