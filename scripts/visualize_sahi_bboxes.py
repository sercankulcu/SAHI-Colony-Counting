"""Draw class-colored bounding boxes from SAHI predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DEFAULT_CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (255, 0, 255),
    4: (0, 165, 255),
    5: (255, 105, 180),
    6: (255, 255, 255),
    7: (128, 255, 0),
    8: (255, 0, 128),
    9: (0, 255, 128),
    10: (180, 105, 255),
    11: (255, 215, 0),
    12: (0, 128, 255),
    13: (255, 192, 203),
    14: (50, 205, 50),
    15: (255, 165, 0),
    16: (255, 20, 147),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAHI predictions with bounding boxes only.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="yolov5", choices=["yolov5", "yolov8", "yolo11"])
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("results/sahi_colored_bboxes.png"))
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--box-thickness", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        x1, y1 = int(bbox.minx), int(bbox.miny)
        x2, y2 = int(bbox.maxx), int(bbox.maxy)
        class_id = int(prediction.category.id)
        color = DEFAULT_CLASS_COLORS.get(class_id, (0, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=args.box_thickness)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 18))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Detected objects: {len(result.object_prediction_list)}")
    print(f"Bounding-box visualization saved to: {args.output_path}")


if __name__ == "__main__":
    main()
