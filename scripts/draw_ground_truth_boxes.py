"""Draw YOLO-format ground-truth bounding boxes on an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def draw_yolo_ground_truth(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    image_height, image_width = image.shape[:2]

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 5:
                continue
            _, center_x, center_y, box_width, box_height = map(float, values)

            center_x_px = int(center_x * image_width)
            center_y_px = int(center_y * image_height)
            box_width_px = int(box_width * image_width)
            box_height_px = int(box_height * image_height)

            x1 = int(center_x_px - box_width_px / 2)
            y1 = int(center_y_px - box_height_px / 2)
            x2 = int(center_x_px + box_width_px / 2)
            y2 = int(center_y_px + box_height_px / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Ground-truth visualization saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO ground-truth boxes.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("results/ground_truth_boxes.jpg"))
    parser.add_argument("--thickness", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_yolo_ground_truth(args.image_path, args.label_path, args.output_path, thickness=args.thickness)


if __name__ == "__main__":
    main()
