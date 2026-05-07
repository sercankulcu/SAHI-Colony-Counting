"""Create a sliced YOLO dataset from large annotated images.

The script crops large images into fixed-size overlapping patches and converts
YOLO-format bounding boxes to the coordinate system of each patch.
Only patches containing at least one object are saved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    classes: list[int] = []
    boxes: list[list[float]] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 5:
                continue
            classes.append(int(float(values[0])))
            boxes.append([float(value) for value in values[1:5]])

    return classes, boxes


def save_yolo_patch(
    image,
    boxes: list[list[float]],
    classes: list[int],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    output_stem: str,
    output_image_dir: Path,
    output_label_dir: Path,
) -> None:
    patch_width = x_max - x_min
    patch_height = y_max - y_min
    patch_image = image[y_min:y_max, x_min:x_max]

    image_height, image_width = image.shape[:2]
    patch_boxes: list[list[float]] = []
    patch_classes: list[int] = []

    for box, class_id in zip(boxes, classes):
        center_x, center_y, width, height = box
        x1 = (center_x - width / 2) * image_width
        y1 = (center_y - height / 2) * image_height
        x2 = (center_x + width / 2) * image_width
        y2 = (center_y + height / 2) * image_height

        box_center_x = x1 + (x2 - x1) / 2
        box_center_y = y1 + (y2 - y1) / 2

        if x_min < box_center_x < x_max and y_min < box_center_y < y_max:
            new_x1 = max(x_min, x1) - x_min
            new_y1 = max(y_min, y1) - y_min
            new_x2 = min(x_max, x2) - x_min
            new_y2 = min(y_max, y2) - y_min

            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1
            new_center_x = new_x1 + new_width / 2
            new_center_y = new_y1 + new_height / 2

            patch_boxes.append([
                new_center_x / patch_width,
                new_center_y / patch_height,
                new_width / patch_width,
                new_height / patch_height,
            ])
            patch_classes.append(class_id)

    if not patch_boxes:
        return

    output_image_path = output_image_dir / f"{output_stem}.jpg"
    output_label_path = output_label_dir / f"{output_stem}.txt"

    cv2.imwrite(str(output_image_path), patch_image)
    with output_label_path.open("w", encoding="utf-8") as file:
        for class_id, box in zip(patch_classes, patch_boxes):
            file.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice large YOLO images into overlapping patches.")
    parser.add_argument("--input-dir", type=Path, default=Path("images"), help="Directory containing images and YOLO label files.")
    parser.add_argument("--output-image-dir", type=Path, default=Path("parts/images"), help="Output image directory.")
    parser.add_argument("--output-label-dir", type=Path, default=Path("parts/labels"), help="Output label directory.")
    parser.add_argument("--slice-size", type=int, default=640, help="Patch size in pixels.")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between adjacent patches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    args.output_label_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    step = int(args.slice_size * (1 - args.overlap))
    processed_images = 0

    for image_path in sorted(args.input_dir.iterdir()):
        if image_path.suffix.lower() not in valid_extensions:
            continue

        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"Warning: label file not found for {image_path.name}; skipping image.")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: image could not be read: {image_path}")
            continue

        classes, boxes = read_yolo_labels(label_path)
        image_height, image_width = image.shape[:2]
        patch_index = 0

        for y in range(0, image_height, step):
            for x in range(0, image_width, step):
                x_max = min(x + args.slice_size, image_width)
                y_max = min(y + args.slice_size, image_height)
                x_min = max(0, x_max - args.slice_size)
                y_min = max(0, y_max - args.slice_size)

                output_stem = f"{image_path.stem}_slice_{patch_index}"
                save_yolo_patch(
                    image,
                    boxes,
                    classes,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    output_stem,
                    args.output_image_dir,
                    args.output_label_dir,
                )
                patch_index += 1

        processed_images += 1
        print(f"Processed {image_path.name}: {patch_index} candidate patches generated.")

    print(f"Completed. Processed source images: {processed_images}")


if __name__ == "__main__":
    main()
