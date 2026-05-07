"""Train YOLOv5n on the original-image YOLO dataset."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov5n.pt")
    model.train(
        data="configs/dataset_original.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        name="yolov5n_original_dataset",
    )


if __name__ == "__main__":
    main()
