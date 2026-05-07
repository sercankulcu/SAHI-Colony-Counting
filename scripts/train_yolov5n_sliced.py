"""Train YOLOv5n on the sliced YOLO dataset."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov5n.pt")
    model.train(
        data="configs/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        name="yolov5n_sliced_dataset",
    )


if __name__ == "__main__":
    main()
