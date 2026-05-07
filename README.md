# YOLO and SAHI-Based Object Detection Pipeline

This repository contains the source code used for YOLO-based object detection, SAHI-based tiled inference, visual evaluation, and result visualization. The scripts are organized to support reproducible model training, inference, and figure generation.

## Repository Structure

```text
.
├── configs/
│   └── dataset.yaml
├── scripts/
│   ├── train_yolo_model.py
│   ├── train_yolov5n_sliced.py
│   ├── train_yolov5n_original.py
│   ├── create_sliced_yolo_dataset.py
│   ├── run_sahi_inference.py
│   ├── visualize_sahi_bboxes.py
│   ├── draw_ground_truth_boxes.py
│   ├── create_detection_heatmap.py
│   ├── plot_validation_metrics.py
│   └── plot_confusion_matrix.py
├── legacy_detectron2/
│   ├── train_detectron2_colonies.py
│   └── inference_detectron2_colonies.py
└── results/
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Format

The YOLO dataset should be organized as follows:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Update `configs/dataset.yaml` according to the dataset location and class names.

## Training

Train a YOLO model using the generic training script:

```bash
python scripts/train_yolo_model.py --weights yolov11n.pt --data configs/dataset.yaml --epochs 50 --imgsz 640 --batch 4 --device 0 --name yolov11n_sahi_pipeline
```

For the sliced dataset setup:

```bash
python scripts/train_yolov5n_sliced.py
```

For the original-image setup:

```bash
python scripts/train_yolov5n_original.py
```

## SAHI-Based Tiled Inference

```bash
python scripts/run_sahi_inference.py --model-path runs/detect/yolov11n_sahi_pipeline/weights/best.pt --model-type yolo11 --image-path examples/sample_image.jpg --output-dir results/sahi_predictions
```

## Heatmap Generation

```bash
python scripts/create_detection_heatmap.py --model-path runs/detect/yolov11n_sahi_pipeline/weights/best.pt --model-type yolo11 --image-path examples/sample_image.jpg --output-path results/detection_heatmap.jpg
```

## Notes

The Detectron2 scripts are kept in `legacy_detectron2/` only for archival purposes. They should not be used as the main implementation if the related article describes a YOLO and SAHI-based pipeline.
