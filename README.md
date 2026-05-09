# SAHI Colony Counting Code

This package contains the cleaned code needed for a YOLO + SAHI bacterial colony counting pipeline.

## Main contents

- `configs/data.yaml`: 24-class YOLO configuration for the sliced/tiled dataset.
- `configs/data_original.yaml`: 24-class YOLO configuration for the resized full-plate baseline.
- `scripts/split_yolo_dataset_by_image.py`: splits original full-plate images into train/val/test.
- `scripts/create_sliced_yolo_dataset.py`: creates overlapping 640×640 YOLO tiles and updates annotations.
- `scripts/train_yolo_model.py`: generic YOLO training script for YOLOv5n, YOLOv8n, and YOLOv11n.
- `scripts/evaluate_yolo_model.py`: evaluates trained YOLO checkpoints on train/val/test.
- `scripts/run_sahi_inference.py`: performs SAHI-based tiled inference.
- `scripts/create_detection_heatmap.py`: creates detection-density heatmaps.
- `scripts/run_cross_validation_original_level.py`: 5-fold CV while grouping tiles by original image ID.
- `scripts/run_overlap_sensitivity.py`: overlap-ratio timing and detection-count check.

## Installation

```bash
pip install -r requirements.txt
```

## Recommended dataset structure

Original full-plate dataset before splitting:

```text
dataset/raw/images/
dataset/raw/labels/
```

After splitting:

```text
dataset/original/images/train
dataset/original/images/val
dataset/original/images/test
dataset/original/labels/train
dataset/original/labels/val
dataset/original/labels/test
```

After tiling:

```text
dataset/parts/images/train
dataset/parts/images/val
dataset/parts/images/test
dataset/parts/labels/train
dataset/parts/labels/val
dataset/parts/labels/test
```

## 1. Split the original full-plate dataset

This keeps the split at original-image level.

```bash
python scripts/split_yolo_dataset_by_image.py \
  --images-dir dataset/raw/images \
  --labels-dir dataset/raw/labels \
  --output-dir dataset/original \
  --test-ratio 0.20 \
  --val-ratio 0.16 \
  --seed 42
```

## 2. Create 640×640 overlapping tiles

```bash
python scripts/create_sliced_yolo_dataset.py \
  --input-root dataset/original \
  --output-root dataset/parts \
  --tile-size 640 \
  --overlap 0.20
```

## 3. Train YOLO models

Tiled YOLOv11n:

```bash
python scripts/train_yolo_model.py \
  --model yolo11n.pt \
  --data configs/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --name yolov11n_tiled
```

Resized full-plate baseline:

```bash
python scripts/train_yolo_model.py \
  --model yolo11n.pt \
  --data configs/data_original.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --name yolov11n_original
```

For YOLOv5n or YOLOv8n, replace `--model yolo11n.pt` with `yolov5n.pt` or `yolov8n.pt`.

## 4. Evaluate a trained model

```bash
python scripts/evaluate_yolo_model.py \
  --weights runs/detect/yolov11n_tiled/weights/best.pt \
  --data configs/data.yaml \
  --split test \
  --imgsz 640 \
  --device 0
```

## 5. Run SAHI tiled inference

```bash
python scripts/run_sahi_inference.py \
  --weights runs/detect/yolov11n_tiled/weights/best.pt \
  --image dataset/original/images/test/example.jpg \
  --model-type yolo11 \
  --slice-size 640 \
  --overlap 0.20 \
  --conf 0.50 \
  --device cuda:0 \
  --output-dir outputs/sahi \
  --name example_sahi
```

## 6. Create a heatmap

```bash
python scripts/create_detection_heatmap.py \
  --weights runs/detect/yolov11n_tiled/weights/best.pt \
  --image dataset/original/images/test/example.jpg \
  --model-type yolo11 \
  --output outputs/heatmap.jpg
```

## 7. Plot validation metrics

```bash
python scripts/plot_validation_metrics.py \
  --results-csv runs/detect/yolov11n_tiled/results.csv \
  --output outputs/yolov11n_validation_metrics.png
```

## 8. Cross-validation without tile leakage

This script groups files by original image ID. It expects tile names such as `image001_slice_0001.jpg`.

```bash
python scripts/run_cross_validation_original_level.py \
  --images-dir dataset/parts/images/train \
  --labels-dir dataset/parts/labels/train \
  --model yolo11n.pt \
  --folds 5 \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

## Notes

- The `archive/detectron2_experiments` folder contains early Detectron2 experiments. These are not part of the reported YOLO-SAHI pipeline.
- The dataset itself is not included in this ZIP.
- Update the `path:` field in YAML files if your dataset folder is in a different location.
