"""Legacy Detectron2 Mask R-CNN training script for COCO-format annotations.

This file is kept for archival purposes. It is not part of the main YOLO and
SAHI-based pipeline unless the related study explicitly reports Detectron2
experiments.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from sklearn.model_selection import train_test_split

DATASET_ROOT = Path(".")
ANNOTATION_PATH = DATASET_ROOT / "annotations_coco.json"
IMAGE_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = DATASET_ROOT / "output_detectron2"


def save_coco_split(data: dict, image_list: list[dict], output_path: Path) -> None:
    image_ids = {image["id"] for image in image_list}
    split = {
        "images": image_list,
        "annotations": [annotation for annotation in data["annotations"] if annotation["image_id"] in image_ids],
        "categories": data["categories"],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(split, file)


def main() -> None:
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    random.seed(42)
    train_images, validation_images = train_test_split(data["images"], test_size=0.15, random_state=42)

    train_annotation_path = DATASET_ROOT / "train_coco.json"
    validation_annotation_path = DATASET_ROOT / "validation_coco.json"
    save_coco_split(data, train_images, train_annotation_path)
    save_coco_split(data, validation_images, validation_annotation_path)

    register_coco_instances("colonies_train", {}, str(train_annotation_path), str(IMAGE_DIR))
    register_coco_instances("colonies_validation", {}, str(validation_annotation_path), str(IMAGE_DIR))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("colonies_train",)
    cfg.DATASETS.TEST = ("colonies_validation",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 9000
    cfg.SOLVER.STEPS = (7000, 8500)
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4, 8, 16, 32, 64]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.TEST.DETECTIONS_PER_IMAGE = 1500
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
