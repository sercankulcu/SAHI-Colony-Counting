"""Legacy Detectron2 inference script for COCO-format colony detection."""

from __future__ import annotations

from pathlib import Path

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

CONFIG_PATH = Path("config.yaml")
MODEL_WEIGHTS = Path("output_detectron2/model_final.pth")
INPUT_IMAGE = Path("examples/sample_image.jpg")
OUTPUT_IMAGE = Path("results/detectron2_prediction.jpg")
SCORE_THRESHOLD = 0.50


def main() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG_PATH))
    cfg.MODEL.WEIGHTS = str(MODEL_WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    predictor = DefaultPredictor(cfg)
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {INPUT_IMAGE}")

    outputs = predictor(image)
    visualizer = Visualizer(image[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), output.get_image()[:, :, ::-1])
    print(f"Prediction saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
