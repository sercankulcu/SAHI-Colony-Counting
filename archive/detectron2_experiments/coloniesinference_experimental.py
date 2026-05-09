# inference.py
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import os

cfg = get_cfg()
cfg.merge_from_file("... aynı config ...")  # yukarıdaki cfg ayarlarını buraya kopyala
cfg.MODEL.WEIGHTS = "D:/colonies_project/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

img = cv2.imread("test_resmi.jpg")
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("result.jpg", out.get_image()[:, :, ::-1])