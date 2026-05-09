# train_colonies.py
import os
import json
import random
from sklearn.model_selection import train_test_split
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import wandb

# --------------------- AYARLAR ---------------------
DATASET_ROOT = "."        # BURAYI DEĞİŞTİR!
JSON_PATH = os.path.join(DATASET_ROOT, "annot_COCO.json")
IMG_DIR = os.path.join(DATASET_ROOT, "images")

# Train/val böl (%85 train, %15 val)
with open(JSON_PATH) as f:
    data = json.load(f)

images = data["images"]
random.seed(42)
train_imgs, val_imgs = train_test_split(images, test_size=0.15, random_state=42)

def save_split(img_list, ann_list, name):
    split = {
        "images": img_list,
        "annotations": [a for a in data["annotations"] if a["image_id"] in [i["id"] for i in img_list]],
        "categories": data["categories"]
    }
    with open(f"{name}.json", "w") as f:
        json.dump(split, f)

save_split(train_imgs, data["annotations"], "train_coco")
save_split(val_imgs, data["annotations"], "val_coco")

# Detectron2’a kayıt et
register_coco_instances("colonies_train", {}, "train_coco.json", IMG_DIR)
register_coco_instances("colonies_val", {}, "val_coco.json", IMG_DIR)

# --------------------- EN İYİ CONFIG (370 fotoğraf için) ---------------------
cfg = get_cfg()

# En iyi iki seçenekten birini seç (ikincisi biraz daha iyi ama yavaş)
# 1) X101-FPN (hızlı ve çok iyi)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"

# 2) Swin-L (2025’te en yüksek doğruluk – VRAM yetiyorsa bunu kullan)
# cfg.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_swin_l_fpn_3x_coco.yaml"))
# cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/swin/mask_rcnn_swin_l_fpn_3x/123456789/model_final.pth"

cfg.DATASETS.TRAIN = ("colonies_train",)
cfg.DATASETS.TEST = ("colonies_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 4        # 24 GB VRAM → 6, 12 GB → 3, 8 GB → 2 yap
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 9000          # 370 fotoğrafla ~30 epoch
cfg.SOLVER.STEPS = (7000, 8500)
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.GAMMA = 0.1

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9   # senin json’unda 9 sınıf var
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

# Küçük koloniler için çok kritik!
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4, 8, 16, 32, 64]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

cfg.TEST.DETECTIONS_PER_IMAGE = 1500
cfg.OUTPUT_DIR = os.path.join(DATASET_ROOT, "output")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Eğitim başlıyor... Çay/kahve hazırla, 6–12 saat sürecek")
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Eğitim bitti! En iyi model:")
print(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))