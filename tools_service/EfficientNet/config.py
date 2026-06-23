import os

import torch


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.join(_CURRENT_DIR, "data")


class Config:
    DATA_ROOT = os.getenv("EFFICIENTNET_DATA_ROOT", _DEFAULT_DATA_ROOT)
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    SAVE_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    BEST_MODEL = os.path.join(SAVE_DIR, "best_model.pth")
    LAST_MODEL = os.path.join(SAVE_DIR, "last_model.pth")

    MODEL_NAME = "efficientnet_v2_s"
    NUM_CLASSES = 50
    PRETRAINED = True
    FEATURE_EXTRACT = False

    EPOCHS = 30
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    IMG_SIZE = 384

    LR_BACKBONE = 2e-4
    LR_HEAD = 2e-3
    WEIGHT_DECAY = 1e-4
    ETA_MIN = 1e-6

    FEW_SHOT_THRESHOLD = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 42
    EARLY_STOP = 8


cfg = Config()

os.makedirs(cfg.SAVE_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)
