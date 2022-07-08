import os
import time

class FeedlaneConfig():
    ROOT_DIR = "/content/feedlane"

    DATA_DIR = os.path.join(ROOT_DIR, "data/classified_data")
    DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
    DATA_VAL_DIR = os.path.join(DATA_DIR, "val")

    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    CKPT_DIR = os.path.join(OUTPUT_DIR, "train/lightning_logs", time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    CKPT_PATH = "/content/feedlane/output/train/lightning_logs/20220708_085117/epoch=92-step=13950.ckpt"

    LABEL_DICT = {
            'empty': 0.0,
            'minimal': 1.0,
            'normal': 2.0,
            'full': 3.0,
        }
    CLASSNAMES = ['empty', 'minimal', 'normal', 'full']

    IMG_SIZE = (224,224) # (W,H)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    IMG_EXTENSIONS = ["*.jpg", "*.png", "*.bmp"]

    THRESHOLDS=[0.5,1.5,2.5]
