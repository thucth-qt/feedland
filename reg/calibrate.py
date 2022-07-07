import os, sys

import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

from validate import validate

DATA_DIR = "/content/feedlane/data/classified_data"
OUTPUT_DIR = "/content/feedlane/output"
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def calibrate():
    PREDS_DICT = validate("/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt")

    threshold_range = np.linspace(0, 1, 100, endpoint=False)

    THRESHOLDS = []
    CLASSES = ['empty', 'minimal', 'normal', 'full']

    i = 0; j = 1
    for idx in range(len(CLASSES) - 1):
        best_acc = 0
        best_thresold = 0
        for threshold in threshold_range:
            threshold = threshold + i
            preds_i = PREDS_DICT[CLASSES[i]]
            preds_j = PREDS_DICT[CLASSES[j]]
            total_i = len(preds_i)
            total_j = len(preds_j)
            correct_i =  (preds_i < threshold).sum().item()
            correct_j =  (preds_j >= threshold).sum().item()
            # acc = (correct_i + correct_j) / (total_i + total_j)
            acc_i = correct_i / total_i
            acc_j = correct_j / total_j
            acc = (2 * acc_i * acc_j) / (acc_i + acc_j)

            if acc > best_acc:
                best_acc = acc
                best_thresold = threshold
        i += 1; j+=1
        
        THRESHOLDS.append(best_thresold)

    print(THRESHOLDS)

    return THRESHOLDS

if __name__=="__main__":
    calibrate()