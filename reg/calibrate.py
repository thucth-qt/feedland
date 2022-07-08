import os, sys

import torch
import numpy as np

from config import FeedlaneConfig
from validate import validate

def calibrate(ckpt_path):
    PREDS_DICT = validate(ckpt_path)

    threshold_range = np.linspace(0, 1, 100, endpoint=False)
    THRESHOLDS = []
    i = 0; j = 1

    for idx in range(len(FeedlaneConfig.CLASSNAMES) - 1):
        best_acc = 0
        best_threshold = 0
        preds_i = PREDS_DICT[FeedlaneConfig.CLASSNAMES[i]]
        preds_j = PREDS_DICT[FeedlaneConfig.CLASSNAMES[j]]
        total_i = len(preds_i)
        total_j = len(preds_j)
        for threshold in threshold_range[48:52]:
            threshold = threshold + i
            
            correct_i =  (preds_i < threshold).sum().item()
            correct_j =  (preds_j >= threshold).sum().item()
            # acc = (correct_i + correct_j) / (total_i + total_j)
            acc_i = correct_i / total_i
            acc_j = correct_j / total_j
            acc = (2 * acc_i * acc_j) / (acc_i + acc_j)

            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        i += 1; j+=1
        
        THRESHOLDS.append(best_threshold)

    print(THRESHOLDS)

    return THRESHOLDS

if __name__=="__main__":
    calibrate(ckpt_path=FeedlaneConfig.CKPT_PATH)