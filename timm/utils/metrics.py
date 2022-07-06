""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def accuracy_reg(preds: torch.Tensor, targets: torch.Tensor, label_dict: dict, thresholds=[0.5,0.5,0.5]):  
    pass
    # assert len(thresholds) == 3
    # assert preds.dim() == 1 and targets.dim() == 1

    # size = targets.size(0)
    # running_corrects = 0

    # for idx, value in enumerate(targets):
    #     if (value == label_dict["empty"] and preds[idx] < value + thresholds[0]) or \
    #         (value == label_dict["minimal"] and preds[idx] > value - thresholds[0] and preds[idx] < value + thresholds[1]) or \
    #         (value == label_dict["normal"] and preds[idx] > value - thresholds[1] and preds[idx] < value + thresholds[2]) or \
    #         (value == label_dict["full"] and preds[idx] > value - thresholds[2]):

    #         running_corrects += 1

    # return running_corrects / size

def accuracy_threshold(preds: torch.Tensor, targets: torch.Tensor, label_dict: dict, thresholds=[0.5,1.5,2.5]):  
    assert len(thresholds) == 3
    assert preds.dim() == 1 and targets.dim() == 1

    size = targets.size(0)
    running_corrects = 0

    for idx, value in enumerate(targets):
        if (value == label_dict["empty"] and preds[idx] < thresholds[0]) or \
            (value == label_dict["minimal"] and preds[idx] < thresholds[1]) or \
            (value == label_dict["normal"] and preds[idx] < thresholds[2]) or \
            (value == label_dict["full"]  and preds[idx] >= thresholds[2]):

            running_corrects += 1

    return running_corrects / size

def get_pred(output, topk=(1,)):
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    return pred.t()[0]

def visualize(target, predict, checkpoint, num_classes=4):
    """Computes the confusion matrix and plot it"""
    # pred_onehot = nn.functional.one_hot(pred.max(axis=1).values, num_classes=num_classes)
    checkpoint = checkpoint.split("/")[-2]
    if not os.path.isdir(f"/content/feedlane/output/validate/{checkpoint}"):
        os.makedirs(f"/content/feedlane/output/validate/{checkpoint}") 
    cm = confusion_matrix(target, predict, labels=[i for i in range(num_classes)])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"/content/feedlane/output/validate/{checkpoint}/cm.jpg")
    return None
