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

def get_pred(output, topk=(1,)):
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    return pred.max(axis=1).indices

def visualize(target, predict, checkpoint, num_classes=4):
    """Computes the confusion matrix and plot it"""
    # pred_onehot = nn.functional.one_hot(pred.max(axis=1).values, num_classes=num_classes)
    checkpoint = checkpoint.split("/")[-2]
    if not os.path.isdir(f"/content/feedlane/output/validate/{checkpoint}"):
        os.makedirs(f"/content/feedlane/output/validate/{checkpoint}") 
    cm = confusion_matrix(target, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"/content/feedlane/output/validate/{checkpoint}/cm.jpg")
    return None
