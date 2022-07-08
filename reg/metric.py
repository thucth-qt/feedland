import os
import torch

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