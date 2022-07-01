""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSE(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(self, reduction: str = 'mean'):
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        target = target.to(torch.float32)
        target = torch.unsqueeze(target, dim=1)
        # import pdb; pdb.set_trace()
        return F.mse_loss(x, target, reduction=self.reduction)
