import torch
import math
import mmcv
import torch.nn as nn
from mmseg.models.losses.utils import weighted_loss
from mmseg.models.builder import LOSSES
from pdb import set_trace as st

@weighted_loss
def dice_loss(input, target, ignore_index=None, eps=0.001):
    N,H,W = input.shape
    input = input.contiguous().view(N, H*W)
    target = target.contiguous().view(N, H*W).float()
    if ignore_index is not None:
        valid_mask = target != ignore_index
        input = input[valid_mask].unsqueeze(0)
        target = target[valid_mask].unsqueeze(0)
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    return 1 - d

@LOSSES.register_module()
class DiceLoss_PDBEV(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DiceLoss_PDBEV, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0
        
    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
    
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=255,
            **kwargs)

        return loss
