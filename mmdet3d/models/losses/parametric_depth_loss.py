# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import math
import mmcv
import torch.nn as nn
from mmseg.models.losses.utils import weighted_loss
from mmseg.models.builder import LOSSES

def gaussian(mu, sigma, labels):
    return torch.exp(-0.5*(mu-labels)** 2/ sigma** 2)/sigma

def laplacian(mu, b, labels):
    return 0.5 * torch.exp(-(torch.abs(mu-labels)/b))/b

def distribution(mu, sigma, labels, dist="gaussian"):
    return gaussian(mu, sigma, labels) if dist=="gaussian" else \
           laplacian(mu, sigma, labels)

@weighted_loss
def parametric_depth_NLL_loss(mu, sigma, labels, dist='gaussian', reduction='mean'):
    likelihood = distribution(mu, sigma, labels, dist=dist)
    neg_log_likelihood = -torch.log(likelihood)
    if reduction == 'mean':
        return neg_log_likelihood.mean()
    elif reduction == 'sum':
        return neg_log_likelihood.sum()
    else:
        return neg_log_likelihood

@LOSSES.register_module()
class Parametric_Depth_NLL_Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, dist="gaussian"):
        super(Parametric_Depth_NLL_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.dist = dist
        
    def forward(self,
                mus,
                sigmas,
                labels,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
    
        loss = self.loss_weight * parametric_depth_NLL_loss(
            mus,
            sigmas,
            labels,
            self.dist
            )

        return loss
