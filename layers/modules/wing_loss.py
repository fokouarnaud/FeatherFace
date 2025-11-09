#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * np.log(1.0 + w / epsilon)
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask_small = diff < self.w
        loss = torch.where(
            mask_small,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C
        )
        return loss.mean()
    
    def extra_repr(self):
        return f"w={self.w}, epsilon={self.epsilon}, C={self.C:.4f}"


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def forward(self, pred, target):
        y = pred - target
        y_abs = torch.abs(y)
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y_abs)))
        C = self.theta * A - self.omega * torch.log(1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y_abs))
        mask = y_abs < self.theta
        loss = torch.where(
            mask,
            self.omega * torch.log(1.0 + torch.pow(y_abs / self.epsilon, self.alpha - y_abs)),
            A * y_abs - C
        )
        return loss.mean()
