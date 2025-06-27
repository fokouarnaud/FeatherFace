"""
Knowledge Distillation and Training Utilities for FeatherFace V2
Includes DropBlock, distillation losses, and augmentation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable


class DropBlock2D(nn.Module):
    """
    DropBlock regularization for 2D inputs
    Paper: https://arxiv.org/abs/1810.12890
    """
    def __init__(self, drop_prob=0.1, block_size=3):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        # Calculate gamma (drop probability per location)
        gamma = self._compute_gamma(x)
        
        # Sample mask
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.to(x.device)
        
        # Place mask on input
        block_mask = self._compute_block_mask(mask)
        
        # Normalize and apply mask
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        
        return out
    
    def _compute_gamma(self, x):
        """Compute gamma for DropBlock"""
        return (self.drop_prob / (self.block_size ** 2) * 
                (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2))
    
    def _compute_block_mask(self, mask):
        """Compute block mask for DropBlock"""
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining multiple components:
    - KL divergence on classification logits
    - L2 loss on intermediate features
    - Original task loss
    """
    def __init__(self, temperature=4.0, alpha=0.7, feature_weight=0.1):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.feature_weight = feature_weight
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_outputs, teacher_outputs, targets, criterion):
        """
        Args:
            student_outputs: (logits, features) from student model
            teacher_outputs: (logits, features) from teacher model
            targets: Ground truth labels
            criterion: Original task loss (MultiBoxLoss)
        """
        student_logits, student_features = student_outputs
        teacher_logits, teacher_features = teacher_outputs
        
        # Original task loss
        task_loss = criterion(student_logits, targets)
        
        # KL divergence loss on softened logits
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Feature matching loss
        feature_loss = 0
        if student_features and teacher_features:
            for s_feat, t_feat in zip(student_features, teacher_features):
                # Align dimensions if necessary
                if s_feat.shape[1] != t_feat.shape[1]:
                    # Use 1x1 conv to match channels
                    align_conv = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1).to(s_feat.device)
                    s_feat = align_conv(s_feat)
                feature_loss += self.mse_loss(s_feat, t_feat.detach())
        
        # Combine losses
        total_loss = (1 - self.alpha) * task_loss + \
                     self.alpha * distill_loss + \
                     self.feature_weight * feature_loss
        
        return total_loss, task_loss, distill_loss, feature_loss


def mixup_data(x, y, alpha=0.2):
    """
    MixUp augmentation
    Args:
        x: input images
        y: labels (can be a list for detection tasks)
        alpha: mixup parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    return mixed_x, y, y[index] if not isinstance(y, list) else None, lam, index


def cutmix_data(x, y, alpha=1.0, prob=0.5):
    """
    CutMix augmentation
    Args:
        x: input images
        y: labels
        alpha: cutmix parameter
        prob: probability of applying cutmix
    """
    if np.random.rand() > prob:
        return x, y, None, 1.0, None
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    lam = np.random.beta(alpha, alpha)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index] if not isinstance(y, list) else None, lam, index


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class FeatureExtractor(nn.Module):
    """
    Wrapper to extract intermediate features from a model for distillation
    """
    def __init__(self, model, feature_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.feature_layers = feature_layers
        self.features = []
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on specified layers"""
        def hook_fn(module, input, output):
            self.features.append(output)
            
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook_fn)
    
    def forward(self, x):
        self.features = []
        output = self.model(x)
        return output, self.features


def cosine_annealing_with_warmup(optimizer, current_epoch, max_epoch, 
                                lr_min=1e-6, lr_max=1e-3, warmup_epochs=5):
    """
    Cosine annealing scheduler with warmup
    """
    if current_epoch < warmup_epochs:
        # Linear warmup
        lr = lr_max * (current_epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (current_epoch - warmup_epochs) / (max_epoch - warmup_epochs)
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr