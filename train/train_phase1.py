#!/usr/bin/env python3
"""
FeatherFace ECA-CBAM - Phase 1 Training
========================================

Phase 1: Backbone Pre-training (30 epochs)
- Attention modules DISABLED (both ECA and SAM)
- Focus: Train backbone to extract meaningful features
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=30)

Scientific Justification:
Stable backbone features are essential before introducing attention mechanisms.

Usage:
    python train/train_phase1.py --training_dataset ./data/widerface/train/label.txt
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('..')
sys.path.append('.')

from data import cfg_eca_cbam
from models.featherface_eca_cbam import FeatherFaceECAcbaM
from data.wider_face import WiderFaceDetection, detection_collate
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import preproc

# Training configuration
PHASE = 1
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

print("="*80)
print("PHASE 1: BACKBONE PRE-TRAINING")
print("Attention: DISABLED | Epochs: 30 | Optimizer: Adam")
print("="*80)

# TODO: Implement full training loop
# Key steps:
# 1. Create model and disable attention: model.disable_all()
# 2. Use Adam optimizer with weight_decay=1e-4
# 3. Use CosineAnnealingWarmRestarts scheduler
# 4. Train for 30 epochs
# 5. Save checkpoint: phase1_epoch_30.pth
