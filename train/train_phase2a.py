#!/usr/bin/env python3
"""
FeatherFace ECA-CBAM - Phase 2a Training
=========================================

Phase 2a: ECA Activation (25 epochs)
- ECA ENABLED, SAM DISABLED
- Focus: Refine channel attention on pre-trained features
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=25)

Scientific Justification:
Progressive attention activation prevents instability.
ECA learns channel relationships on stable backbone features.

Usage:
    python train/train_phase2a.py --resume_net ./weights/phase1/phase1_epoch_30.pth
"""

import os
import sys
import torch
import torch.optim as optim

sys.path.append('..')
sys.path.append('.')

from data import cfg_eca_cbam
from models.featherface_eca_cbam import FeatherFaceECAcbaM

# Training configuration
PHASE = "2a"
EPOCHS = 25
LR = 5e-4
WEIGHT_DECAY = 1e-4

print("="*80)
print("PHASE 2a: ECA ACTIVATION")
print("Attention: ECA ENABLED, SAM DISABLED | Epochs: 25")
print("="*80)

# TODO: Implement training loop
# Key steps:
# 1. Load checkpoint from Phase 1
# 2. Enable ECA only: model.enable_eca_only()
# 3. Use Adam with lr=5e-4
# 4. Train for 25 epochs
# 5. Save: phase2a_epoch_25.pth
