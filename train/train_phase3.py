#!/usr/bin/env python3
"""
FeatherFace ECA-CBAM - Phase 3 Training
========================================

Phase 3: Global Fine-tuning (30 epochs)
- ECA ENABLED, SAM ENABLED (Sequential: ECA -> SAM)
- Focus: Fine-tune entire model with regularization
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=30)
- Regularization: Mixup (alpha=0.2), Dropout (p=0.3), Gradient Clipping (norm=5.0)

Scientific Justification:
Final fine-tuning with strong regularization improves generalization.
Mixup data augmentation prevents overfitting.

Usage:
    python train/train_phase3.py --resume_net ./weights/phase2b/phase2b_epoch_25.pth
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
PHASE = 3
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
DROPOUT = 0.3
GRAD_CLIP = 5.0

print("="*80)
print("PHASE 3: GLOBAL FINE-TUNING")
print("Attention: FULL SEQUENTIAL | Epochs: 30")
print("Regularization: Mixup + Dropout + Grad Clipping")
print("="*80)

# TODO: Implement training loop
# Key steps:
# 1. Load checkpoint from Phase 2b
# 2. Both enabled: model.enable_both()
# 3. Use Adam with lr=1e-4
# 4. Apply Mixup augmentation (alpha=0.2)
# 5. Add Dropout (p=0.3) to detection heads
# 6. Gradient clipping (max_norm=5.0)
# 7. Train for 30 epochs
# 8. Save final: phase3_epoch_30.pth (FINAL MODEL)
