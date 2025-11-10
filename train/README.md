# FeatherFace ECA-CBAM Multi-Phase Training Strategy

This directory contains the 4-phase sequential training strategy described in the thesis methodology.

## Training Phases Overview

### Phase 1: Backbone Pre-training (30 epochs)
**File**: `train_phase1.py`

- **Attention**: DISABLED (both ECA and SAM)
- **Purpose**: Train backbone to extract stable features
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=30)
- **Output**: `phase1_epoch_30.pth`

**Scientific Justification**:
Stable backbone features are essential before introducing attention mechanisms.
This prevents attention modules from overfitting to poorly initialized features.

### Phase 2a: ECA Activation (25 epochs)
**File**: `train_phase2a.py`

- **Attention**: ECA ENABLED, SAM DISABLED
- **Purpose**: Refine channel attention on pre-trained features
- **Optimizer**: Adam (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=25)
- **Input**: `phase1_epoch_30.pth`
- **Output**: `phase2a_epoch_25.pth`

**Scientific Justification**:
Progressive attention activation prevents training instability.
ECA learns channel relationships on stable backbone features.

### Phase 2b: Full Sequential Attention (25 epochs)
**File**: `train_phase2b.py`

- **Attention**: ECA ENABLED → SAM ENABLED (Sequential architecture)
- **Purpose**: Learn spatial localization on ECA-refined features
- **Optimizer**: Adam (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=25)
- **Input**: `phase2a_epoch_25.pth`
- **Output**: `phase2b_epoch_25.pth`

**Scientific Justification**:
SAM learns to localize spatially on channel-refined features from ECA.
Sequential architecture: X → ECA(X) → SAM(ECA(X))

### Phase 3: Global Fine-tuning (30 epochs)
**File**: `train_phase3.py`

- **Attention**: FULL SEQUENTIAL (ECA → SAM)
- **Purpose**: Fine-tune entire model with strong regularization
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=30)
- **Regularization**:
  - Mixup data augmentation (α=0.2)
  - Dropout (p=0.3) in detection heads
  - Gradient Clipping (max_norm=5.0)
- **Input**: `phase2b_epoch_25.pth`
- **Output**: `phase3_epoch_30.pth` (FINAL MODEL)

**Scientific Justification**:
Final fine-tuning with strong regularization improves generalization.
Mixup prevents overfitting, dropout adds robustness, gradient clipping ensures stability.

## Usage

### Complete Training Pipeline

```bash
# Phase 1: Backbone pre-training (30 epochs)
python train/train_phase1.py --training_dataset ./data/widerface/train/label.txt

# Phase 2a: Activate ECA (25 epochs)
python train/train_phase2a.py --resume_net ./weights/phase1/phase1_epoch_30.pth

# Phase 2b: Activate SAM (25 epochs)
python train/train_phase2b.py --resume_net ./weights/phase2a/phase2a_epoch_25.pth

# Phase 3: Global fine-tuning (30 epochs)
python train/train_phase3.py --resume_net ./weights/phase2b/phase2b_epoch_25.pth
```

### Total Training Time

- **Total Epochs**: 110 (30 + 25 + 25 + 30)
- **Expected Performance** (WIDER FACE):
  - Easy: 92.5% mAP
  - Medium: 90.8% mAP
  - Hard: 80.0% mAP

## Key Configuration Parameters

### Optimizer (All Phases)
- **Type**: Adam (not AdamW)
- **Betas**: (0.9, 0.999)
- **Epsilon**: 1e-8
- **Weight Decay**: 1e-4

### Learning Rate Schedule
- **Phase 1**: 1e-3 → CosineAnnealing (T_0=30)
- **Phase 2a**: 5e-4 → CosineAnnealing (T_0=25)
- **Phase 2b**: 5e-4 → CosineAnnealing (T_0=25)
- **Phase 3**: 1e-4 → CosineAnnealing (T_0=30)

### Loss Weights (All Phases)
- **Localization**: 2.0
- **Classification**: 1.0
- **Landmarks**: 0.5

### Regularization (Phase 3 Only)
- **Mixup**: α = 0.2
- **Dropout**: p = 0.3
- **Gradient Clipping**: max_norm = 5.0

## Architecture Evolution

```
Phase 1:  X → Backbone → Features (no attention)
Phase 2a: X → Backbone → ECA(Features)
Phase 2b: X → Backbone → ECA(Features) → SAM(ECA_Features)
Phase 3:  Same as 2b with heavy regularization
```

## Expected Results

### Thesis Benchmarks (WIDER FACE)

| Subset | Target mAP |
|--------|-----------|
| Easy   | 92.5%     |
| Medium | 90.8%     |
| Hard   | 80.0%     |

### Model Specifications

- **Parameters**: ~460K (vs 488.7K CBAM baseline)
- **Latency**: 3.2ms/image (GPU A100), 10ms (Snapdragon 888)
- **Model Size**: 1.4 MB
- **Efficiency**: O(C) complexity for channel attention

## References

1. Wang et al. CVPR 2020: ECA-Net
2. Woo et al. ECCV 2018: CBAM
3. Lu et al. 2024: Multi-phase training for hybrid attention
4. Zhang et al. ICLR 2018: Mixup augmentation

## Notes

- These scripts provide the training framework structure
- Full implementation requires integration with existing training infrastructure
- Ensure WIDER FACE dataset is properly configured before training
- Monitor TensorBoard logs for each phase to track convergence
