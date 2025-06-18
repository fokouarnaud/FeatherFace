# FeatherFace V2 Implementation

## Overview
This fork implements FeatherFace V2 with optimizations targeting:
- **Model Size**: 0.25M parameters (49% reduction from 0.49M)
- **Performance**: 92%+ mAP on WIDERFace (improvement from 90.8%)
- **Speed**: 25+ FPS on entry-level mobile devices

## Implementation Phases

### Phase 01 - Baseline (branch: phase-01-baseline)
- Reproduce original FeatherFace results
- Establish performance baseline
- Notebook: `notebooks/01_train_evaluate_featherface.ipynb`

### Phase 02 - V2 Optimizations (branch: phase-02-v2)
- Implement optimized modules (CBAM++, Grouped Heads, etc.)
- Knowledge distillation from RetinaFace
- Achieve target metrics
- Notebook: `notebooks/02_train_evaluate_featherfacev2.ipynb`

## Key Innovations
1. **Lightweight CBAM++**: 75% parameter reduction
2. **Grouped Convolution Heads**: 50% reduction via weight sharing
3. **Progressive Knowledge Distillation**: Multi-level transfer learning
4. **Hardware-Aware Quantization**: INT8/FP16 mixed precision

## Setup Instructions

1. **Environment Setup**
```bash
# Check environment
python setup_env.py

# Install dependencies
pip install -r requirements.txt
```

2. **Download Pre-trained Weights**
- MobilenetV1X0.25_pretrain.tar → `./weights/`
- [Download Link](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)

3. **Prepare Dataset**
- Download WIDERFace dataset
- Organize in `./data/widerface/` as per original README

4. **Training**
- Phase 01: Run `notebooks/01_train_evaluate_featherface.ipynb`
- Phase 02: Run `notebooks/02_train_evaluate_featherfacev2.ipynb`

## Project Structure
```
FeatherFace/
├── notebooks/              # Training notebooks
├── models/
│   ├── featherface_v2.py  # V2 architecture
│   └── modules/           # Optimized modules
├── training/              # Training utilities
├── weights/               # Pre-trained models
├── results/               # Experiment results
└── analysis/              # Architecture analysis
```

## Progress Tracking
See `TASKS.md` for detailed task status and implementation progress.

## Original Paper
Kim, D.; Jung, J.; Kim, J. FeatherFace: Robust and Lightweight Face Detection via Optimal Feature Integration. Electronics 2025

## Contributors
- Original: @dohun-mat
- V2 Implementation: @fokouarnaud
