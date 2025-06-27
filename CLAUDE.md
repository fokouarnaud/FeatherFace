# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Training Commands
- **Train V1 (Original)**: `CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --network mobile0.25`
- **Train V2 (Optimized)**: `python train_v2.py --epochs 400 --teacher_model weights/mobilenet0.25_Final.pth`
- **Quick V2 Start**: `python start_v2_training.py`

### Testing & Evaluation
- **Test on WIDERFace**: `python test_widerface.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25`
- **V1 vs V2 Comparison**: `python test_v1_v2_comparison.py`
- **Run Unit Tests**: `python -m pytest tests/test_modules_v2.py -v`
- **WIDERFace Evaluation**: `cd widerface_evaluate && python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth`

### Model Utilities
- **Check Model Compatibility**: `python check_model_compatibility.py`
- **Parameter Analysis**: See `analysis/analyze_architecture.py`

### Development Tools
- **Linting**: `black --line-length 100 . && isort --profile black .`
- **Install Dependencies**: `pip install -e .` (uses pyproject.toml)

## Architecture Overview

### Two-Model System
1. **FeatherFace V1** (`models/retinaface.py`): Original 592K parameter model
2. **FeatherFace V2** (`models/retinaface_v2.py`): Optimized 256K parameter model (56.7% reduction)

### Key Components
- **Backbone**: MobileNetV1 0.25x (`models/net.py`)
- **Feature Pyramid**: BiFPN for multi-scale features
- **Attention**: CBAM (Convolutional Block Attention Module)
- **Detection Head**: SSH (Single Shot Hierarchical) modules
- **V2 Optimizations**: Lightweight modules in `models/modules_v2.py`

### V2 Optimizations
- **BiFPN_Light**: 88.3% parameter reduction from original BiFPN
- **SSH_Grouped**: 89.7% parameter reduction using grouped convolutions
- **CBAM_Plus**: Shared attention weights across network
- **SharedMultiHead**: Unified detection heads
- **Knowledge Distillation**: V1 model teaches V2 (`layers/modules_distill.py`)

### Configuration System
- **V1 Config**: `cfg_mnet` in `data/config.py`
- **V2 Config**: `cfg_mnet_v2` in `data/config.py`
- Key differences: V2 uses 32 output channels vs 64, extended training epochs

### Data Pipeline
- **Dataset**: WIDERFace (`data/wider_face.py`)
- **Augmentation**: Standard + V2 advanced (MixUp, CutMix, DropBlock)
- **Expected Structure**: 
  ```
  data/widerface/
  ├── train/images/ & label.txt
  └── val/images/ & wider_val.txt
  ```

### Training Strategy
- **V1**: Standard RetinaFace training
- **V2**: Knowledge distillation with teacher model
  - Temperature: 4.0
  - Alpha: 0.7 (distillation weight)
  - Advanced augmentations enabled

### Model Weights
- **Pretrained Backbone**: `weights/mobilenetV1X0.25_pretrain.tar`
- **V1 Final**: `weights/mobilenet0.25_Final.pth`
- **V2 Weights**: `weights/v2/` directory

### Evaluation System
- **Quick Tests**: Compare V1/V2 performance with test scripts
- **Full Evaluation**: WIDERFace protocol with official evaluation tools
- **Notebooks**: Comprehensive analysis in `notebooks/` (01, 02, 03)

## Important Notes

### Development Workflow
1. Start with V1 training to generate teacher model
2. Use teacher model for V2 knowledge distillation
3. Compare performance using comparison scripts
4. Full evaluation on WIDERFace dataset

### Model Selection
- Use V1 for maximum accuracy
- Use V2 for deployment (faster, smaller, 92%+ mAP maintained)

### Testing
- Unit tests focus on V2 optimized modules
- Integration tests compare V1/V2 outputs
- Performance tests validate parameter reduction claims

### Dependencies
- PyTorch ecosystem (torch, torchvision)
- Computer vision (opencv, albumentations)
- ONNX export capabilities included
- Jupyter notebooks for interactive development