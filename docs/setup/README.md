# Installation & Setup Guide

Complete setup instructions for FeatherFace development and deployment environment.

## 🚀 Quick Installation

```bash
# Clone repository
git clone https://github.com/dohun-mat/FeatherFace
cd FeatherFace

# Install dependencies
pip install -e .

# Verify installation
python -c "from models.retinaface import RetinaFace; print('✅ Installation successful')"
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 20GB+ free space

## 🔧 Detailed Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv featherface_env
source featherface_env/bin/activate  # Linux/Mac
# featherface_env\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Core Dependencies
```bash
# Essential packages
pip install torch torchvision torchaudio
pip install opencv-python
pip install albumentations
pip install numpy pandas matplotlib

# Training and evaluation
pip install tensorboard
pip install tqdm
pip install scikit-learn

# Export and deployment
pip install onnx onnxruntime
pip install coremltools  # For iOS deployment
```

### 3. Development Tools (Optional)
```bash
# Code quality
pip install black isort flake8

# Testing
pip install pytest pytest-cov

# Documentation
pip install jupyter notebook
pip install mkdocs mkdocs-material
```

## 📊 Dataset Preparation

### WIDERFace Dataset
```bash
# Create data directory
mkdir -p data/widerface

# Download dataset (choose one method)
# Method 1: Direct download
wget https://drive.google.com/uc?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS -O data/widerface.zip

# Method 2: Use gdown (recommended)
pip install gdown
gdown https://drive.google.com/uc?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS -O data/widerface.zip

# Extract dataset
cd data && unzip widerface.zip && cd ..
```

### Dataset Structure Verification
```bash
# Expected structure:
data/widerface/
├── train/
│   ├── images/
│   └── label.txt
└── val/
    ├── images/
    └── wider_val.txt

# Verify structure
python -c "
import os
train_label = 'data/widerface/train/label.txt'
val_label = 'data/widerface/val/wider_val.txt'
print('✅ Train label found' if os.path.exists(train_label) else '❌ Train label missing')
print('✅ Val label found' if os.path.exists(val_label) else '❌ Val label missing')
"
```

### Pre-trained Weights
```bash
# Create weights directory
mkdir -p weights

# Download MobileNet backbone
gdown https://drive.google.com/uc?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1 -O weights/mobilenetV1X0.25_pretrain.tar

# Verify download
ls -la weights/
```

## 🖥️ Hardware Configuration

### CUDA Setup (NVIDIA GPUs)
```bash
# Check CUDA availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Memory Optimization
```bash
# For limited GPU memory, add to training scripts:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# For CPU-only training:
export CUDA_VISIBLE_DEVICES=""
```

## 🧪 Installation Verification

### Quick Test Suite
```bash
# Run basic tests
python -c "
# Test imports
from models.retinaface import RetinaFace
from models.featherface_v2_simple import FeatherFaceV2Simple
from data.config import cfg_mnet, cfg_v2
print('✅ All imports successful')

# Test model creation
import torch
model_v1 = RetinaFace(cfg=cfg_mnet, phase='test')
print(f'✅ V1 model: {sum(p.numel() for p in model_v1.parameters()):,} parameters')

model_v2 = FeatherFaceV2Simple(cfg=cfg_v2, phase='test')
print(f'✅ V2 model: {sum(p.numel() for p in model_v2.parameters()):,} parameters')
"
```

### Test Training Pipeline
```bash
# Quick training test (1 epoch)
python train_v1.py \
    --training_dataset ./data/widerface/train/label.txt \
    --network mobile0.25 \
    --epoch 1 \
    --batch_size 4
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Install in development mode
pip install -e .

# Error: CUDA out of memory
# Solution: Reduce batch size
# In training scripts, use --batch_size 16 or lower
```

#### 2. Dataset Issues
```bash
# Error: Dataset not found
# Solution: Verify paths
ls data/widerface/train/label.txt
ls data/widerface/val/wider_val.txt

# Error: Permission denied
# Solution: Check file permissions
chmod +r data/widerface/train/label.txt
```

#### 3. Performance Issues
```bash
# Slow training on CPU
# Solution: Use GPU or reduce model size
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Memory issues
# Solution: Reduce workers and batch size
# --num_workers 2 --batch_size 8
```

### Environment Debugging
```bash
# Full environment check
python scripts/validation/validate_environment.py

# Check specific components
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import cv2
print(f'OpenCV: {cv2.__version__}')

import albumentations
print(f'Albumentations: {albumentations.__version__}')
"
```

## 📖 Next Steps

### After Successful Installation
1. **[Training Tutorial](../guides/training.md)** - Train your first model
2. **[Architecture Guide](../architecture/README.md)** - Understand the models
3. **[Jupyter Notebooks](../../notebooks/)** - Interactive learning

### For Development
1. **[Contributing Guide](../technical/development.md)** - Development workflow
2. **[Testing Guide](../technical/testing.md)** - Running tests
3. **[Code Style](../technical/style.md)** - Coding standards

---

**Setup Status**: ✅ Complete installation guide  
**Platforms**: Windows, macOS, Linux  
**Last Updated**: January 2025