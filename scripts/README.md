# Scripts Directory

This directory contains organized scripts for different FeatherFace operations.

## 📁 Directory Structure

```
scripts/
├── setup/              # Environment setup scripts
│   ├── install_dependencies.py # Dependency installer
│   └── README.md               # Setup documentation
├── training/           # Training scripts
│   ├── train.py                # V1 training (original)
│   ├── train_v2.py             # V2 training with knowledge distillation
│   └── start_v2_training.py    # Quick V2 training starter
├── validation/         # Validation and testing scripts
│   ├── validate_parameters.py  # Parameter count validation
│   ├── test_parameters.py      # Quick parameter testing
│   ├── quick_test.py           # Parameter debugging
│   └── final_validation.py     # Comprehensive model validation
├── deployment/         # Deployment and export scripts
│   └── export_dynamic_onnx.py  # Dynamic ONNX export
└── detection/          # Detection and inference scripts
    ├── detect.py               # Basic face detection
    └── detect_faces_v2_fixed.py # Fixed V2 detection script
```

## 🚀 Quick Usage

### Training
```bash
# V1 training
python scripts/training/train.py --network mobile0.25

# V2 training with knowledge distillation
python scripts/training/train_v2.py --teacher_model weights/mobilenet0.25_Final.pth

# Quick V2 start
python scripts/training/start_v2_training.py
```

### Setup
```bash
# Install all dependencies
python scripts/setup/install_dependencies.py
```

### Validation
```bash
# Validate parameter counts
python scripts/validation/validate_parameters.py

# Quick parameter test
python scripts/validation/quick_test.py

# Test model parameters
python scripts/validation/test_parameters.py

# Comprehensive validation
python scripts/validation/final_validation.py
```

### Deployment
```bash
# Export ONNX models
python scripts/deployment/export_dynamic_onnx.py --model v1 --weights weights/mobilenet0.25_Final.pth
python scripts/deployment/export_dynamic_onnx.py --model v2 --weights weights/v2/FeatherFaceV2_final.pth
```

### Detection
```bash
# Basic detection
python scripts/detection/detect.py --image path/to/image.jpg

# V2 detection (fixed version)
python scripts/detection/detect_faces_v2_fixed.py --image path/to/image.jpg
```

## 📖 Detailed Information

### Training Scripts
- **train.py**: Original training script for V1 model
- **train_v2.py**: Advanced training with knowledge distillation for V2
- **start_v2_training.py**: Simplified V2 training wrapper

### Validation Scripts
- **validate_parameters.py**: Validates model parameter counts against targets
- **final_validation.py**: Comprehensive model validation including architecture, performance, and compatibility

### Deployment Scripts
- **export_dynamic_onnx.py**: Exports models to ONNX format with dynamic input sizes

### Detection Scripts
- **detect.py**: Basic face detection inference
- **detect_faces_v2_fixed.py**: Enhanced V2 detection with device compatibility fixes

## 🔗 Related Documentation

- **Training Guide**: [docs/training_v2_guide.md](../docs/training_v2_guide.md)
- **Technical Documentation**: [docs/technical/TECHNICAL_DOCUMENTATION.md](../docs/technical/TECHNICAL_DOCUMENTATION.md)
- **Deployment Guide**: [deployment/README.md](../deployment/README.md)

## 💡 Tips

- Use **notebooks in notebooks/** for interactive development
- Use **scripts/** for automated training and batch processing
- Check **utils/** for GPU optimization and monitoring utilities
- See **deployment/** for production-ready models and configurations

---

**Note**: For interactive development and experimentation, prefer using the Jupyter notebooks in the `notebooks/` directory. These scripts are optimized for command-line usage and batch processing.