# Scripts Directory

This directory contains organized scripts for FeatherFace V1 baseline and V2 Ultra operations.

## 📁 Directory Structure

```
scripts/
├── setup/              # Environment setup scripts
│   ├── install_dependencies.py # Dependency installer
│   └── README.md               # Setup documentation
├── training/           # Training scripts
│   └── train.py                # V1 baseline training (teacher model)
├── validation/         # Validation and testing scripts
│   ├── validate_parameters.py  # Parameter count validation
│   ├── test_parameters.py      # Quick parameter testing
│   ├── quick_test.py           # Parameter debugging
│   └── final_validation.py     # Comprehensive model validation
├── deployment/         # Deployment and export scripts
│   └── export_dynamic_onnx.py  # Dynamic ONNX export
└── detection/          # Detection and inference scripts
    └── detect.py               # Face detection inference
```

## 🚀 Quick Usage

### Training
```bash
# V1 baseline training (teacher model)
python scripts/training/train.py --network mobile0.25

# V2 Ultra training with revolutionary knowledge distillation
python train_v2_ultra.py --teacher_model weights/mobilenet0.25_Final.pth --epochs 400
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
python scripts/deployment/export_dynamic_onnx.py --model v2_ultra --weights weights/v2_ultra/v2_ultra_final.pth
```

### Detection
```bash
# Face detection inference (supports V1 and V2 Ultra)
python scripts/detection/detect.py --image path/to/image.jpg --model v2_ultra
```

## 📖 Detailed Information

### Training Scripts
- **train.py**: V1 baseline training script (teacher model for knowledge distillation)
- **../train_v2_ultra.py**: Revolutionary V2 Ultra training with Intelligence > Capacity paradigm

### Validation Scripts
- **validate_parameters.py**: Validates model parameter counts against targets
- **final_validation.py**: Comprehensive model validation including architecture, performance, and compatibility

### Deployment Scripts
- **export_dynamic_onnx.py**: Exports models to ONNX format with dynamic input sizes

### Detection Scripts
- **detect.py**: Face detection inference supporting V1 baseline and V2 Ultra models

## 🔗 Related Documentation

- **V2 Ultra Architecture**: [docs/V2_ULTRA_ARCHITECTURE.md](../docs/V2_ULTRA_ARCHITECTURE.md)
- **Technical Documentation**: [docs/technical/TECHNICAL_DOCUMENTATION.md](../docs/technical/TECHNICAL_DOCUMENTATION.md)
- **Deployment Guide**: [deployment/README.md](../deployment/README.md)

## 💡 Tips

- Use **notebooks in notebooks/** for interactive development
- Use **scripts/** for automated training and batch processing
- Check **utils/** for GPU optimization and monitoring utilities
- See **deployment/** for production-ready models and configurations

---

**Note**: For interactive development, prefer using the Jupyter notebooks:
- `notebooks/01_train_evaluate_featherface.ipynb` for V1 baseline
- `notebooks/03_train_evaluate_featherface_v2_ultra.ipynb` for V2 Ultra revolutionary training

These scripts are optimized for command-line usage and batch processing.