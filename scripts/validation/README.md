# Validation Scripts

Scripts for comprehensive model validation and parameter verification.

## 📋 Scripts Overview

### `validate_parameters.py` - Parameter Count Validation
Validates model parameter counts against paper specifications.

```bash
# Validate V1 model
python validate_parameters.py

# Validate Nano-B Enhanced model
python validate_parameters.py --model nano_b --config cfg_nano_b

# Verbose output
python validate_parameters.py --verbose
```

**Features**:
- Validates V1 baseline target of 494K parameters
- Validates Nano-B Enhanced target of 120K-180K parameters
- Compares against Enhanced 2024 specifications
- Detailed parameter breakdown by module
- Architecture compatibility checks

### `final_validation.py` - Comprehensive Model Validation
Complete validation suite for model architecture, performance, and deployment readiness.

```bash
# Full validation suite
python final_validation.py

# Validate specific model
python final_validation.py --model_path weights/mobilenet0.25_Final.pth

# Quick validation
python final_validation.py --quick

# Export validation report
python final_validation.py --export_report validation_report.json
```

**Features**:
- Architecture validation (parameter counts, layer structure)
- Forward pass testing (multiple input sizes, batch sizes)
- Device compatibility (CPU/CUDA testing)
- Memory efficiency analysis
- Training readiness verification
- ONNX export compatibility
- Performance benchmarking

## 🔧 Validation Categories

### 1. Architecture Validation
```bash
# Check parameter counts
python validate_parameters.py --check_architecture

# Verify layer structure
python final_validation.py --test_architecture
```

**Validates**:
- Total parameter count vs target
- Trainable vs non-trainable parameters
- Module-wise parameter distribution
- Layer initialization quality
- Gradient flow potential

### 2. Functional Validation
```bash
# Test forward pass
python final_validation.py --test_forward

# Test different input sizes
python final_validation.py --test_inputs 320,416,640,832
```

**Validates**:
- Forward pass with various input sizes
- Batch processing capabilities
- Output shape consistency
- Numerical stability

### 3. Device Compatibility
```bash
# Test CPU/GPU compatibility
python final_validation.py --test_devices

# Memory usage analysis
python final_validation.py --test_memory
```

**Validates**:
- CPU inference capability
- CUDA inference capability
- Memory usage patterns
- Optimal batch size recommendations

### 4. Training Readiness
```bash
# Check training compatibility
python final_validation.py --test_training

# Validate gradients
python final_validation.py --test_gradients
```

**Validates**:
- Gradient computation capability
- Optimizer compatibility
- Loss function integration
- Learning rate sensitivity

## 📊 Validation Reports

### Parameter Validation Report
```json
{
  "model": "FeatherFace_V1",
  "total_parameters": 494000,
  "target_parameters": 494000,
  "difference": 243,
  "within_tolerance": true,
  "module_breakdown": {
    "backbone": 220156,
    "bifpn": 112606,
    "ssh_modules": 155481
  }
}
```

### Comprehensive Validation Report
```json
{
  "validation_passed": true,
  "architecture": {
    "parameter_count": "PASSED",
    "structure": "PASSED",
    "initialization": "PASSED"
  },
  "functionality": {
    "forward_pass": "PASSED",
    "input_sizes": "PASSED",
    "batch_processing": "PASSED"
  },
  "compatibility": {
    "cpu": "PASSED",
    "cuda": "PASSED",
    "memory_efficient": "PASSED"
  },
  "performance": {
    "inference_speed": "25.3ms",
    "memory_usage": "1.2GB",
    "throughput": "39.5 FPS"
  }
}
```

## 🎯 Target Specifications

### FeatherFace V1 (Baseline)
- **Parameters**: 494,000 ± 3,000
- **Architecture**: SSH detection with BiFPN
- **Performance**: 87.0% mAP on WIDERFace
- **Role**: Teacher model for Nano-B Enhanced
- **Memory**: ~1.9MB model size

### FeatherFace Nano-B Enhanced 2024
- **Parameters**: 120,000-180,000 (variable Bayesian optimization)
- **Architecture**: Differential pipeline with 3 specialized research modules
- **Performance**: Competitive mAP + 15-20% small face improvement
- **Memory**: ~0.6-0.9MB model size
- **Efficiency**: 48-65% parameter reduction with specialized detection

## 🚀 Usage Examples

### Quick Parameter Check
```bash
# Fast parameter validation
python validate_parameters.py --model v1
# Expected output: ✅ V1: 494,000 parameters (target: 494,000 ± 3,000)

python validate_parameters.py --model nano_b
# Expected output: ✅ Nano-B Enhanced: 150,000 parameters (target: 120,000-180,000)
```

### Pre-training Validation
```bash
# Before starting training
python final_validation.py --test_architecture --test_forward --test_training
```

### Pre-deployment Validation
```bash
# Before deploying to production
python final_validation.py --test_devices --test_memory --export_report deployment_validation.json
```

### CI/CD Integration
```bash
# For automated testing
python final_validation.py --quick --export_report ci_validation.json
if [ $? -eq 0 ]; then echo "✅ Validation passed"; else echo "❌ Validation failed"; exit 1; fi
```

## 🔧 Advanced Options

### Custom Validation
```bash
# Custom parameter tolerance
python validate_parameters.py --tolerance 10000

# Custom input sizes
python final_validation.py --input_sizes 224,320,416,640,832,1024

# Memory limit testing
python final_validation.py --memory_limit 4GB
```

### Batch Validation
```bash
# Validate multiple models
for model in weights/*.pth; do
    python final_validation.py --model_path "$model" --quick
done
```

## 📈 Performance Benchmarks

### Expected Validation Times
- **Parameter validation**: < 5 seconds
- **Quick validation**: < 30 seconds
- **Full validation**: 2-5 minutes
- **Memory analysis**: 1-2 minutes

### Resource Requirements
- **CPU**: Minimal (any modern CPU)
- **Memory**: 2-4GB RAM recommended
- **GPU**: Optional (for CUDA compatibility testing)

## 🔗 Integration with Other Tools

### With Training Scripts
```bash
# Validate before training
python scripts/validation/validate_parameters.py && python train_nano_b.py --teacher_model weights/mobilenet0.25_Final.pth
```

### With Monitoring Utils
```python
from utils.validation import quick_model_validation
from utils.monitoring import setup_training_monitoring

# In training script
if quick_model_validation(model, expected_params=494000):  # V1 baseline
    tracker = setup_training_monitoring("v1_baseline")
    # Start V1 training
elif quick_model_validation(model, expected_params=150000):  # Nano-B Enhanced
    tracker = setup_training_monitoring("nano_b_enhanced_2024")
    # Start Nano-B Enhanced training
```

### With GPU Optimization
```python
from utils.gpu_optimization import quick_gpu_setup
from utils.validation import ModelValidator

# Optimized validation
model, gpu_opt, _ = quick_gpu_setup(model)
validator = ModelValidator(device=gpu_opt.device)
results = validator.run_comprehensive_validation(model)
```

## 🔧 Troubleshooting

### Common Issues
1. **Parameter count mismatch**: Check model configuration in `data/config.py`
2. **CUDA compatibility errors**: Ensure proper PyTorch CUDA installation
3. **Memory validation failures**: Reduce batch sizes or use CPU mode
4. **Forward pass errors**: Check input dimensions and model architecture

### Debug Commands
```bash
# Debug parameter counting
python validate_parameters.py --debug --verbose

# Debug forward pass
python final_validation.py --debug --test_forward

# Check model loading
python -c "from models.retinaface import RetinaFace; from data.config import cfg_mnet; print('Model loads:', RetinaFace(cfg_mnet, phase='test'))"
```

---

**Best Practice**: Run validation before training, after training, and before deployment to ensure model quality and compatibility.