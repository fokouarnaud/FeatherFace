# Training Scripts

Command-line training scripts for FeatherFace models.

## 📋 Scripts Overview

### `train.py` - V1 Baseline Training
Training script for FeatherFace V1 baseline (494K parameters) - serves as teacher model for Nano-B.

```bash
# Basic usage
python train.py --network mobile0.25

# With custom parameters
python train.py --network mobile0.25 --batch_size 32 --epochs 350 --lr 1e-3
```

**Features**:
- V1 baseline architecture (494K parameters)
- Teacher model for Nano-B knowledge distillation
- Standard training pipeline
- Supports resume from checkpoint
- Automatic weight saving

### `../../train_nano_b.py` - Nano-B 2024 Training
Training script for FeatherFace Nano-B 2024 (120K-180K parameters) with optimized small face detection.

```bash
# Basic usage
python train_nano_b.py --teacher_model weights/mobilenet0.25_Final.pth

# With advanced options
python train_nano_b.py \
    --teacher_model weights/mobilenet0.25_Final.pth \
    --epochs 400 \
    --temperature 4.0 \
    --alpha 0.7 \
    --target_reduction 0.5 \
    --bayesian_iterations 25
```

**Enhanced 2024 Features**:
- Knowledge distillation from V1 baseline
- 3 specialized research modules (ASSN, MSE-FPN, Scale Decoupling)
- Differential pipeline (P3 specialized vs P4/P5 standard)
- Bayesian-optimized pruning (B-FPGM)
- 48-65% parameter reduction with small face specialization
- Scientific foundation with 10 peer-reviewed papers (2017-2025)

## 🔧 Common Parameters

### V1 Training (`train.py`)
- `--network`: Network architecture (default: mobile0.25)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 350)
- `--lr`: Learning rate (default: 1e-3)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--resume_net`: Path to checkpoint for resuming
- `--resume_epoch`: Epoch to resume from

### Nano-B Enhanced Training (`../../train_nano_b.py`)
All V1 parameters plus:
- `--teacher_model`: Path to V1 teacher model weights (required)
- `--temperature`: Distillation temperature (default: 4.0)
- `--alpha`: Distillation weight (default: 0.7)
- `--target_reduction`: Pruning target reduction (default: 0.5)
- `--bayesian_iterations`: Bayesian optimization iterations (default: 25)
- `--acquisition_function`: BO acquisition function (default: 'ei')
- `--small_face_optimization`: Enable P3 specialization (default: True)

## 📊 Training Monitoring

Both scripts support comprehensive monitoring:

```bash
# Enable TensorBoard logging
tensorboard --logdir ./runs

# Monitor GPU usage
nvidia-smi -l 1

# Check training logs
tail -f training.log
```

## 🚀 Performance Tips

### GPU Optimization
```bash
# Use mixed precision training
export TORCH_CUDA_ARCH_LIST="8.0"  # For RTX 30xx series
python train_nano_b.py --mixed_precision

# Optimize batch size for your GPU
python scripts/validation/validate_parameters.py --optimize_batch_size
```

### Memory Management
```bash
# For limited GPU memory
python train_nano_b.py --batch_size 16 --gradient_accumulation 2

# For high-memory GPUs
python train_nano_b.py --batch_size 64
```

## 📈 Expected Results

### V1 Baseline Training
- **Target**: 87.0% mAP on WIDERFace
- **Parameters**: 494K (baseline)
- **Training time**: ~6-8 hours on RTX 3080
- **Role**: Teacher model for Nano-B Enhanced

### Nano-B Enhanced 2024 Training
- **Target**: Competitive mAP + 15-20% small face improvement
- **Parameters**: 120K-180K (48-65% reduction)
- **Training time**: ~8-10 hours with Bayesian optimization
- **Achievement**: Specialized small face detection with differential pipeline

## 🔧 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Teacher model not found**: Check teacher model path and ensure V1 is trained
3. **Slow training**: Enable mixed precision and optimize batch size
4. **Poor convergence**: Adjust learning rate and distillation parameters

### Solutions
```bash
# Check model compatibility
python scripts/validation/validate_parameters.py

# Validate teacher model
python scripts/validation/final_validation.py --model_path weights/mobilenet0.25_Final.pth

# Monitor training health
python -c "from utils.monitoring import setup_training_monitoring; print('Monitoring ready')"
```

## 🔗 Related Files

- **Interactive Training**: [notebooks/](../../notebooks/) - Jupyter notebooks for development
- **Validation**: [../validation/](../validation/) - Model validation scripts
- **Monitoring Utils**: [utils/monitoring.py](../../utils/monitoring.py) - Training monitoring tools
- **GPU Utils**: [utils/gpu_optimization.py](../../utils/gpu_optimization.py) - GPU optimization

---

**Recommendation**: For first-time training or experimentation, use the Jupyter notebooks:
- `notebooks/01_train_evaluate_featherface.ipynb` for V1 baseline
- `notebooks/04_train_evaluate_featherface_nano_b.ipynb` for Nano-B Enhanced 2024