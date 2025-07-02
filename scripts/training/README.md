# Training Scripts

Command-line training scripts for FeatherFace models.

## 📋 Scripts Overview

### `train.py` - V1 Training
Original training script for FeatherFace V1 (489K parameters).

```bash
# Basic usage
python train.py --network mobile0.25

# With custom parameters
python train.py --network mobile0.25 --batch_size 32 --epochs 350 --lr 1e-3
```

**Features**:
- Paper-compliant V1 architecture (489K parameters)
- Standard training pipeline
- Supports resume from checkpoint
- Automatic weight saving

### `train_v2.py` - V2 Training with Knowledge Distillation
Advanced training script for FeatherFace V2 (256K parameters) with knowledge distillation.

```bash
# Basic usage
python train_v2.py --teacher_model weights/mobilenet0.25_Final.pth

# With advanced options
python train_v2.py \
    --teacher_model weights/mobilenet0.25_Final.pth \
    --epochs 400 \
    --temperature 4.0 \
    --alpha 0.7 \
    --mixup_alpha 0.2 \
    --cutmix_prob 0.5
```

**Features**:
- Knowledge distillation from V1 teacher model
- Advanced data augmentation (MixUp, CutMix, DropBlock)
- Dynamic α scheduling for distillation
- Smart early stopping
- Gradient clipping and monitoring
- Real-time performance tracking

### `start_v2_training.py` - Quick V2 Starter
Simplified wrapper for V2 training with sensible defaults.

```bash
# Quick start
python start_v2_training.py

# With custom teacher model
python start_v2_training.py --teacher_model path/to/teacher.pth
```

**Features**:
- Preconfigured optimal settings
- Automatic teacher model validation
- Quick setup for experimentation

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

### V2 Training (`train_v2.py`)
All V1 parameters plus:
- `--teacher_model`: Path to teacher model weights (required)
- `--temperature`: Distillation temperature (default: 4.0)
- `--alpha`: Distillation weight (default: 0.7)
- `--feature_weight`: Feature distillation weight (default: 0.1)
- `--mixup_alpha`: MixUp augmentation strength (default: 0.2)
- `--cutmix_prob`: CutMix probability (default: 0.5)
- `--dropblock_prob`: DropBlock probability (default: 0.1)
- `--gradient_clip`: Gradient clipping norm (default: 1.0)

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
python train_v2.py --mixed_precision

# Optimize batch size for your GPU
python scripts/validation/validate_parameters.py --optimize_batch_size
```

### Memory Management
```bash
# For limited GPU memory
python train_v2.py --batch_size 16 --gradient_accumulation 2

# For high-memory GPUs
python train_v2.py --batch_size 64
```

## 📈 Expected Results

### V1 Training
- **Target**: 87.2% mAP on WIDERFace
- **Parameters**: 489K (paper-compliant)
- **Training time**: ~6-8 hours on RTX 3080

### V2 Training
- **Target**: 89.0% mAP on WIDERFace
- **Parameters**: 256K (56.7% reduction)
- **Training time**: ~8-10 hours with knowledge distillation

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

- **Interactive Training**: [experiments/](../../experiments/) - Jupyter notebooks for development
- **Validation**: [../validation/](../validation/) - Model validation scripts
- **Monitoring Utils**: [utils/monitoring.py](../../utils/monitoring.py) - Training monitoring tools
- **GPU Utils**: [utils/gpu_optimization.py](../../utils/gpu_optimization.py) - GPU optimization

---

**Recommendation**: For first-time training or experimentation, use the Jupyter notebooks in `experiments/`. Use these command-line scripts for production training or automated pipelines.