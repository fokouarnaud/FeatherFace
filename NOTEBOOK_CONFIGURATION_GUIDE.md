# 📘 Notebook Configuration Guide - ECA-CBAM Training

**Notebook**: `notebooks/02_train_eca_cbam.ipynb`
**Date**: 2025-11-13
**Purpose**: Flexible CPU/GPU usage and training control

---

## 🎯 Quick Start

### Configuration Cell (Cell 3)

All notebook behavior is controlled by these variables at the top of **Cell 3**:

```python
# Device configuration
USE_GPU_FOR_TRAINING = True      # Use GPU for training (recommended)
USE_GPU_FOR_EVALUATION = False   # Use GPU for evaluation (can use CPU to save GPU)
USE_GPU_FOR_EXPORT = False       # Use GPU for export (can use CPU to save GPU)

# Training configuration
SKIP_TRAINING = True             # Skip training if model already exists
FORCE_TRAINING = False           # Force training even if model exists

# Model paths
TRAINED_MODEL_PATH = 'weights/eca_cbam/featherface_eca_cbam_final.pth'
```

---

## 📋 Common Use Cases

### 1. ⚡ First Time Training (Full GPU)

**Goal**: Train the model from scratch using GPU

```python
USE_GPU_FOR_TRAINING = True
USE_GPU_FOR_EVALUATION = True
USE_GPU_FOR_EXPORT = True
SKIP_TRAINING = False
FORCE_TRAINING = False
```

**What happens**:
- ✅ Cell 13: Trains model on GPU (~6-10 hours)
- ✅ Cell 17: Evaluates on GPU
- ✅ Cell 19: Exports using GPU

---

### 2. 💾 Evaluation Only (Save GPU - Recommended)

**Goal**: Skip training, use CPU for evaluation and export

```python
USE_GPU_FOR_TRAINING = True      # Not used (training skipped)
USE_GPU_FOR_EVALUATION = False   # Use CPU
USE_GPU_FOR_EXPORT = False       # Use CPU
SKIP_TRAINING = True             # Skip training
FORCE_TRAINING = False
```

**What happens**:
- ⏭️ Cell 13: **Training SKIPPED** (model exists)
- ✅ Cell 17: Evaluates on **CPU** (saves GPU memory)
- ✅ Cell 19: Exports using **CPU** (saves GPU memory)

**Benefits**:
- 💰 Saves GPU costs
- ⚡ Frees GPU for other tasks
- ✅ Still works perfectly (just slower)

---

### 3. 🔄 Re-train Model (Force Training)

**Goal**: Re-train even if model exists

```python
USE_GPU_FOR_TRAINING = True
USE_GPU_FOR_EVALUATION = True
USE_GPU_FOR_EXPORT = False
SKIP_TRAINING = False            # Don't skip
FORCE_TRAINING = True            # Force re-training
```

**What happens**:
- ✅ Cell 13: **Trains AGAIN** even if model exists
- ✅ Cell 17: Evaluates on GPU
- ✅ Cell 19: Exports using CPU

---

### 4. 🚀 Hybrid: GPU Training, CPU Eval/Export

**Goal**: Use GPU only where needed most

```python
USE_GPU_FOR_TRAINING = True      # GPU for training (fast)
USE_GPU_FOR_EVALUATION = False   # CPU for eval (save GPU)
USE_GPU_FOR_EXPORT = False       # CPU for export (save GPU)
SKIP_TRAINING = True             # Skip if model exists
FORCE_TRAINING = False
```

**What happens**:
- ⏭️ Cell 13: Skipped (or uses GPU if needed)
- ✅ Cell 17: Evaluates on **CPU**
- ✅ Cell 19: Exports using **CPU**

**Best for**: Saving GPU while still having fast training option

---

## 🔧 Configuration Reference

### Device Settings

| Variable | Values | Purpose |
|----------|--------|---------|
| `USE_GPU_FOR_TRAINING` | `True`/`False` | Use GPU for training (Cell 13) |
| `USE_GPU_FOR_EVALUATION` | `True`/`False` | Use GPU for evaluation (Cell 17) |
| `USE_GPU_FOR_EXPORT` | `True`/`False` | Use GPU for export (Cell 19) |

**Notes**:
- If CUDA not available, all automatically use CPU
- CPU is slower but works perfectly
- GPU saves time but costs more

---

### Training Control

| Variable | Values | Purpose |
|----------|--------|---------|
| `SKIP_TRAINING` | `True`/`False` | Skip training if model exists |
| `FORCE_TRAINING` | `True`/`False` | Force training even if model exists |

**Logic**:
```python
if SKIP_TRAINING and model_exists and not FORCE_TRAINING:
    # Training is SKIPPED
else:
    # Training RUNS
```

**Examples**:
- `SKIP_TRAINING=True, FORCE_TRAINING=False, model exists` → ⏭️ **SKIP**
- `SKIP_TRAINING=True, FORCE_TRAINING=True, model exists` → ✅ **TRAIN** (force)
- `SKIP_TRAINING=False, model exists` → ✅ **TRAIN**
- `SKIP_TRAINING=True, model doesn't exist` → ✅ **TRAIN** (required)

---

## 📊 Performance Comparison

### GPU vs CPU Performance

| Operation | GPU Time | CPU Time | Speedup |
|-----------|----------|----------|---------|
| Training (350 epochs) | ~6-10 hours | ~48-72 hours | **~8x** |
| Evaluation (3,226 images) | ~5-10 minutes | ~15-30 minutes | **~3x** |
| Export (3 formats) | ~1-2 minutes | ~2-4 minutes | **~2x** |

**Recommendation**:
- 🎯 **Training**: Always use GPU (huge speedup)
- ⚖️ **Evaluation**: CPU is fine (saves GPU)
- ⚖️ **Export**: CPU is fine (saves GPU)

---

## 🎓 Step-by-Step Workflow

### Scenario: Just Run Evaluation and Export

**Goal**: You already trained the model, just want to evaluate and export

**Steps**:

1. **Open Notebook**
   ```
   notebooks/02_train_eca_cbam.ipynb
   ```

2. **Edit Cell 3** (Configuration)
   ```python
   USE_GPU_FOR_EVALUATION = False   # Use CPU to save GPU
   USE_GPU_FOR_EXPORT = False       # Use CPU to save GPU
   SKIP_TRAINING = True             # Skip training
   ```

3. **Run All Cells**
   - Cells 1-12: Setup and validation (fast)
   - Cell 13: Training **SKIPPED** ⏭️
   - Cell 17: Evaluation on **CPU** ✅
   - Cell 19: Export on **CPU** ✅
   - Cell 21: Summary ✅

4. **Results**
   - Evaluation: `widerface_evaluate/widerface_txt/`
   - Exports: `exports/eca_cbam/`
   - mAP scores: Displayed in Cell 17

---

## 💡 Tips and Tricks

### 1. Save GPU Memory

If you're running low on GPU memory:
```python
USE_GPU_FOR_EVALUATION = False
USE_GPU_FOR_EXPORT = False
```

### 2. Quick Iteration

For testing changes without full training:
```python
SKIP_TRAINING = True  # Skip lengthy training
```

### 3. Force Fresh Training

To re-train from scratch:
```python
FORCE_TRAINING = True
```

### 4. Check What Will Happen

After setting config, Cell 3 shows:
```
📋 USER CONFIGURATION:
  • GPU for training: ✅ ENABLED
  • GPU for evaluation: ❌ DISABLED (CPU)
  • GPU for export: ❌ DISABLED (CPU)
  • Skip training: ✅ YES
  • Force training: ❌ NO

✅ Trained model found: weights/eca_cbam/featherface_eca_cbam_final.pth
   → Training will be SKIPPED (model exists)
```

---

## 🚨 Troubleshooting

### Issue: Training takes too long on CPU

**Solution**: Use GPU for training
```python
USE_GPU_FOR_TRAINING = True
```

### Issue: Out of GPU memory during evaluation

**Solution**: Use CPU for evaluation
```python
USE_GPU_FOR_EVALUATION = False
```

### Issue: Model not found

**Error**: `❌ Trained model NOT found`

**Solution**: Either:
1. Train the model: `SKIP_TRAINING = False`
2. Check model path: `TRAINED_MODEL_PATH = '...'`

### Issue: Training runs when I don't want it to

**Solution**: Make sure:
```python
SKIP_TRAINING = True
FORCE_TRAINING = False
```

And verify model exists at: `weights/eca_cbam/featherface_eca_cbam_final.pth`

---

## 📁 File Locations

### Model Files
- **Trained model**: `weights/eca_cbam/featherface_eca_cbam_final.pth`
- **Checkpoints**: `weights/eca_cbam/featherface_eca_cbam_epoch_*.pth`

### Output Files
- **Evaluation results**: `widerface_evaluate/widerface_txt/`
- **Exports**: `exports/eca_cbam/`
  - PyTorch: `featherface_eca_cbam_hybrid.pth`
  - ONNX: `featherface_eca_cbam_hybrid.onnx`
  - TorchScript: `featherface_eca_cbam_hybrid.pt`

---

## ✅ Summary

**Configuration is in Cell 3** - modify these variables to control:
- ✅ Which device to use (GPU vs CPU)
- ✅ Whether to skip training
- ✅ Whether to force re-training

**Default for evaluation/export only**:
```python
USE_GPU_FOR_EVALUATION = False   # Save GPU
USE_GPU_FOR_EXPORT = False       # Save GPU
SKIP_TRAINING = True             # Don't re-train
```

**Run all cells** - the notebook will:
- ⏭️ Skip training if model exists
- ✅ Run evaluation on CPU
- ✅ Export model on CPU
- ✅ Show results

---

**Status**: ✅ READY TO USE
**Last Updated**: 2025-11-13
**Configuration Cell**: Cell 3
