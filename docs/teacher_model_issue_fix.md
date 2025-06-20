# Teacher Model Issue Resolution Guide

## Problem
The saved model `mobilenet0.25_Final.pth` uses FPN architecture while the current RetinaFace implementation expects BiFPN architecture, causing a mismatch in layer names.

## Solutions

### Option 1: Train V2 Without Knowledge Distillation (Recommended for now)
```bash
# Simple training without teacher model
python train_v2_simple.py --epochs 100 --batch_size 32

# This will train V2 from scratch without distillation
# You'll get ~0.256M params but may need more epochs to reach 92% mAP
```

### Option 2: Check and Convert Teacher Model
```bash
# Analyze the teacher model
python check_teacher_model.py

# This will create a compatible model if possible
# Use the generated compatible model:
python train_v2.py --teacher_model ./weights/mobilenet0.25_compatible.pth
```

### Option 3: Train with Modified Script
```bash
# Use the updated train_v2.py with strict=False loading
python train_v2.py --teacher_model ./weights/mobilenet0.25_Final.pth

# This will load what it can and use partial knowledge distillation
```

### Option 4: Train a New Teacher Model First
```bash
# Train a fresh V1 model with current architecture
python train.py --epochs 100

# Then use it as teacher
python train_v2.py --teacher_model ./weights/new_teacher.pth
```

## Architecture Differences

### Original Model (FPN)
- Uses standard FPN (Feature Pyramid Network)
- Layers: `fpn.output1`, `fpn.output2`, `fpn.output3`
- Simpler architecture

### Current Model (BiFPN)
- Uses BiFPN (Bidirectional FPN)
- Layers: `bifpn.0`, `bifpn.1`, `bifpn.2`
- More complex with attention weights

## Quick Fix for Training

The easiest solution is to train V2 without distillation first:

```python
# In your notebook or script:
from train_v2_simple import train_v2_simple

# Train for 100 epochs
model = train_v2_simple(epochs=100, batch_size=32, lr=1e-3)
```

Once you have a working V2 model, you can:
1. Evaluate its performance
2. Use it as a baseline
3. Later train with distillation when you have a compatible teacher

## Expected Results

- **Without Distillation**: May need 150-200 epochs to reach 90%+ mAP
- **With Distillation**: Can reach 92%+ mAP in 100-150 epochs
- **Parameters**: Still 0.256M regardless of training method

## Next Steps

1. Start with `train_v2_simple.py` to get a working V2 model
2. Evaluate performance after 100 epochs
3. If needed, continue training or implement distillation later
4. Focus on deployment once you have satisfactory performance