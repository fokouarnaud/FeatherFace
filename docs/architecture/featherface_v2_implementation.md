# FeatherFace V2 Implementation Guide

## 🚀 Quick Start Implementation

### 1. Environment Setup
```bash
# Clone and setup
git clone https://github.com/dohun-mat/FeatherFace
cd FeatherFace
pip install -e .

# Verify V2 components
python -c "from models.featherface_v2_simple import FeatherFaceV2Simple; print('✅ V2 Ready')"
```

### 2. V2 Model Creation
```python
import torch
from models.featherface_v2_simple import FeatherFaceV2Simple
from data.config import cfg_v2

# Create V2 model
model = FeatherFaceV2Simple(cfg=cfg_v2, phase='train')
print(f"V2 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
dummy_input = torch.randn(1, 3, 640, 640)
outputs = model(dummy_input)
print(f"V2 Outputs: {[o.shape for o in outputs]}")
```

### 3. V2 Training Pipeline
```python
# Complete V2 training with knowledge distillation
python train_v2.py \
    --teacher_model weights/mobilenet0.25_Final.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --experiment_name v2_coordinate_attention
```

## 🔧 Implementation Details

### Coordinate Attention Integration
```python
# V2 Architecture with Coordinate Attention
class FeatherFaceV2Simple(nn.Module):
    def __init__(self, cfg, phase='train'):
        super(FeatherFaceV2Simple, self).__init__()
        
        # Standard V1 components
        self.backbone = MobileNetV1(cfg)
        self.fpn = BiFPN(cfg)
        self.ssh = SSH(cfg)
        
        # V2 Innovation: Coordinate Attention
        from models.attention_v2 import CoordinateAttention
        self.coordinate_attention = CoordinateAttention(
            inp=cfg['coordinate_attention_config']['input_channels'],
            oup=cfg['coordinate_attention_config']['output_channels'],
            reduction=cfg['coordinate_attention_config']['reduction']
        )
        
        # Detection heads
        self.ClassHead = self._make_class_head(cfg)
        self.BboxHead = self._make_bbox_head(cfg)
        self.LandmarkHead = self._make_landmark_head(cfg)
        
    def forward(self, inputs):
        # Backbone features
        backbone_features = self.backbone(inputs)
        
        # Feature pyramid
        fpn_features = self.fpn(backbone_features)
        
        # Apply coordinate attention
        enhanced_features = []
        for feat in fpn_features:
            enhanced_feat = self.coordinate_attention(feat)
            enhanced_features.append(enhanced_feat)
        
        # SSH processing
        ssh_features = self.ssh(enhanced_features)
        
        # Detection heads
        classifications = self.ClassHead(ssh_features)
        bbox_regressions = self.BboxHead(ssh_features)
        ldm_regressions = self.LandmarkHead(ssh_features)
        
        return bbox_regressions, classifications, ldm_regressions
```

### Knowledge Distillation Implementation
```python
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_outputs, teacher_outputs, targets, priors, task_criterion):
        # Task loss (student vs ground truth)
        student_bbox, student_cls, student_landmark = student_outputs
        task_loss_l, task_loss_c, task_loss_landm = task_criterion(
            student_outputs, priors, targets
        )
        task_loss = 2.0 * task_loss_l + task_loss_c + task_loss_landm
        
        # Distillation loss (student vs teacher)
        teacher_bbox, teacher_cls, teacher_landmark = teacher_outputs
        
        # Classification distillation
        student_cls_soft = F.log_softmax(student_cls / self.temperature, dim=-1)
        teacher_cls_soft = F.softmax(teacher_cls / self.temperature, dim=-1)
        distill_loss_cls = nn.KLDivLoss(reduction='batchmean')(
            student_cls_soft, teacher_cls_soft
        ) * (self.temperature ** 2)
        
        # Regression distillation
        distill_loss_bbox = F.mse_loss(student_bbox, teacher_bbox.detach())
        distill_loss_landmark = F.mse_loss(student_landmark, teacher_landmark.detach())
        
        # Combined loss
        distill_loss = distill_loss_cls + distill_loss_bbox + distill_loss_landmark
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item(),
            'total_loss': total_loss.item()
        }
```

## 🎯 Training Configuration

### V2 Configuration (data/config.py)
```python
cfg_v2 = {
    'name': 'featherface_v2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'optim': 'adamw',
    'lr': 1e-4,
    
    # V2 specific configuration
    'coordinate_attention_config': {
        'input_channels': 64,
        'output_channels': 64,
        'reduction': 32,
        'use_activation': True
    },
    
    'attention_mechanism': 'coordinate_attention',
    'model_version': 'v2',
    'innovation': 'coordinate_attention'
}
```

### Training Parameters
```python
# V2 Training Configuration
TRAINING_CONFIG = {
    'teacher_model': 'weights/mobilenet0.25_Final.pth',
    'temperature': 4.0,
    'alpha': 0.7,
    'epochs': 250,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 5e-4,
    'optimizer': 'adamw',
    'scheduler': 'onecycle',
    'save_folder': 'weights/v2/'
}
```

## 📊 Performance Monitoring

### Training Monitoring
```python
# Monitor training progress
def monitor_training(epoch, iteration, loss_dict, lr):
    if iteration % 10 == 0:
        print(f'Epoch:{epoch} || Iter:{iteration} || '
              f'Total:{loss_dict["total_loss"]:.4f} || '
              f'Task:{loss_dict["task_loss"]:.4f} || '
              f'Distill:{loss_dict["distill_loss"]:.4f} || '
              f'LR:{lr:.8f}')
    
    # Detailed logging every 100 iterations
    if iteration % 100 == 0:
        print(f"  📊 Detailed Loss - "
              f"Cls: {loss_dict['task_cls']:.4f} | "
              f"Bbox: {loss_dict['task_loc']:.4f} | "
              f"Landmark: {loss_dict['task_landmark']:.4f}")
```

### Model Validation
```python
# V2 Model Validation
def validate_v2_model(model, dataloader, device):
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # Quick validation
                break
                
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]
            
            # V2 predictions
            outputs = model(images)
            
            # Performance analysis
            attention_maps = model.get_attention_maps(images)
            performance_stats = model.get_performance_stats()
            
            print(f"Batch {batch_idx}: Attention maps generated")
    
    model.train()
    return True
```

## 🔍 Testing & Validation

### Unit Testing
```python
# Test V2 components
def test_v2_components():
    # Test model creation
    model = FeatherFaceV2Simple(cfg=cfg_v2, phase='train')
    assert model is not None
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    assert len(outputs) == 3
    
    # Test attention maps
    attention_maps = model.get_attention_maps(dummy_input)
    assert len(attention_maps) > 0
    
    # Test performance comparison
    from models.retinaface import RetinaFace
    v1_model = RetinaFace(cfg=cfg_mnet, phase='train')
    comparison = model.compare_with_v1(v1_model)
    assert comparison['v2_parameters'] > comparison['v1_parameters']
    
    print("✅ All V2 tests passed!")
```

### Integration Testing
```bash
# Run V2 validation script
python test_v2_training.py

# Expected output:
# ✅ Models test passed
# ✅ Knowledge distillation test passed
# ✅ Training pipeline test passed
# ✅ Performance comparison test passed
```

## 📱 Deployment Guide

### Model Export
```python
# Export V2 model for deployment
def export_v2_model(model, input_size=(640, 640)):
    model.eval()
    
    # PyTorch export
    torch.save(model.state_dict(), 'featherface_v2.pth')
    
    # ONNX export
    dummy_input = torch.randn(1, 3, *input_size)
    torch.onnx.export(
        model, 
        dummy_input, 
        'featherface_v2.onnx',
        input_names=['input'],
        output_names=['bbox', 'classification', 'landmark'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
    
    # TorchScript export
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save('featherface_v2_traced.pt')
    
    print("✅ V2 model exported successfully!")
```

### Mobile Optimization
```python
# Mobile deployment optimization
def optimize_for_mobile(model):
    # Quantization
    model.eval()
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Mobile-specific optimizations
    model_mobile = torch.jit.optimize_for_mobile(
        torch.jit.script(model)
    )
    
    return model_quantized, model_mobile
```

## 🎯 Best Practices

### Training Tips
1. **Start with V1 teacher**: Always use pre-trained V1 as teacher
2. **Temperature tuning**: Start with 4.0, adjust based on convergence
3. **Alpha balancing**: Use 0.7 for balanced task/distillation loss
4. **Batch size**: 32 works well for most GPUs
5. **Learning rate**: 1e-4 with AdamW optimizer

### Performance Optimization
1. **Attention placement**: Apply coordinate attention after BiFPN
2. **Memory management**: Use gradient checkpointing for large batches
3. **Mixed precision**: Enable AMP for faster training
4. **Data loading**: Use multiple workers for data loading

### Common Issues
1. **Memory errors**: Reduce batch size or use gradient checkpointing
2. **Convergence issues**: Adjust temperature or alpha parameters
3. **Attention maps**: Ensure proper attention module initialization
4. **Export issues**: Check model compatibility for deployment

## 📚 Additional Resources

- **[V2 Training Notebook](../../notebooks/02_train_evaluate_featherface_v2.ipynb)**
- **[V2 Architecture Details](featherface_v2.md)**
- **[V2 Performance Analysis](featherface_v2_performance.md)**
- **[Coordinate Attention Paper](https://arxiv.org/abs/2103.02907)**

---

**Status**: ✅ Implementation Ready  
**Version**: V2.0  
**Last Updated**: January 2025  
**Support**: Full production support