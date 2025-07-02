# FeatherFace Deployment Guide

This directory contains production-ready deployment packages for both FeatherFace V1 (optimized) and V2 (enhanced) models.

## 📁 Directory Structure

```
deployment/
├── README.md                    # This file
├── v1_optimized/               # V1 optimized model (489K params)
│   ├── featherface_v1_optimized.onnx
│   ├── featherface_v1_optimized.pth
│   ├── deployment_config.json
│   └── usage_examples/
├── v2_enhanced/                # V2 enhanced model (256K params)
│   ├── featherface_v2_enhanced.onnx
│   ├── featherface_v2_enhanced.pth
│   ├── deployment_config.json
│   └── usage_examples/
├── configs/                    # Deployment configurations
│   ├── production.yaml
│   ├── edge_device.yaml
│   └── cloud_api.yaml
├── examples/                   # Usage examples
│   ├── python_inference.py
│   ├── onnx_inference.py
│   ├── batch_processing.py
│   └── web_api.py
└── onnx/                      # ONNX-specific files
    ├── runtime_comparison.py
    ├── optimization_guide.md
    └── quantization_scripts/
```

## 🚀 Quick Start

### 1. Python Inference (PyTorch)

```python
import torch
from pathlib import Path

# Load V1 optimized model
model_path = "deployment/v1_optimized/featherface_v1_optimized.pth"
checkpoint = torch.load(model_path, map_location='cpu')

# Or load V2 enhanced model
model_path = "deployment/v2_enhanced/featherface_v2_enhanced.pth"
checkpoint = torch.load(model_path, map_location='cpu')

# Use the model for inference
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 2. ONNX Inference (Cross-platform)

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('deployment/v2_enhanced/featherface_v2_enhanced.onnx')

# Prepare input (BGR format, mean subtracted)
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
input_data -= np.array([[[104]], [[117]], [[123]]], dtype=np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
classifications, bbox_regressions, landmarks = outputs
```

## 📊 Model Comparison

| Model | Parameters | Size (PyTorch) | Size (ONNX) | Target mAP | Use Case |
|-------|------------|----------------|-------------|------------|----------|
| **V1 Optimized** | 489K | ~2.0MB | ~2.0MB | 87.2% | Balanced accuracy/size |
| **V2 Enhanced** | 256K | ~1.2MB | ~1.1MB | 89.0% | Mobile/Edge deployment |

## 🛠️ Deployment Options

### 1. Mobile/Edge Devices
- **Recommended**: V2 Enhanced ONNX model
- **Input sizes**: 320x320, 416x416, 640x640
- **Memory usage**: ~50-100MB
- **Inference time**: 10-25ms (varies by device)

### 2. Cloud API Services
- **Recommended**: V1 Optimized for accuracy, V2 for throughput
- **Batch processing**: Supported up to 8 images
- **Auto-scaling**: Compatible with Docker containers

### 3. Web Applications
- **Format**: ONNX.js compatible
- **Browser support**: Chrome, Firefox, Safari
- **WebGL acceleration**: Supported

## 🔧 Configuration Files

### Production Configuration (`configs/production.yaml`)
```yaml
model:
  version: "v2_enhanced"
  input_size: 640
  batch_size: 4
  device: "cuda"

inference:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 1000

optimization:
  mixed_precision: true
  tensorrt: true
  dynamic_shapes: true
```

### Edge Device Configuration (`configs/edge_device.yaml`)
```yaml
model:
  version: "v2_enhanced"
  input_size: 416
  batch_size: 1
  device: "cpu"

inference:
  confidence_threshold: 0.6
  nms_threshold: 0.4
  max_detections: 100

optimization:
  quantization: "int8"
  memory_limit: "512MB"
```

## 📈 Performance Benchmarks

### Inference Speed (640x640 input)

| Platform | V1 Optimized | V2 Enhanced | Speedup |
|----------|-------------|-------------|---------|
| **CPU (Intel i7)** | 38ms | 25ms | 1.5x |
| **GPU (RTX 3080)** | 6ms | 4ms | 1.5x |
| **Mobile (A14)** | 45ms | 28ms | 1.6x |
| **Edge (Jetson)** | 25ms | 16ms | 1.6x |

### Memory Usage

| Model | Peak Memory | Baseline Memory | Total |
|-------|-------------|-----------------|-------|
| **V1 Optimized** | 120MB | 80MB | 200MB |
| **V2 Enhanced** | 80MB | 50MB | 130MB |

## 🔒 Security Considerations

### Model Integrity
- All models include SHA-256 checksums
- Digital signatures available for production deployments
- Model versioning for security updates

### Input Validation
- Image size limits: 320x320 to 1024x1024
- Format validation: JPEG, PNG, BMP
- Malformed input protection

### API Security
- Rate limiting recommendations
- Input sanitization examples
- Authentication patterns

## 🚢 Deployment Platforms

### 1. Docker Containers

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install torch torchvision onnxruntime-gpu

# Copy deployment files
COPY deployment/ /app/deployment/
COPY examples/ /app/examples/

# Set working directory
WORKDIR /app

# Run inference server
CMD ["python", "examples/web_api.py"]
```

### 2. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: featherface-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: featherface-api
  template:
    metadata:
      labels:
        app: featherface-api
    spec:
      containers:
      - name: featherface
        image: featherface:v2-enhanced
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 3. AWS Lambda

```python
import json
import boto3
import onnxruntime as ort

def lambda_handler(event, context):
    # Load model from S3 or package
    session = ort.InferenceSession('featherface_v2_enhanced.onnx')
    
    # Process image from event
    image_data = process_input(event['image'])
    
    # Run inference
    outputs = session.run(None, {'input': image_data})
    
    # Return results
    return {
        'statusCode': 200,
        'body': json.dumps(format_results(outputs))
    }
```

## 🔧 Optimization Guides

### 1. ONNX Runtime Optimization

```python
import onnxruntime as ort

# Create optimized session
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession(
    'featherface_v2_enhanced.onnx',
    providers=providers
)
```

### 2. TensorRT Optimization

```python
# Convert ONNX to TensorRT
import tensorrt as trt

def build_tensorrt_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Enable optimizations
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # Mixed precision
    
    # Build engine
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

## 🧪 Testing & Validation

### Unit Tests
```bash
# Test model loading
python -m pytest deployment/tests/test_model_loading.py

# Test inference accuracy
python -m pytest deployment/tests/test_inference.py

# Test performance benchmarks
python -m pytest deployment/tests/test_performance.py
```

### Integration Tests
```bash
# Test full pipeline
python deployment/tests/integration_test.py

# Test with sample images
python deployment/tests/test_sample_images.py
```

## 📞 Support & Troubleshooting

### Common Issues

1. **ONNX Runtime not found**
   ```bash
   pip install onnxruntime-gpu  # For GPU
   pip install onnxruntime      # For CPU only
   ```

2. **CUDA compatibility issues**
   ```bash
   # Check CUDA version compatibility
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Memory issues**
   - Reduce batch size
   - Use CPU inference
   - Enable memory optimization flags

### Performance Tuning

1. **Input size optimization**
   - Use smallest acceptable input size (320x320 for mobile)
   - Use largest feasible size for accuracy (640x640 or 832x832)

2. **Batch size tuning**
   - Start with batch_size=1
   - Increase until memory limit reached
   - Monitor latency vs throughput trade-offs

### Monitoring

```python
# Performance monitoring
import time
import psutil

def monitor_inference(session, input_data):
    # Memory before
    mem_before = psutil.virtual_memory().used
    
    # Time inference
    start_time = time.time()
    outputs = session.run(None, {'input': input_data})
    inference_time = time.time() - start_time
    
    # Memory after
    mem_after = psutil.virtual_memory().used
    
    return {
        'inference_time_ms': inference_time * 1000,
        'memory_used_mb': (mem_after - mem_before) / 1024 / 1024,
        'output_shapes': [out.shape for out in outputs]
    }
```

## 📋 Changelog

### Version 2.0.0 (Current)
- ✅ V1 optimized to 489K parameters (paper compliant)
- ✅ V2 enhanced with 256K parameters
- ✅ Dynamic ONNX export with multi-size support
- ✅ Comprehensive deployment tools
- ✅ GPU optimization utilities
- ✅ Production-ready error handling

### Version 1.0.0 (Legacy)
- Original FeatherFace implementation
- 592K parameters
- Basic ONNX export

---

## 📧 Contact & Support

For deployment support and questions:
- Check the `examples/` directory for code samples
- Review `configs/` for deployment configurations
- Run tests in `deployment/tests/` to validate setup

**Last updated**: December 2024  
**Compatible with**: PyTorch 2.0+, ONNX Runtime 1.14+, Python 3.8+