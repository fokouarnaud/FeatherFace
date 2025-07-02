#!/usr/bin/env python3
"""
Dynamic ONNX Export for FeatherFace V1 & V2
Exports models with dynamic input sizes for production deployment
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import argparse
import os
import sys
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retinaface import RetinaFace
from models.retinaface_v2 import get_retinaface_v2
from data.config import cfg_mnet, cfg_mnet_v2


def export_dynamic_onnx(model: nn.Module,
                       output_path: str,
                       model_name: str = "FeatherFace",
                       input_size: tuple = (640, 640),
                       opset_version: int = 14) -> bool:
    """
    Export model to ONNX with dynamic input sizes
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        model_name: Name for the model
        input_size: Default input size (height, width)
        opset_version: ONNX opset version
    
    Returns:
        bool: Success status
    """
    try:
        print(f"🚀 Exporting {model_name} to ONNX...")
        
        # Prepare model
        model.eval()
        model = model.cpu()  # Use CPU for maximum compatibility
        
        # Create dummy input
        batch_size = 1
        channels = 3
        height, width = input_size
        dummy_input = torch.randn(batch_size, channels, height, width)
        
        print(f"   Input shape: {dummy_input.shape}")
        
        # Test forward pass first
        with torch.no_grad():
            test_outputs = model(dummy_input)
        print(f"   Test outputs: {len(test_outputs)} tensors")
        print(f"   Output shapes: {[out.shape for out in test_outputs]}")
        
        # Export to ONNX with dynamic axes
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['bbox_regressions', 'classifications', 'landmarks'],
            dynamic_axes={
                # Dynamic batch size and spatial dimensions
                'input': {
                    0: 'batch_size',
                    2: 'height', 
                    3: 'width'
                },
                'bbox_regressions': {
                    0: 'batch_size',
                    1: 'num_anchors'  # Anchors scale with input size
                },
                'classifications': {
                    0: 'batch_size',
                    1: 'num_anchors'
                },
                'landmarks': {
                    0: 'batch_size',
                    1: 'num_anchors'
                }
            },
            verbose=False
        )
        
        # Get file size
        file_size = os.path.getsize(output_path)
        print(f"   ✅ Export successful: {output_path}")
        print(f"   📦 File size: {file_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return False


def validate_dynamic_onnx(onnx_path: str, test_sizes: list) -> bool:
    """
    Validate ONNX model with multiple input sizes
    
    Args:
        onnx_path: Path to ONNX model
        test_sizes: List of (batch, channels, height, width) tuples to test
    
    Returns:
        bool: Validation success
    """
    try:
        print(f"🧪 Validating ONNX model: {os.path.basename(onnx_path)}")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"   ✅ ONNX model structure valid")
        
        # Create inference session
        providers = ['CPUExecutionProvider']  # Use CPU for compatibility
        if ort.get_available_providers():
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"   🔧 Inference provider: {session.get_providers()[0]}")
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"   📥 Input: {input_info.name} - Shape: {input_info.shape}")
        print(f"   📤 Outputs: {len(output_info)} tensors")
        for i, out in enumerate(output_info):
            print(f"      {i+1}. {out.name} - Shape: {out.shape}")
        
        # Test different input sizes
        print(f"   🎯 Testing {len(test_sizes)} different input sizes...")
        
        all_passed = True
        for i, (batch_size, channels, height, width) in enumerate(test_sizes):
            try:
                # Create test input
                test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
                
                # Run inference
                start_time = time.time()
                outputs = session.run(None, {input_info.name: test_input})
                inference_time = (time.time() - start_time) * 1000
                
                print(f"      ✅ Size {(batch_size, channels, height, width)}: "
                      f"{inference_time:.1f}ms")
                print(f"         Output shapes: {[out.shape for out in outputs]}")
                
                # Validate output shapes make sense
                expected_anchors = calculate_expected_anchors(height, width)
                actual_anchors = outputs[0].shape[1] if len(outputs) > 0 else 0
                
                if abs(expected_anchors - actual_anchors) / expected_anchors > 0.1:  # 10% tolerance
                    print(f"         ⚠️  Anchor count mismatch: expected ~{expected_anchors}, got {actual_anchors}")
                
            except Exception as e:
                print(f"      ❌ Size {(batch_size, channels, height, width)}: Failed - {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False


def calculate_expected_anchors(height: int, width: int) -> int:
    """
    Calculate expected number of anchors for given input size
    Based on FeatherFace architecture with 3 FPN levels
    """
    # FPN strides: [8, 16, 32]
    # 2 anchors per location
    anchors = 0
    for stride in [8, 16, 32]:
        feat_h = height // stride
        feat_w = width // stride
        anchors += feat_h * feat_w * 2
    
    return anchors


def create_deployment_package(model_path: str, onnx_path: str, config: dict, output_dir: str):
    """
    Create a complete deployment package with model, config, and documentation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model files
    import shutil
    if os.path.exists(model_path):
        shutil.copy2(model_path, os.path.join(output_dir, 'model.pth'))
    shutil.copy2(onnx_path, os.path.join(output_dir, 'model.onnx'))
    
    # Create deployment config
    deployment_config = {
        'model_info': {
            'name': 'FeatherFace',
            'version': '2.0',
            'parameters': config.get('parameters', 'unknown'),
            'input_format': 'BGR',
            'input_mean': [104, 117, 123],
            'input_std': [1, 1, 1],
            'input_size_default': [640, 640],
            'input_size_range': [[320, 320], [1024, 1024]]
        },
        'preprocessing': {
            'resize_method': 'bilinear',
            'maintain_aspect_ratio': False,
            'normalize': True,
            'mean_subtraction': True
        },
        'postprocessing': {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'top_k': 5000,
            'keep_top_k': 750
        },
        'deployment': {
            'recommended_batch_size': 1,
            'supports_dynamic_shapes': True,
            'min_input_size': [320, 320],
            'max_input_size': [1024, 1024],
            'optimal_input_sizes': [[416, 416], [640, 640], [832, 832]]
        }
    }
    
    # Save config
    import json
    with open(os.path.join(output_dir, 'deployment_config.json'), 'w') as f:
        json.dump(deployment_config, f, indent=2)
    
    # Create usage example
    usage_example = '''# FeatherFace ONNX Usage Example

import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx')

# Load and preprocess image
image = cv2.imread('face.jpg')
height, width = image.shape[:2]

# Resize to model input size (can be dynamic)
input_size = 640
image_resized = cv2.resize(image, (input_size, input_size))

# Normalize (BGR format, mean subtraction)
image_norm = image_resized.astype(np.float32)
image_norm -= np.array([104, 117, 123])  # BGR mean
image_norm = np.transpose(image_norm, (2, 0, 1))  # HWC -> CHW
image_input = np.expand_dims(image_norm, 0)  # Add batch dimension

# Run inference
outputs = session.run(None, {'input': image_input})
bbox_regressions, classifications, landmarks = outputs

# Process outputs (add your postprocessing here)
print(f"Detected {classifications.shape[1]} potential faces")
'''
    
    with open(os.path.join(output_dir, 'usage_example.py'), 'w') as f:
        f.write(usage_example)
    
    # Create README
    readme_content = f'''# FeatherFace Deployment Package

## Contents
- `model.pth`: PyTorch model weights
- `model.onnx`: ONNX model for cross-platform deployment  
- `deployment_config.json`: Model configuration and parameters
- `usage_example.py`: Python usage example

## Model Information
- Architecture: FeatherFace V2 optimized
- Parameters: {config.get('parameters', 'unknown')}
- Input: Dynamic size (minimum 320x320, maximum 1024x1024)
- Output: Face bounding boxes, classifications, landmarks

## Quick Start
```python
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
# See usage_example.py for complete example
```

## Performance Tips
1. Use batch size 1 for lowest latency
2. Optimal input sizes: 416x416, 640x640, 832x832
3. Enable GPU acceleration when available
4. Consider quantization for edge deployment

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
'''
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"📦 Deployment package created: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export FeatherFace to ONNX with dynamic sizes')
    parser.add_argument('--model', choices=['v1', 'v2'], default='v2',
                       help='Model version to export')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default='./exports/',
                       help='Output directory')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640],
                       help='Default input size (height width)')
    parser.add_argument('--test_sizes', action='store_true',
                       help='Test multiple input sizes after export')
    parser.add_argument('--deployment_package', action='store_true',
                       help='Create complete deployment package')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🎯 FeatherFace ONNX Export Tool")
    print("="*50)
    
    try:
        # Load model
        print(f"📥 Loading {args.model.upper()} model...")
        
        if args.model == 'v1':
            model = RetinaFace(cfg=cfg_mnet, phase='test')
            config = cfg_mnet
            model_name = "FeatherFace_V1"
        else:
            model = get_retinaface_v2(cfg_mnet_v2, phase='test')
            config = cfg_mnet_v2
            model_name = "FeatherFace_V2"
        
        # Load weights
        if os.path.exists(args.weights):
            checkpoint = torch.load(args.weights, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ Weights loaded: {args.weights}")
        else:
            print(f"   ⚠️  No weights file found, using random initialization")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 Model parameters: {total_params:,} ({total_params/1e6:.3f}M)")
        
        # Export to ONNX
        onnx_path = os.path.join(args.output, f'{model_name}_dynamic.onnx')
        success = export_dynamic_onnx(
            model=model,
            output_path=onnx_path,
            model_name=model_name,
            input_size=tuple(args.input_size)
        )
        
        if not success:
            return 1
        
        # Test multiple sizes if requested
        if args.test_sizes:
            test_sizes = [
                (1, 3, 320, 320),   # Minimum size
                (1, 3, 416, 416),   # Small
                (1, 3, 640, 640),   # Default  
                (1, 3, 832, 832),   # Large
                (2, 3, 640, 640),   # Batch size 2
            ]
            
            validation_success = validate_dynamic_onnx(onnx_path, test_sizes)
            if not validation_success:
                print("⚠️  Some validation tests failed")
        
        # Create deployment package if requested
        if args.deployment_package:
            package_dir = os.path.join(args.output, f'{model_name}_deployment')
            create_deployment_package(
                model_path=args.weights,
                onnx_path=onnx_path,
                config={'parameters': f'{total_params:,}'},
                output_dir=package_dir
            )
        
        print("\n🎉 Export completed successfully!")
        print(f"   ONNX model: {onnx_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import time
    exit(main())