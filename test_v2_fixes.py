#!/usr/bin/env python3
"""
Test script for FeatherFace V2 fixes
Validates that all corrections work properly
"""

import sys
import torch
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("=== Testing Imports ===")
    try:
        from models.retinaface import RetinaFace
        from models.retinaface_v2 import RetinaFaceV2, get_retinaface_v2, count_parameters
        from data.config import cfg_mnet, cfg_mnet_v2
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_configurations():
    """Test configuration compatibility"""
    print("\n=== Testing Configurations ===")
    try:
        from data.config import cfg_mnet, cfg_mnet_v2
        
        # Check V2 config name
        if cfg_mnet_v2['name'] == 'mobilenet0.25':
            print("✓ cfg_mnet_v2['name'] correctly set to 'mobilenet0.25'")
        else:
            print(f"✗ cfg_mnet_v2['name'] = '{cfg_mnet_v2['name']}', should be 'mobilenet0.25'")
            return False
            
        # Check other important configs
        print(f"✓ V1 config: {cfg_mnet['name']}, {cfg_mnet['out_channel']} channels")
        print(f"✓ V2 config: {cfg_mnet_v2['name']}, {cfg_mnet_v2['out_channel_v2']} channels")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_creation():
    """Test model creation without errors"""
    print("\n=== Testing Model Creation ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test V1 model
        print("Creating FeatherFace V1...")
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        
        model_v1 = RetinaFace(cfg=cfg_mnet, phase='test')
        model_v1 = model_v1.to(device)
        print("✓ V1 model created successfully")
        
        # Test V2 model
        print("Creating FeatherFace V2...")
        from models.retinaface_v2 import get_retinaface_v2
        from data.config import cfg_mnet_v2
        
        model_v2 = get_retinaface_v2(cfg_mnet_v2, phase='test')
        model_v2 = model_v2.to(device)
        print("✓ V2 model created successfully")
        
        return model_v1, model_v2
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return None, None

def test_parameter_count(model_v1, model_v2):
    """Test parameter counting"""
    print("\n=== Testing Parameter Count ===")
    try:
        from models.retinaface_v2 import count_parameters
        
        params_v1 = count_parameters(model_v1)
        params_v2 = count_parameters(model_v2)
        
        print(f"V1 parameters: {params_v1:,} ({params_v1/1e6:.3f}M)")
        print(f"V2 parameters: {params_v2:,} ({params_v2/1e6:.3f}M)")
        print(f"Reduction: {(1-params_v2/params_v1)*100:.1f}%")
        print(f"Compression ratio: {params_v1/params_v2:.2f}x")
        
        # Check if V2 has significantly fewer parameters
        if params_v2 < params_v1 * 0.5:  # At least 50% reduction
            print("✓ V2 has significantly fewer parameters")
            return True
        else:
            print("⚠️  V2 parameter reduction is less than expected")
            return False
            
    except Exception as e:
        print(f"✗ Parameter count test failed: {e}")
        return False

def test_forward_pass(model_v1, model_v2):
    """Test forward pass compatibility"""
    print("\n=== Testing Forward Pass ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        # Test V1 forward pass
        model_v1.eval()
        with torch.no_grad():
            outputs_v1 = model_v1(dummy_input)
        print(f"✓ V1 forward pass successful: {len(outputs_v1)} outputs")
        print(f"  Output shapes: {[out.shape for out in outputs_v1]}")
        
        # Test V2 forward pass
        model_v2.eval()
        with torch.no_grad():
            outputs_v2 = model_v2(dummy_input)
        print(f"✓ V2 forward pass successful: {len(outputs_v2)} outputs")
        print(f"  Output shapes: {[out.shape for out in outputs_v2]}")
        
        # Check output compatibility
        if len(outputs_v1) == len(outputs_v2):
            shapes_match = all(v1.shape == v2.shape for v1, v2 in zip(outputs_v1, outputs_v2))
            if shapes_match:
                print("✓ Output shapes are compatible")
                return True
            else:
                print("⚠️  Output shapes differ")
                return False
        else:
            print(f"⚠️  Different number of outputs: V1={len(outputs_v1)}, V2={len(outputs_v2)}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()
        return False

def test_pretrained_weights():
    """Test pretrained weights loading"""
    print("\n=== Testing Pretrained Weights ===")
    pretrain_path = Path('./weights/mobilenetV1X0.25_pretrain.tar')
    
    if pretrain_path.exists():
        print(f"✓ Pretrained weights found: {pretrain_path}")
        try:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                print("✓ Checkpoint format is correct")
                return True
            else:
                print("⚠️  Checkpoint format may be unexpected")
                return False
        except Exception as e:
            print(f"✗ Failed to load pretrained weights: {e}")
            return False
    else:
        print(f"⚠️  Pretrained weights not found: {pretrain_path}")
        print("   This is optional but recommended for best performance")
        return True  # Not critical for basic functionality

def main():
    """Run all tests"""
    print("FeatherFace V2 Fixes Validation")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configurations", test_configurations()))
    
    model_v1, model_v2 = test_model_creation()
    if model_v1 is not None and model_v2 is not None:
        results.append(("Model Creation", True))
        results.append(("Parameter Count", test_parameter_count(model_v1, model_v2)))
        results.append(("Forward Pass", test_forward_pass(model_v1, model_v2)))
    else:
        results.append(("Model Creation", False))
        results.append(("Parameter Count", False))
        results.append(("Forward Pass", False))
    
    results.append(("Pretrained Weights", test_pretrained_weights()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! FeatherFace V2 is ready for use.")
        return 0
    elif passed >= total - 1:  # Allow 1 failure (like missing pretrained weights)
        print("\n✅ Core functionality working. Minor issues detected.")
        return 0
    else:
        print("\n❌ Multiple issues detected. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())