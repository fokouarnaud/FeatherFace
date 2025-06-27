#!/usr/bin/env python3
"""
Test script to verify V1 and V2 output alignment
Validates that knowledge distillation will work correctly
"""

import sys
import torch
import traceback
from pathlib import Path

def test_output_alignment():
    """Test that V1 and V2 outputs are properly aligned"""
    print("=== Testing V1 vs V2 Output Alignment ===")
    
    try:
        # Import models
        from models.retinaface import RetinaFace
        from models.retinaface_v2 import get_retinaface_v2, count_parameters
        from data.config import cfg_mnet, cfg_mnet_v2
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load models
        print("\nLoading models...")
        model_v1 = RetinaFace(cfg=cfg_mnet, phase='test').to(device).eval()
        model_v2 = get_retinaface_v2(cfg_mnet_v2, phase='test').to(device).eval()
        
        # Count parameters
        params_v1 = count_parameters(model_v1)
        params_v2 = count_parameters(model_v2)
        
        print(f"✓ V1 loaded: {params_v1:,} parameters ({params_v1/1e6:.3f}M)")
        print(f"✓ V2 loaded: {params_v2:,} parameters ({params_v2/1e6:.3f}M)")
        print(f"✓ Compression: {params_v1/params_v2:.2f}x ({(1-params_v2/params_v1)*100:.1f}% reduction)")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            outputs_v1 = model_v1(dummy_input)
            outputs_v2 = model_v2(dummy_input)
        
        print(f"\nOutput analysis:")
        print(f"V1 outputs: {len(outputs_v1)} tensors")
        print(f"V2 outputs: {len(outputs_v2)} tensors")
        
        # Check shapes
        shapes_v1 = [out.shape for out in outputs_v1]
        shapes_v2 = [out.shape for out in outputs_v2]
        
        print(f"\nV1 shapes: {shapes_v1}")
        print(f"V2 shapes: {shapes_v2}")
        
        # Validate alignment
        if len(outputs_v1) != len(outputs_v2):
            print("❌ Different number of outputs")
            return False
            
        shapes_match = all(s1 == s2 for s1, s2 in zip(shapes_v1, shapes_v2))
        if not shapes_match:
            print("❌ Output shapes don't match")
            return False
            
        # Test knowledge distillation compatibility
        print(f"\n=== Knowledge Distillation Compatibility ===")
        
        # Expected format: (bbox_regressions, classifications, landmarks)
        bbox_v1, cls_v1, ldm_v1 = outputs_v1
        bbox_v2, cls_v2, ldm_v2 = outputs_v2
        
        print(f"Bbox regression - V1: {bbox_v1.shape}, V2: {bbox_v2.shape}")
        print(f"Classification - V1: {cls_v1.shape}, V2: {cls_v2.shape}")
        print(f"Landmarks - V1: {ldm_v1.shape}, V2: {ldm_v2.shape}")
        
        # Check if we can compute distillation loss
        try:
            # Test KL divergence computation (common in knowledge distillation)
            temperature = 4.0
            
            # Softmax with temperature
            soft_targets_v1 = torch.nn.functional.softmax(cls_v1 / temperature, dim=-1)
            soft_predictions_v2 = torch.nn.functional.log_softmax(cls_v2 / temperature, dim=-1)
            
            # KL divergence
            kl_loss = torch.nn.functional.kl_div(soft_predictions_v2, soft_targets_v1, reduction='batchmean')
            
            print(f"✓ KL divergence computable: {kl_loss.item():.6f}")
            
            # Test MSE for features (bbox and landmarks)
            bbox_mse = torch.nn.functional.mse_loss(bbox_v2, bbox_v1)
            ldm_mse = torch.nn.functional.mse_loss(ldm_v2, ldm_v1)
            
            print(f"✓ Bbox MSE computable: {bbox_mse.item():.6f}")
            print(f"✓ Landmark MSE computable: {ldm_mse.item():.6f}")
            
            print("\n✅ Knowledge distillation compatibility VERIFIED")
            return True
            
        except Exception as e:
            print(f"❌ Knowledge distillation test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run output alignment test"""
    print("FeatherFace V2 Output Alignment Test")
    print("=" * 50)
    
    success = test_output_alignment()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if success:
        print("🎉 SUCCESS: V1 and V2 outputs are properly aligned!")
        print("✅ Knowledge distillation ready")
        print("✅ Training can proceed with confidence")
        return 0
    else:
        print("❌ FAILURE: Output alignment issues detected")
        print("⚠️  Knowledge distillation may not work correctly")
        return 1

if __name__ == "__main__":
    sys.exit(main())