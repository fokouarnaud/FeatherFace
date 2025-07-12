#!/usr/bin/env python3
"""
Fine-tune out_channel to achieve exact 488,700 parameters
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_paper_exact import FeatherFacePaperExact
from data.config import cfg_paper_accurate

def find_exact_parameter_match():
    """Find the exact out_channel value to achieve 488,700 parameters"""
    
    target = 488700
    print(f"Finding exact out_channel for {target:,} parameters")
    print("=" * 50)
    
    # Test range around current best
    best_diff = float('inf')
    best_config = None
    
    # Test intermediate values between 52 and 56
    for out_ch in [52, 53, 54, 55, 56]:
        try:
            test_cfg = cfg_paper_accurate.copy()
            test_cfg['out_channel'] = out_ch
            
            model = FeatherFacePaperExact(cfg=test_cfg, phase='test')
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            diff = total_params - target
            print(f"out_channel={out_ch:2d}: {total_params:,} params (diff: {diff:+,})")
            
            if abs(diff) < abs(best_diff):
                best_diff = diff
                best_config = (out_ch, total_params)
                
        except Exception as e:
            print(f"out_channel={out_ch:2d}: Error - {e}")
    
    print()
    if best_config:
        print(f"🎯 BEST MATCH: out_channel={best_config[0]}")
        print(f"   Parameters: {best_config[1]:,}")
        print(f"   Target: {target:,}")
        print(f"   Difference: {best_config[1] - target:+,}")
        
        if abs(best_config[1] - target) <= 500:
            print("   ✅ EXCELLENT MATCH!")
        elif abs(best_config[1] - target) <= 2000:
            print("   ✅ GOOD MATCH!")
        else:
            print("   ⚠️  Need further tuning")
            
        return best_config[0]
    
    return None

def test_optimal_configuration(optimal_out_channel):
    """Test the optimal configuration thoroughly"""
    
    print(f"\nTesting optimal configuration: out_channel={optimal_out_channel}")
    print("=" * 50)
    
    test_cfg = cfg_paper_accurate.copy()
    test_cfg['out_channel'] = optimal_out_channel
    
    try:
        model = FeatherFacePaperExact(cfg=test_cfg, phase='test')
        
        # Detailed parameter breakdown
        param_info = model.get_parameter_count()
        
        print("Detailed parameter breakdown:")
        for component, count in param_info.items():
            if component not in ['paper_target', 'difference']:
                print(f"  {component:<20}: {count:>8,}")
        
        print("-" * 35)
        print(f"  {'Total parameters':<20}: {param_info['total']:>8,}")
        print(f"  {'Paper target':<20}: {param_info['paper_target']:>8,}")
        print(f"  {'Difference':<20}: {param_info['difference']:>+8,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print(f"✅ Forward pass successful")
        print(f"   BBox: {outputs[0].shape}")
        print(f"   Class: {outputs[1].shape}")
        print(f"   Landmarks: {outputs[2].shape}")
        
        # Validation
        validation, _ = model.validate_paper_exact()
        print("\nValidation results:")
        for check, passed in validation.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def main():
    print("FeatherFace Paper-Exact Parameter Fine-Tuning")
    print("=" * 50)
    
    # Find optimal out_channel
    optimal_out_channel = find_exact_parameter_match()
    
    if optimal_out_channel:
        # Test optimal configuration
        success = test_optimal_configuration(optimal_out_channel)
        
        if success:
            print(f"\n🎉 SUCCESS!")
            print(f"   Optimal out_channel: {optimal_out_channel}")
            print(f"   Update cfg_paper_accurate['out_channel'] = {optimal_out_channel}")
        else:
            print(f"\n❌ Configuration needs further adjustment")
    else:
        print("\n❌ Could not find optimal configuration")

if __name__ == "__main__":
    main()