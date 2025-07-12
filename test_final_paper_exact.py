#!/usr/bin/env python3
"""
Final test of paper-exact configuration
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_paper_exact import FeatherFacePaperExact, create_paper_exact_model
from data.config import cfg_paper_accurate

def test_final_configuration():
    """Test the final paper-exact configuration"""
    
    print("="*80)
    print("FINAL FEATHERFACE PAPER-EXACT VALIDATION")
    print("="*80)
    print(f"Target: 488,700 parameters (Electronics 2025 paper)")
    print(f"Configuration: out_channel = {cfg_paper_accurate['out_channel']}")
    print()
    
    try:
        # Create model
        model = create_paper_exact_model(cfg_paper_accurate, phase='test')
        
        # Get parameter info
        param_info = model.get_parameter_count()
        
        print("PARAMETER ANALYSIS:")
        print("-" * 40)
        for component, count in param_info.items():
            if component not in ['paper_target', 'difference', 'total']:
                print(f"  {component:<20}: {count:>8,}")
        
        print("-" * 32)
        print(f"  {'TOTAL':<20}: {param_info['total']:>8,}")
        print(f"  {'TARGET':<20}: {param_info['paper_target']:>8,}")
        print(f"  {'DIFFERENCE':<20}: {param_info['difference']:>+8,}")
        
        accuracy = abs(param_info['difference']) / param_info['paper_target'] * 100
        print(f"  {'ACCURACY':<20}: {100-accuracy:>7.2f}%")
        
        print()
        
        # Validation
        validation, _ = model.validate_paper_exact()
        
        print("VALIDATION RESULTS:")
        print("-" * 40)
        for check, passed in validation.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {check}")
        
        print()
        
        # Forward pass test
        print("FORWARD PASS TEST:")
        print("-" * 40)
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print(f"✅ Forward pass successful")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  BBox output: {outputs[0].shape}")
        print(f"  Class output: {outputs[1].shape}")
        print(f"  Landmark output: {outputs[2].shape}")
        
        print()
        
        # Final assessment
        print("FINAL ASSESSMENT:")
        print("-" * 40)
        
        all_passed = all(validation.values())
        param_close = abs(param_info['difference']) <= 2000
        
        if all_passed and param_close:
            print("🎉 EXCELLENT: Paper-exact implementation validated!")
            print(f"   ✅ Parameters: {param_info['total']:,} (within 2K of target)")
            print(f"   ✅ Architecture: Complete and validated")
            print(f"   ✅ Forward pass: Functional")
            status = "PRODUCTION_READY"
        elif param_close:
            print("✅ GOOD: Close to paper specifications")
            print(f"   ✅ Parameters: {param_info['total']:,} (within 2K of target)")
            print(f"   ⚠️  Some validation checks failed")
            status = "ACCEPTABLE"
        else:
            print("⚠️  NEEDS IMPROVEMENT: Parameter count too far from target")
            print(f"   ❌ Parameters: {param_info['total']:,} (diff: {param_info['difference']:+,})")
            status = "NEEDS_TUNING"
        
        print()
        print(f"STATUS: {status}")
        
        return status, param_info
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", None

def main():
    """Main test function"""
    status, param_info = test_final_configuration()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if status == "PRODUCTION_READY":
        print("🎯 RECOMMENDATION: Use this configuration for paper-exact reproduction")
        print("   This implementation closely matches the Electronics 2025 paper")
    elif status == "ACCEPTABLE":
        print("✅ RECOMMENDATION: Good enough for research purposes")
        print("   Minor discrepancies but functionally equivalent")
    elif status == "NEEDS_TUNING":
        print("⚠️  RECOMMENDATION: Further parameter tuning needed")
        print("   Consider adjusting architecture components")
    else:
        print("❌ RECOMMENDATION: Fix implementation errors first")
    
    if param_info:
        print(f"\nFinal parameter count: {param_info['total']:,}")
        print(f"Paper target: {param_info['paper_target']:,}")
        print(f"Difference: {param_info['difference']:+,}")

if __name__ == "__main__":
    main()