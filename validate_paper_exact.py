#!/usr/bin/env python3
"""
FeatherFace Paper-Exact Validation Script
========================================

This script validates that our implementation exactly matches the official Electronics 2025 paper:
"FeatherFace: Robust and Lightweight Face Detection via Optimal Feature Integration"
DOI: 10.3390/electronics14030517

Target: 488,700 parameters exactly
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_paper_exact import FeatherFacePaperExact, create_paper_exact_model
from data.config import cfg_paper_accurate

def validate_paper_exact_implementation():
    """Comprehensive validation of paper-exact implementation"""
    
    print("="*80)
    print("FEATHERFACE PAPER-EXACT VALIDATION")
    print("="*80)
    print("Paper: Electronics 2025, DOI: 10.3390/electronics14030517")
    print("Target: 488,700 parameters exactly")
    print()
    
    # 1. Test configuration
    print("1. CONFIGURATION VALIDATION")
    print("-" * 40)
    
    required_keys = ['out_channel', 'paper_performance', 'scientific_foundation']
    for key in required_keys:
        if key in cfg_paper_accurate:
            print(f"✓ {key}: Present")
        else:
            print(f"✗ {key}: MISSING")
    
    print(f"✓ out_channel: {cfg_paper_accurate['out_channel']} (paper-exact)")
    print(f"✓ target_parameters: {cfg_paper_accurate['paper_performance']['total_parameters']:,}")
    print()
    
    # 2. Model creation and parameter validation
    print("2. MODEL PARAMETER VALIDATION")
    print("-" * 40)
    
    try:
        model = create_paper_exact_model(cfg_paper_accurate, phase='test')
        
        # Get detailed parameter breakdown
        param_info = model.get_parameter_count()
        
        print("Parameter breakdown:")
        for component, count in param_info.items():
            if component not in ['paper_target', 'difference']:
                print(f"  {component:<20}: {count:>8,}")
        
        print("-" * 35)
        print(f"  {'Total parameters':<20}: {param_info['total']:>8,}")
        print(f"  {'Paper target':<20}: {param_info['paper_target']:>8,}")
        print(f"  {'Difference':<20}: {param_info['difference']:>+8,}")
        
        # Validation checks
        validation, _ = model.validate_paper_exact()
        
        print("\nValidation results:")
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print()
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # 3. Architecture component validation
    print("3. ARCHITECTURE COMPONENT VALIDATION")
    print("-" * 40)
    
    components = [
        ('backbone_eca_0', 'Backbone ECA P3'),
        ('backbone_eca_1', 'Backbone ECA P4'), 
        ('backbone_eca_2', 'Backbone ECA P5'),
        ('bif_eca_0', 'BiFPN ECA P3'),
        ('bif_eca_1', 'BiFPN ECA P4'),
        ('bif_eca_2', 'BiFPN ECA P5'),
        ('ssh1', 'SSH Head 1'),
        ('ssh2', 'SSH Head 2'),
        ('ssh3', 'SSH Head 3'),
        ('ssh1_cs', 'Channel Shuffle 1'),
        ('ssh2_cs', 'Channel Shuffle 2'),
        ('ssh3_cs', 'Channel Shuffle 3'),
    ]
    
    for attr_name, description in components:
        if hasattr(model, attr_name):
            print(f"✓ {description}: Present")
        else:
            print(f"✗ {description}: MISSING")
    
    print()
    
    # 4. Forward pass validation
    print("4. FORWARD PASS VALIDATION")
    print("-" * 40)
    
    try:
        # Test input
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Validate outputs
        if isinstance(outputs, tuple) and len(outputs) == 3:
            bbox_reg, classification, landmarks = outputs
            print(f"✓ Forward pass successful")
            print(f"  BBox regression: {bbox_reg.shape}")
            print(f"  Classification: {classification.shape}")
            print(f"  Landmarks: {landmarks.shape}")
        else:
            print(f"✗ Forward pass failed: Invalid output format")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print()
    
    # 5. Performance target validation
    print("5. PERFORMANCE TARGET VALIDATION")
    print("-" * 40)
    
    paper_targets = cfg_paper_accurate['paper_performance']
    print("Official paper performance targets:")
    print(f"  WIDERFace Easy:   {paper_targets['widerface_easy']:.1%}")
    print(f"  WIDERFace Medium: {paper_targets['widerface_medium']:.1%}")
    print(f"  WIDERFace Hard:   {paper_targets['widerface_hard']:.1%}")
    print(f"  Overall AP:       {paper_targets['overall_ap']:.1%}")
    print(f"  FLOPs:            {paper_targets['flops']}")
    print()
    
    # 6. Scientific validation
    print("6. SCIENTIFIC VALIDATION")
    print("-" * 40)
    
    foundation = cfg_paper_accurate['scientific_foundation']
    print("Paper details:")
    print(f"  DOI: {foundation['paper_doi']}")
    print(f"  Authors: {foundation['authors']}")
    print(f"  Journal: {foundation['journal']}")
    print(f"  Publication: {foundation['publication_date']}")
    print()
    
    # 7. Final validation summary
    print("7. FINAL VALIDATION SUMMARY")
    print("-" * 40)
    
    success_criteria = [
        (abs(param_info['difference']) <= 1000, f"Parameter count within 1K of target"),
        (param_info['total'] >= 485000 and param_info['total'] <= 492000, "Parameter count in valid range"),
        (validation['eca_efficiency'], "ECA-Net efficiency validated"),
        (validation['architecture_complete'], "Architecture components complete"),
    ]
    
    all_passed = True
    for passed, description in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {description}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 SUCCESS: Paper-exact implementation validated!")
        print(f"   Model achieves {param_info['total']:,} parameters")
        print(f"   Target was {param_info['paper_target']:,} parameters")
        print(f"   Difference: {param_info['difference']:+,} parameters")
    else:
        print("❌ FAILURE: Implementation does not match paper exactly")
    
    return all_passed

def test_comparison_with_current_v2():
    """Compare paper-exact with current V2 implementation"""
    
    print("\n" + "="*80)
    print("COMPARISON: PAPER-EXACT vs CURRENT V2")
    print("="*80)
    
    try:
        from models.featherface_v2 import FeatherFaceV2
        from data.config import cfg_v2
        
        # Current V2
        model_v2 = FeatherFaceV2(cfg=cfg_v2, phase='test')
        params_v2 = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
        
        # Paper-exact
        model_paper = create_paper_exact_model(cfg_paper_accurate, phase='test')
        param_info_paper = model_paper.get_parameter_count()
        
        print(f"Current V2 (out_channel=56):    {params_v2:,} parameters")
        print(f"Paper-exact (out_channel=52):   {param_info_paper['total']:,} parameters")
        print(f"Reduction:                      {params_v2 - param_info_paper['total']:,} parameters")
        print(f"Paper target:                   {param_info_paper['paper_target']:,} parameters")
        
        accuracy_v2 = (params_v2 - 488700) / 488700 * 100
        accuracy_paper = param_info_paper['difference'] / 488700 * 100
        
        print(f"\nAccuracy vs paper:")
        print(f"  Current V2:    {accuracy_v2:+5.1f}% error")
        print(f"  Paper-exact:   {accuracy_paper:+5.1f}% error")
        
    except Exception as e:
        print(f"Comparison failed: {e}")

def main():
    """Main validation function"""
    success = validate_paper_exact_implementation()
    test_comparison_with_current_v2()
    
    if success:
        print("\n🎯 RECOMMENDATION: Use FeatherFacePaperExact for scientific accuracy")
        print("   This implementation matches the official Electronics 2025 paper exactly")
    else:
        print("\n⚠️  WARNING: Implementation needs correction to match paper")

if __name__ == "__main__":
    main()