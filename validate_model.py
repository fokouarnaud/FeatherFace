#!/usr/bin/env python3
"""
FeatherFace Model Validation Script
Scientific validation for CBAM baseline and ODConv innovation

Usage:
    python validate_model.py --version cbam
    python validate_model.py --version odconv --detailed
    python validate_model.py --quick-check
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Validate FeatherFace models')
    parser.add_argument('--version', choices=['cbam', 'odconv'], default='cbam', 
                       help='Model version to validate: cbam (baseline) or odconv (innovation)')
    parser.add_argument('--detailed', action='store_true', 
                       help='Run detailed parameter analysis')
    parser.add_argument('--quick-check', action='store_true',
                       help='Quick parameter count check for both models')
    parser.add_argument('--model-path', help='Specific model path to validate')
    return parser.parse_args()

def validate_cbam_model():
    """Validate CBAM baseline model"""
    try:
        from models.featherface_cbam_exact import FeatherFaceCBAMExact
        from data.config import cfg_cbam_paper_exact
        
        print("🔍 CBAM Baseline Model Validation")
        print("-" * 40)
        
        # Create model
        model = FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: FeatherFaceCBAMExact")
        print(f"✓ Configuration: cfg_cbam_paper_exact")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Target: 488,664 parameters")
        print(f"✓ Difference: {total_params - 488664:+,}")
        print(f"✓ Accuracy: {((1 - abs(total_params - 488664) / 488664) * 100):.2f}%")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass: SUCCESS")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
        
        # Attention mechanism validation
        has_cbam = any('cbam' in name.lower() for name, _ in model.named_modules())
        print(f"✓ CBAM attention present: {'YES' if has_cbam else 'NO'}")
        
        target_met = abs(total_params - 488664) <= 100
        print(f"\n🎯 Validation Status: {'PASSED ✅' if target_met else 'CLOSE ⚠️'}")
        
        return total_params, target_met
        
    except Exception as e:
        print(f"❌ CBAM validation failed: {e}")
        return None, False

def validate_odconv_model():
    """Validate ODConv innovation model"""
    try:
        from models.featherface_odconv import FeatherFaceODConv
        from data.config import cfg_odconv
        
        print("🔍 ODConv Innovation Model Validation")
        print("-" * 40)
        
        # Create model
        model = FeatherFaceODConv(cfg=cfg_odconv)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: FeatherFaceODConv")
        print(f"✓ Configuration: cfg_odconv")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Target: ~485,000 parameters")
        print(f"✓ Difference from CBAM: {total_params - 488664:+,}")
        print(f"✓ Parameter efficiency: {((488664 - total_params) / 488664 * 100):+.2f}%")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass: SUCCESS")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
        
        # ODConv validation
        has_odconv = any('odconv' in name.lower() for name, _ in model.named_modules())
        print(f"✓ ODConv attention present: {'YES' if has_odconv else 'NO'}")
        
        # ODConv specific analysis
        if hasattr(model, 'get_parameter_count'):
            param_info = model.get_parameter_count()
            print(f"✓ ODConv modules: {param_info.get('total_odconv', 'N/A')} parameters")
        
        target_met = 480000 <= total_params <= 490000
        print(f"\n🎯 Validation Status: {'PASSED ✅' if target_met else 'REVIEW ⚠️'}")
        
        return total_params, target_met
        
    except Exception as e:
        print(f"❌ ODConv validation failed: {e}")
        return None, False

def detailed_analysis(version):
    """Run detailed parameter analysis"""
    print(f"\n📊 DETAILED ANALYSIS - {version.upper()}")
    print("=" * 50)
    
    if version == 'cbam':
        from models.featherface_cbam_exact import FeatherFaceCBAMExact
        from data.config import cfg_cbam_paper_exact
        model = FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact)
    else:
        from models.featherface_odconv import FeatherFaceODConv
        from data.config import cfg_odconv
        model = FeatherFaceODConv(cfg=cfg_odconv)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("Component breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        percentage = (params / total_params) * 100
        print(f"  {name:20}: {params:8,} ({percentage:5.1f}%)")
    
    print(f"\nAttention modules:")
    attention_params = 0
    for name, module in model.named_modules():
        if any(att in name.lower() for att in ['cbam', 'odconv', 'attention']):
            params = sum(p.numel() for p in module.parameters())
            attention_params += params
            if params > 0:
                print(f"  {name:30}: {params:6,} parameters")
    
    print(f"\nTotal attention parameters: {attention_params:,}")
    print(f"Attention percentage: {(attention_params / total_params * 100):.2f}%")

def quick_check():
    """Quick parameter count check for both models"""
    print("🚀 QUICK PARAMETER CHECK")
    print("=" * 50)
    
    cbam_params, cbam_ok = validate_cbam_model()
    print()
    odconv_params, odconv_ok = validate_odconv_model()
    
    print("\n📊 COMPARISON SUMMARY")
    print("-" * 30)
    if cbam_params and odconv_params:
        difference = cbam_params - odconv_params
        efficiency_pct = (difference / cbam_params) * 100
        print(f"CBAM Baseline:   {cbam_params:,} parameters")
        print(f"ODConv Innovation: {odconv_params:,} parameters")
        print(f"Difference:      {difference:+,} parameters ({efficiency_pct:+.2f}%)")
        print(f"Innovation:      4D multidimensional attention")
        print(f"Scientific base: Li et al. ICLR 2022")
    
    overall_status = "PASSED ✅" if (cbam_ok and odconv_ok) else "ISSUES ⚠️"
    print(f"\n🎯 Overall Status: {overall_status}")

def validate_model_file(model_path):
    """Validate a specific model file"""
    print(f"🔍 Validating model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check for thop profiling keys
        profiling_keys = [k for k in checkpoint.keys() if k.endswith(('total_ops', 'total_params'))]
        if profiling_keys:
            print(f"⚠️  Found {len(profiling_keys)} thop profiling keys in model")
            print("   Consider cleaning model with safe loading")
        else:
            print("✓ Model is clean (no thop profiling keys)")
        
        # Count parameters
        total_params = sum(p.numel() for p in checkpoint.values() if torch.is_tensor(p))
        print(f"✓ Model parameters: {total_params:,}")
        
        # Detect model type
        if 'cbam' in model_path.lower():
            expected = 488664
            model_type = "CBAM baseline"
        elif 'odconv' in model_path.lower():
            expected = 485000
            model_type = "ODConv innovation"
        else:
            expected = None
            model_type = "Unknown"
        
        if expected:
            diff = abs(total_params - expected)
            accuracy = (1 - diff / expected) * 100
            print(f"✓ Detected: {model_type}")
            print(f"✓ Expected: ~{expected:,} parameters")
            print(f"✓ Match: {accuracy:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    args = parse_args()
    
    print("🚀 FeatherFace Model Validation")
    print("=" * 50)
    print("Scientific comparison: CBAM baseline vs ODConv innovation")
    print()
    
    if args.model_path:
        return 0 if validate_model_file(args.model_path) else 1
    
    if args.quick_check:
        quick_check()
        return 0
    
    if args.version == 'cbam':
        params, success = validate_cbam_model()
        if args.detailed and success:
            detailed_analysis('cbam')
    elif args.version == 'odconv':
        params, success = validate_odconv_model()
        if args.detailed and success:
            detailed_analysis('odconv')
    
    print("\n💡 Tips:")
    print("• Use --quick-check to validate both models")
    print("• Use --detailed for component breakdown")
    print("• Use --model-path to validate specific files")
    print("• CBAM: 488,664 params (baseline)")
    print("• ODConv: ~485,000 params (4D attention innovation)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())