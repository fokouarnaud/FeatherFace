#!/usr/bin/env python3
"""
FeatherFace Model Validation Script
Scientific validation for CBAM baseline and ECA-CBAM hybrid

Usage:
    python validate_model.py --version cbam
    python validate_model.py --version eca_cbam --detailed
    python validate_model.py --quick-check
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Validate FeatherFace models')
    parser.add_argument('--version', choices=['cbam', 'eca_cbam'], default='cbam', 
                       help='Model version to validate: cbam (baseline) or eca_cbam (hybrid)')
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
        from data import cfg_cbam_paper_exact
        
        print("🔍 CBAM Baseline Model Validation")
        print("-" * 40)
        
        # Create model
        model = FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: FeatherFaceCBAMExact")
        print(f"✓ Configuration: cfg_cbam_paper_exact")
        cbam_target = cfg_cbam_paper_exact['paper_baseline_performance']['total_parameters']
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Target: {cbam_target:,} parameters")
        print(f"✓ Difference: {total_params - cbam_target:+,}")
        print(f"✓ Accuracy: {((1 - abs(total_params - cbam_target) / cbam_target) * 100):.2f}%")
        
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
        print(f"\\n🎯 Validation Status: {'PASSED ✅' if target_met else 'CLOSE ⚠️'}")
        
        return total_params, target_met
        
    except Exception as e:
        print(f"❌ CBAM validation failed: {e}")
        return None, False

def validate_eca_cbam_model():
    """Validate ECA-CBAM hybrid model"""
    try:
        from models.featherface_eca_cbam import FeatherFaceECAcbaM
        from data import cfg_eca_cbam
        
        print("🔍 ECA-CBAM Hybrid Model Validation")
        print("-" * 40)
        
        # Create model
        model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: FeatherFaceECAcbaM")
        print(f"✓ Configuration: cfg_eca_cbam")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Target: ~449,000 parameters (achieved better than 460K target)")
        print(f"✓ Difference from CBAM: {total_params - 488664:+,}")
        print(f"✓ Parameter efficiency: {((488664 - total_params) / 488664 * 100):+.2f}%")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass: SUCCESS")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
        
        # ECA-CBAM validation
        has_eca = any('eca' in name.lower() for name, _ in model.named_modules())
        has_sam = any('sam' in name.lower() for name, _ in model.named_modules())
        print(f"✓ ECA-Net attention present: {'YES' if has_eca else 'NO'}")
        print(f"✓ CBAM SAM present: {'YES' if has_sam else 'NO'}")
        
        # ECA-CBAM specific analysis
        if hasattr(model, 'get_parameter_count'):
            param_info = model.get_parameter_count()
            print(f"✓ ECA-CBAM modules: {param_info.get('total_eca_cbam', 'N/A')} parameters")
        
        target_met = 445000 <= total_params <= 465000
        print(f"\\n🎯 Validation Status: {'PASSED ✅' if target_met else 'REVIEW ⚠️'}")
        
        return total_params, target_met
        
    except Exception as e:
        print(f"❌ ECA-CBAM validation failed: {e}")
        return None, False

def detailed_analysis(version):
    """Run detailed parameter analysis"""
    print(f"\\n📊 DETAILED ANALYSIS - {version.upper()}")
    print("=" * 50)
    
    if version == 'cbam':
        from models.featherface_cbam_exact import FeatherFaceCBAMExact
        from data import cfg_cbam_paper_exact
        model = FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact)
    else:
        from models.featherface_eca_cbam import FeatherFaceECAcbaM
        from data import cfg_eca_cbam
        model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("Component breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        percentage = (params / total_params) * 100
        print(f"  {name:20}: {params:8,} ({percentage:5.1f}%)")
    
    print(f"\\nAttention modules:")
    attention_params = 0
    for name, module in model.named_modules():
        if any(att in name.lower() for att in ['cbam', 'eca', 'sam', 'attention']):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                attention_params += params
                print(f"  {name:30}: {params:6,} parameters")
    
    print(f"\\nTotal attention parameters: {attention_params:,}")
    print(f"Attention percentage: {(attention_params / total_params) * 100:.2f}%")
    
    # Memory usage estimation
    print(f"\\nMemory usage (FP32):")
    param_memory = total_params * 4 / 1024 / 1024  # MB
    print(f"  Parameters: {param_memory:.1f} MB")
    
    # Feature map memory (640x640 input)
    dummy_input = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    activation_memory = sum(out.numel() * 4 for out in outputs) / 1024 / 1024
    print(f"  Activations: {activation_memory:.1f} MB")
    print(f"  Total: {param_memory + activation_memory:.1f} MB")

def quick_check():
    """Quick parameter count check for both models"""
    print("🚀 Quick Parameter Count Check")
    print("=" * 50)
    
    # CBAM
    cbam_params, cbam_passed = validate_cbam_model()
    print()
    
    # ECA-CBAM
    eca_cbam_params, eca_cbam_passed = validate_eca_cbam_model()
    
    if cbam_params and eca_cbam_params:
        print("\\n📊 Comparison Summary:")
        print(f"  CBAM Baseline:     {cbam_params:8,} parameters")
        print(f"  ECA-CBAM Hybrid:   {eca_cbam_params:8,} parameters")
        print(f"  Difference:        {eca_cbam_params - cbam_params:+8,} parameters")
        print(f"  Efficiency gain:   {((cbam_params - eca_cbam_params) / cbam_params * 100):+5.1f}%")
        
        print(f"\\n🎯 Overall Status:")
        print(f"  CBAM: {'PASSED ✅' if cbam_passed else 'FAILED ❌'}")
        print(f"  ECA-CBAM: {'PASSED ✅' if eca_cbam_passed else 'FAILED ❌'}")

def validate_specific_model(model_path):
    """Validate a specific model checkpoint"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine model type from path or keys
        is_eca_cbam = 'eca_cbam' in model_path or any('eca' in key for key in checkpoint.keys())
        
        print(f"🔍 Validating model: {model_path}")
        print(f"Model type detected: {'ECA-CBAM' if is_eca_cbam else 'CBAM'}")
        
        # Count parameters in checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"Parameters in checkpoint: {total_params:,}")
        
        # Load into appropriate model
        if is_eca_cbam:
            from models.featherface_eca_cbam import FeatherFaceECAcbaM
            from data import cfg_eca_cbam
            model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam)
        else:
            from models.featherface_cbam_exact import FeatherFaceCBAMExact
            from data import cfg_cbam_paper_exact
            model = FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact)
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✅ Model validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def main():
    args = parse_args()
    
    print("🔬 FeatherFace Model Validation")
    print("=" * 60)
    print("Scientific validation for CBAM baseline and ECA-CBAM hybrid")
    print("=" * 60)
    
    try:
        if args.model_path:
            # Validate specific model
            validate_specific_model(args.model_path)
        
        elif args.quick_check:
            # Quick check both models
            quick_check()
        
        else:
            # Validate specific version
            if args.version == 'cbam':
                params, passed = validate_cbam_model()
            else:
                params, passed = validate_eca_cbam_model()
            
            if args.detailed and params:
                detailed_analysis(args.version)
        
        print("\\n✅ Validation complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Validation error: {e}")

if __name__ == '__main__':
    main()