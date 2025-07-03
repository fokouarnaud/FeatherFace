#!/usr/bin/env python3
"""
FeatherFace Parameter Validation Module

This module provides validation functions for FeatherFace V1 and V2 models
to ensure they meet the paper specifications.
"""

import torch
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))


def count_model_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def validate_v1_parameters(verbose=True):
    """
    Validate FeatherFace V1 parameters match paper specification.
    
    Target: 489K parameters (paper specification)
    Tolerance: ±5K parameters
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        
        # Create V1 model
        model = RetinaFace(cfg=cfg_mnet, phase='test')
        total_params, trainable_params = count_model_parameters(model)
        
        # Paper target: 489K parameters
        target_params = 489000
        tolerance = 5000
        
        # Check if within tolerance
        diff = abs(total_params - target_params)
        validation_passed = diff <= tolerance
        
        if verbose:
            print(f"FeatherFace V1 Parameter Validation")
            print(f"=" * 40)
            print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
            print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f}M)")
            print(f"Target: {target_params:,} (±{tolerance:,})")
            print(f"Difference: {total_params - target_params:+,}")
            print(f"Validation: {'PASSED ✅' if validation_passed else 'FAILED ❌'}")
            
            if not validation_passed:
                print(f"\nSuggestions:")
                if total_params < target_params:
                    print(f"- Increase out_channel in cfg_mnet (currently {cfg_mnet['out_channel']})")
                    print(f"- Consider increasing BiFPN channels")
                else:
                    print(f"- Decrease out_channel in cfg_mnet (currently {cfg_mnet['out_channel']})")
                    print(f"- Consider reducing BiFPN channels")
        
        return validation_passed
        
    except Exception as e:
        if verbose:
            print(f"Validation failed with error: {e}")
        return False


def validate_v2_parameters(verbose=True):
    """
    Validate FeatherFace V2 parameters match specification.
    
    Target: 256K parameters (50% reduction from V1)
    Tolerance: ±5K parameters
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        from models.retinaface_v2 import RetinaFaceV2
        from data.config import cfg_mnet_v2
        
        # Create V2 model
        model = RetinaFaceV2(cfg=cfg_mnet_v2, phase='test')
        total_params, trainable_params = count_model_parameters(model)
        
        # V2 target: 256K parameters
        target_params = 256000
        tolerance = 5000
        
        # Check if within tolerance
        diff = abs(total_params - target_params)
        validation_passed = diff <= tolerance
        
        if verbose:
            print(f"FeatherFace V2 Parameter Validation")
            print(f"=" * 40)
            print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
            print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f}M)")
            print(f"Target: {target_params:,} (±{tolerance:,})")
            print(f"Difference: {total_params - target_params:+,}")
            print(f"Validation: {'PASSED ✅' if validation_passed else 'FAILED ❌'}")
        
        return validation_passed
        
    except Exception as e:
        if verbose:
            print(f"V2 validation failed with error: {e}")
        return False


def analyze_model_components(model_name='v1', verbose=True):
    """
    Analyze model components and their parameter contributions.
    
    Args:
        model_name: 'v1' or 'v2'
        verbose: Whether to print detailed analysis
    """
    try:
        if model_name.lower() == 'v1':
            from models.retinaface import RetinaFace
            from data.config import cfg_mnet
            model = RetinaFace(cfg=cfg_mnet, phase='test')
        else:
            from models.retinaface_v2 import RetinaFaceV2
            from data.config import cfg_mnet_v2
            model = RetinaFaceV2(cfg=cfg_mnet_v2, phase='test')
        
        total_params = 0
        component_analysis = {}
        
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            component_analysis[name] = params
            total_params += params
            
            if verbose:
                percentage = (params / total_params) * 100 if total_params > 0 else 0
                print(f"{name}: {params:,} parameters ({percentage:.1f}%)")
        
        if verbose:
            print(f"\nTotal: {total_params:,} parameters")
        
        return component_analysis, total_params
        
    except Exception as e:
        if verbose:
            print(f"Component analysis failed: {e}")
        return {}, 0


def validate_architecture_forward_pass(model_name='v1', input_size=(1, 3, 640, 640)):
    """
    Test model forward pass with given input size.
    
    Args:
        model_name: 'v1' or 'v2'
        input_size: Input tensor size (batch, channels, height, width)
    
    Returns:
        bool: True if forward pass succeeds, False otherwise
    """
    try:
        if model_name.lower() == 'v1':
            from models.retinaface import RetinaFace
            from data.config import cfg_mnet
            model = RetinaFace(cfg=cfg_mnet, phase='test')
        else:
            from models.retinaface_v2 import RetinaFaceV2
            from data.config import cfg_mnet_v2
            model = RetinaFaceV2(cfg=cfg_mnet_v2, phase='test')
        
        # Create dummy input
        dummy_input = torch.randn(*input_size)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"Forward pass successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False


def main():
    """Main validation script."""
    print("FeatherFace Model Validation")
    print("=" * 50)
    
    # Validate V1
    print("\n1. FeatherFace V1 Validation:")
    v1_valid = validate_v1_parameters()
    
    if v1_valid:
        print("\n2. V1 Architecture Analysis:")
        analyze_model_components('v1')
        
        print("\n3. V1 Forward Pass Test:")
        validate_architecture_forward_pass('v1')
    
    # Try to validate V2 if available
    print("\n" + "=" * 50)
    print("\n4. FeatherFace V2 Validation:")
    try:
        v2_valid = validate_v2_parameters()
        if v2_valid:
            print("\n5. V2 Architecture Analysis:")
            analyze_model_components('v2')
            
            print("\n6. V2 Forward Pass Test:")
            validate_architecture_forward_pass('v2')
    except ImportError:
        print("FeatherFace V2 not available (models/retinaface_v2.py not found)")
    
    print(f"\nValidation Summary:")
    print(f"V1: {'✅ PASSED' if v1_valid else '❌ FAILED'}")


if __name__ == "__main__":
    main()