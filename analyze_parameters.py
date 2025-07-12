#!/usr/bin/env python3
"""
Parameter Count Analysis for FeatherFace Architecture
====================================================

This script analyzes the parameter count discrepancy between the official 
FeatherFace paper (488.7K parameters) and current implementation (~515K).

Official Paper: "FeatherFace: Robust and Lightweight Face Detection via Optimal Feature Integration"
Electronics 2025, DOI: 10.3390/electronics14030517

Target: 488.7K parameters with MobileNet-0.25 + BiFPN + CBAM + DCN + ChannelShuffle
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retinaface import RetinaFace
from models.featherface_v2 import FeatherFaceV2
from data.config import cfg_mnet, cfg_v2

def count_parameters(model, detailed=True):
    """Count parameters in a model with detailed breakdown"""
    total_params = 0
    param_breakdown = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            # Categorize parameters by component
            if 'backbone' in name:
                category = 'backbone'
            elif 'fpn' in name or 'bifpn' in name:
                category = 'fpn/bifpn'
            elif 'cbam' in name or 'attention' in name:
                category = 'attention'
            elif 'ssh' in name:
                category = 'ssh_heads'
            elif 'ClassHead' in name or 'BboxHead' in name or 'LandmarkHead' in name:
                category = 'detection_heads'
            elif 'channel_shuffle' in name:
                category = 'channel_shuffle'
            else:
                category = 'other'
            
            if category not in param_breakdown:
                param_breakdown[category] = 0
            param_breakdown[category] += param_count
            
            if detailed:
                print(f"{name:60} {param_count:>10,} params {list(param.shape)}")
    
    return total_params, param_breakdown

def analyze_architecture_components():
    """Analyze individual components to understand parameter distribution"""
    print("="*80)
    print("FEATHERFACE PARAMETER ANALYSIS")
    print("="*80)
    print(f"Target from paper: 488,700 parameters")
    print()
    
    # Test V1 (RetinaFace baseline)
    print("1. V1 BASELINE (RetinaFace with MobileNet-0.25)")
    print("-" * 50)
    try:
        model_v1 = RetinaFace(cfg=cfg_mnet, phase='test')
        total_v1, breakdown_v1 = count_parameters(model_v1, detailed=False)
        
        print(f"Total V1 parameters: {total_v1:,}")
        print("Component breakdown:")
        for component, count in breakdown_v1.items():
            print(f"  {component:20}: {count:>8,} ({count/total_v1*100:5.1f}%)")
        print()
        
    except Exception as e:
        print(f"Error loading V1: {e}")
        print()
    
    # Test V2 (Current implementation)
    print("2. V2 CURRENT IMPLEMENTATION")
    print("-" * 50)
    try:
        model_v2 = FeatherFaceV2(cfg=cfg_v2, phase='test')
        total_v2, breakdown_v2 = count_parameters(model_v2, detailed=False)
        
        print(f"Total V2 parameters: {total_v2:,}")
        print("Component breakdown:")
        for component, count in breakdown_v2.items():
            print(f"  {component:20}: {count:>8,} ({count/total_v2*100:5.1f}%)")
        
        discrepancy = total_v2 - 488700
        print()
        print(f"Discrepancy vs paper: {discrepancy:+,} parameters ({discrepancy/488700*100:+5.1f}%)")
        print()
        
    except Exception as e:
        print(f"Error loading V2: {e}")
        print()

def test_different_configurations():
    """Test different configurations to find 488.7K target"""
    print("3. CONFIGURATION TESTING TO REACH 488.7K TARGET")
    print("-" * 50)
    
    target = 488700
    
    # Test different out_channel values
    test_configs = [32, 48, 56, 64]
    
    for out_ch in test_configs:
        print(f"\nTesting out_channel = {out_ch}")
        try:
            # Create modified config
            test_cfg = cfg_v2.copy()
            test_cfg['out_channel'] = out_ch
            
            model = FeatherFaceV2(cfg=test_cfg, phase='test')
            total_params, _ = count_parameters(model, detailed=False)
            
            diff = total_params - target
            print(f"  Parameters: {total_params:,} (diff: {diff:+,})")
            
            if abs(diff) < 1000:
                print(f"  *** CLOSE TO TARGET! ***")
                
        except Exception as e:
            print(f"  Error: {e}")

def analyze_paper_architecture():
    """Analyze what the paper architecture should be"""
    print("4. PAPER ARCHITECTURE ANALYSIS")
    print("-" * 50)
    
    print("Based on Electronics 2025 paper Table 1:")
    print("- MobileNet-0.25 backbone")
    print("- BiFPN feature aggregation") 
    print("- CBAM attention (backbone + BiFPN)")
    print("- DCN + shuffle detection heads")
    print("- Final config: 488.7K params")
    print()
    
    print("Expected parameter distribution (estimated):")
    print("  MobileNet-0.25 backbone: ~350K")
    print("  BiFPN (3 layers):        ~80K") 
    print("  CBAM modules:            ~25K")
    print("  SSH + DCN heads:         ~30K")
    print("  Channel shuffle:         ~3K")
    print("  Total estimated:         ~488K")
    print()

def analyze_model_structure():
    """Analyze detailed model structure to identify parameter sources"""
    print("5. DETAILED MODEL STRUCTURE ANALYSIS")
    print("-" * 50)
    
    try:
        model = FeatherFaceV2(cfg=cfg_v2, phase='test')
        
        print("Full parameter breakdown:")
        total_params, breakdown = count_parameters(model, detailed=True)
        
        print()
        print("SUMMARY:")
        print(f"Total parameters: {total_params:,}")
        print("Component breakdown:")
        for component, count in sorted(breakdown.items()):
            print(f"  {component:20}: {count:>8,} ({count/total_params*100:5.1f}%)")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main analysis function"""
    print("FeatherFace Parameter Count Investigation")
    print("Paper target: 488,700 parameters")
    print("Current implementation: ~515,000 parameters")
    print("Discrepancy: ~26,300 parameters")
    print()
    
    analyze_architecture_components()
    test_different_configurations() 
    analyze_paper_architecture()
    analyze_model_structure()
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()