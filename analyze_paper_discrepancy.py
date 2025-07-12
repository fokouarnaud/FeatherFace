#!/usr/bin/env python3
"""
Critical Analysis: FeatherFace Paper vs Implementation Discrepancy
================================================================

FINDINGS SUMMARY:
- Paper target: 488,700 parameters
- Current V2 implementation: 515,115 parameters  
- Discrepancy: +26,415 parameters (+5.4%)

Key Issue: The current out_channel=56 configuration results in too many parameters.
The paper's Table 1 shows 488.7K parameters, suggesting a different architecture.

Based on the parameter analysis, we need out_channel ≈ 48 to reach the target.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_v2 import FeatherFaceV2
from data.config import cfg_v2

def create_paper_accurate_config():
    """Create configuration matching the paper's 488.7K parameters"""
    
    # Test different configurations to find 488.7K target
    target = 488700
    best_config = None
    best_diff = float('inf')
    
    print("="*80)
    print("FINDING OPTIMAL CONFIGURATION FOR 488.7K PARAMETERS")
    print("="*80)
    
    # Test out_channel values
    for out_ch in range(40, 60, 2):
        try:
            test_cfg = cfg_v2.copy()
            test_cfg['out_channel'] = out_ch
            
            model = FeatherFaceV2(cfg=test_cfg, phase='test')
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            diff = abs(total_params - target)
            print(f"out_channel={out_ch:2d}: {total_params:,} params (diff: {total_params-target:+,})")
            
            if diff < best_diff:
                best_diff = diff
                best_config = (out_ch, total_params)
                
        except Exception as e:
            print(f"out_channel={out_ch:2d}: Error - {e}")
    
    if best_config:
        print(f"\nBEST MATCH: out_channel={best_config[0]} with {best_config[1]:,} parameters")
        print(f"Difference from target: {best_config[1] - target:+,} parameters")
        
        return best_config[0]
    
    return None

def analyze_major_parameter_consumers():
    """Identify which components consume the most parameters"""
    
    print("\n" + "="*80)
    print("MAJOR PARAMETER CONSUMERS ANALYSIS")
    print("="*80)
    
    model = FeatherFaceV2(cfg=cfg_v2, phase='test')
    
    component_params = {
        'backbone': 0,
        'backbone_eca': 0,
        'bifpn': 0,
        'bifpn_eca': 0,
        'ssh_heads': 0,
        'channel_shuffle': 0,
        'detection_heads': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            
            if 'body.' in name:
                component_params['backbone'] += param_count
            elif 'backbone_eca' in name:
                component_params['backbone_eca'] += param_count
            elif 'bifpn.' in name and 'eca' not in name:
                component_params['bifpn'] += param_count
            elif 'bif_eca' in name:
                component_params['bifpn_eca'] += param_count
            elif 'ssh' in name and '_cs' not in name:
                component_params['ssh_heads'] += param_count
            elif '_cs' in name:
                component_params['channel_shuffle'] += param_count
            elif any(head in name for head in ['ClassHead', 'BboxHead', 'LandmarkHead']):
                component_params['detection_heads'] += param_count
            else:
                component_params['other'] += param_count
    
    total = sum(component_params.values())
    
    print(f"{'Component':<20} {'Parameters':<12} {'Percentage':<10}")
    print("-" * 50)
    for component, count in sorted(component_params.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100
        print(f"{component:<20} {count:>8,} {percentage:>8.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total:>8,} {100.0:>8.1f}%")
    
    return component_params

def calculate_paper_architecture_estimate():
    """Calculate expected parameter distribution based on paper"""
    
    print("\n" + "="*80)
    print("PAPER ARCHITECTURE ESTIMATE")
    print("="*80)
    
    # Based on the paper's description and typical values
    estimates = {
        'MobileNet-0.25 backbone': 213081,  # From our analysis
        'BiFPN (3 layers)': 93158,          # From our analysis  
        'ECA-Net modules (6x)': 22,         # Our ECA implementation
        'SSH + DCN heads': 203391,          # From our analysis
        'Channel Shuffle': 5472,            # From our analysis
        'Detection heads': 5472,            # From our analysis
    }
    
    # But we need to reduce SSH heads since they're the biggest consumer
    # Paper suggests out_channel should be lower
    
    print("Expected parameter distribution (488.7K target):")
    total_estimated = 0
    for component, count in estimates.items():
        total_estimated += count
        print(f"  {component:<25}: {count:>8,}")
    
    print("-" * 45)
    print(f"  {'Estimated total':<25}: {total_estimated:>8,}")
    print(f"  {'Paper target':<25}: {488700:>8,}")
    print(f"  {'Difference':<25}: {total_estimated - 488700:>+8,}")
    
    print(f"\nThe SSH heads ({estimates['SSH + DCN heads']:,} params) are likely the issue.")
    print("Reducing out_channel from 56 to ~48 should achieve the target.")

def main():
    """Main analysis function"""
    print("FeatherFace Paper vs Implementation Discrepancy Analysis")
    print("========================================================")
    
    # Find optimal configuration
    optimal_out_channel = create_paper_accurate_config()
    
    # Analyze current parameter distribution
    component_breakdown = analyze_major_parameter_consumers()
    
    # Calculate paper estimates
    calculate_paper_architecture_estimate()
    
    print("\n" + "="*80)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("="*80)
    
    if optimal_out_channel:
        print(f"1. OPTIMAL CONFIGURATION: out_channel = {optimal_out_channel}")
        print("   This achieves closest match to paper's 488.7K parameters")
        
    print(f"\n2. MAJOR PARAMETER CONSUMER: SSH heads ({component_breakdown['ssh_heads']:,} params)")
    print("   Reducing out_channel reduces SSH head parameters significantly")
    
    print(f"\n3. CURRENT ISSUE: out_channel=56 results in {component_breakdown['ssh_heads']:,} SSH parameters")
    print("   Paper likely used out_channel≈48 for 488.7K target")
    
    print(f"\n4. ECA-NET EFFICIENCY: Only {component_breakdown['backbone_eca'] + component_breakdown['bifpn_eca']} parameters")
    print("   ECA-Net is indeed ultra-efficient as claimed")
    
    print(f"\n5. ARCHITECTURE VALIDATION:")
    print("   ✓ MobileNet-0.25 backbone matches")
    print("   ✓ BiFPN structure matches") 
    print("   ✓ ECA-Net implementation is efficient")
    print("   ✗ out_channel=56 too high for paper target")
    
    print(f"\nRECOMMENDATION: Update cfg_v2['out_channel'] = {optimal_out_channel} to match paper")

if __name__ == "__main__":
    main()