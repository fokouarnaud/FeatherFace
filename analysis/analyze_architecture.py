import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.retinaface import RetinaFace
from data.config import cfg_mnet
import numpy as np
from thop import profile, clever_format
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import json
import datetime

def analyze_module_params(module, prefix=""):
    """Recursively analyze parameters in a module"""
    params_dict = OrderedDict()
    
    # Count direct parameters
    direct_params = sum(p.numel() for p in module.parameters(recurse=False))
    if direct_params > 0:
        params_dict[prefix] = direct_params
    
    # Recursively analyze named children
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_params = analyze_module_params(child, child_prefix)
        params_dict.update(child_params)
    
    return params_dict

def create_parameter_breakdown(model):
    """Create detailed parameter breakdown by module"""
    breakdown = OrderedDict()
    
    # Analyze main components
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        breakdown[name] = {
            'total_params': module_params,
            'percentage': 0,  # Will calculate later
            'sub_modules': analyze_module_params(module, name)
        }
    
    # Calculate percentages
    total_params = sum(p.numel() for p in model.parameters())
    for module_name in breakdown:
        breakdown[module_name]['percentage'] = (breakdown[module_name]['total_params'] / total_params) * 100
    
    return breakdown, total_params

def analyze_cbam_usage(model):
    """Analyze CBAM module usage in the model"""
    cbam_modules = []
    
    # Find all CBAM instances
    for name, module in model.named_modules():
        if 'cbam' in name.lower() or module.__class__.__name__ == 'CBAM':
            params = sum(p.numel() for p in module.parameters())
            cbam_modules.append({
                'name': name,
                'params': params,
                'type': module.__class__.__name__
            })
    
    return cbam_modules

def analyze_heads(model):
    """Analyze detection heads"""
    heads_info = {
        'ClassHead': [],
        'BboxHead': [],
        'LandmarkHead': []
    }
    
    for head_type in heads_info.keys():
        if hasattr(model, head_type):
            head_module = getattr(model, head_type)
            for idx, head in enumerate(head_module):
                params = sum(p.numel() for p in head.parameters())
                heads_info[head_type].append({
                    'index': idx,
                    'params': params
                })
    
    return heads_info

def main():
    print("=== FeatherFace Architecture Analysis ===\n")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaFace(cfg=cfg_mnet, phase='test').to(device)
    model.eval()
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f}M)")
    
    # Calculate FLOPs
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    print(f"\nComputational complexity: {macs}")
    print(f"Parameters (from thop): {params}")
    
    # Get detailed breakdown
    breakdown, total = create_parameter_breakdown(model)
    
    # Analyze specific components
    cbam_info = analyze_cbam_usage(model)
    heads_info = analyze_heads(model)
    
    # Calculate component statistics
    backbone_params = breakdown.get('body', {}).get('total_params', 0)
    bifpn_params = breakdown.get('bifpn', {}).get('total_params', 0)
    ssh_params = sum(breakdown.get(f'ssh{i}', {}).get('total_params', 0) for i in range(1, 4))
    ssh_cs_params = sum(breakdown.get(f'ssh{i}_cs', {}).get('total_params', 0) for i in range(1, 4))
    cbam_total_params = sum(cb['params'] for cb in cbam_info)
    heads_total = sum(sum(h['params'] for h in heads) for heads in heads_info.values())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Pie chart of major components
    plt.subplot(2, 2, 1)
    components = {
        'Backbone': backbone_params,
        'BiFPN': bifpn_params,
        'SSH modules': ssh_params + ssh_cs_params,
        'CBAM modules': cbam_total_params,
        'Detection Heads': heads_total,
        'Others': total_params - (backbone_params + bifpn_params + ssh_params + ssh_cs_params + cbam_total_params + heads_total)
    }
    
    components = {k: v for k, v in components.items() if v > 0}
    
    plt.pie(components.values(), 
            labels=components.keys(),
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Parameter Distribution by Component Type')
    
    # Bar chart of top modules
    plt.subplot(2, 2, 2)
    top_modules = sorted([(k, v['total_params']) for k, v in breakdown.items()], 
                        key=lambda x: x[1], reverse=True)[:10]
    
    module_names = [m[0] for m in top_modules]
    module_params = [m[1] for m in top_modules]
    
    plt.barh(module_names, module_params)
    plt.xlabel('Number of Parameters')
    plt.title('Top 10 Modules by Parameter Count')
    plt.tight_layout()
    
    # CBAM usage analysis
    plt.subplot(2, 2, 3)
    if cbam_info:
        cbam_names = [cb['name'] for cb in cbam_info]
        cbam_params_list = [cb['params'] for cb in cbam_info]
        
        plt.bar(range(len(cbam_names)), cbam_params_list)
        plt.xticks(range(len(cbam_names)), cbam_names, rotation=45, ha='right')
        plt.ylabel('Parameters')
        plt.title(f'CBAM Modules Analysis (Total: {len(cbam_info)} instances)')
    
    # Heads analysis
    plt.subplot(2, 2, 4)
    head_types = list(heads_info.keys())
    head_params = [sum(h['params'] for h in heads) for heads in heads_info.values()]
    
    plt.bar(head_types, head_params)
    plt.ylabel('Total Parameters')
    plt.title('Detection Heads Parameter Distribution')
    
    plt.tight_layout()
    plt.savefig('analysis/architecture_params_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved parameter distribution visualization to analysis/architecture_params_distribution.png")
    
    # Generate report
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# FeatherFace Baseline Architecture Analysis

## Executive Summary

- **Total Parameters**: {total_params:,} ({total_params/1e6:.3f}M)
- **Trainable Parameters**: {trainable_params:,} ({trainable_params/1e6:.3f}M)
- **Computational Complexity (FLOPs)**: {macs}
- **Input Size**: 640x640x3
- **Target Reduction**: 50% (to ~0.25M parameters)

## Component Breakdown

### Major Components Distribution

| Component | Parameters | Percentage | Description |
|-----------|------------|------------|-------------|
| Backbone (MobileNetV1 0.25x) | {backbone_params:,} | {(backbone_params/total_params*100):.1f}% | Feature extraction |
| BiFPN | {bifpn_params:,} | {(bifpn_params/total_params*100):.1f}% | Feature pyramid network |
| SSH Modules | {ssh_params + ssh_cs_params:,} | {((ssh_params + ssh_cs_params)/total_params*100):.1f}% | Single stage headless |
| CBAM Modules | {cbam_total_params:,} | {(cbam_total_params/total_params*100):.1f}% | Attention mechanisms |
| Detection Heads | {heads_total:,} | {(heads_total/total_params*100):.1f}% | Class, bbox, landmark |

### Detailed Module Analysis

#### 1. Backbone Analysis
- **Architecture**: MobileNetV1 with 0.25x width multiplier
- **Parameters**: {backbone_params:,}
- **Stages**: 3 stages with progressive channel expansion (8→16→32→64→128→256)
- **Already optimized** with depthwise separable convolutions

#### 2. BiFPN Analysis (Biggest Consumer)
- **Parameters**: {bifpn_params:,} ({(bifpn_params/total_params*100):.1f}% of total)
- **Configuration**: 
  - Output channels: 64
  - Repetitions: 3
  - Conv channel coefficients: [64, 128, 256]
- **Optimization potential**: HIGH - Can reduce channels and repetitions

#### 3. CBAM Module Analysis
- **Total instances**: {len(cbam_info)}
- **Total parameters**: {cbam_total_params:,}
- **Distribution**:
  - 3 instances after backbone stages
  - 3 instances after BiFPN outputs
- **Current reduction ratio**: 16
- **Optimization potential**: MEDIUM - Can increase reduction ratio and share weights

#### 4. SSH Module Analysis
- **Total parameters**: {ssh_params + ssh_cs_params:,}
- **Configuration**: 3 SSH modules + 3 Channel Shuffle modules
- **Each SSH module**: ~{ssh_params//3 if ssh_params > 0 else 0:,} parameters
- **Optimization potential**: MEDIUM - Can increase channel grouping

#### 5. Detection Heads Analysis
- **Total parameters**: {heads_total:,}
- **Breakdown**:
  - ClassHead (3x): {sum(h['params'] for h in heads_info['ClassHead']):,} params
  - BboxHead (3x): {sum(h['params'] for h in heads_info['BboxHead']):,} params
  - LandmarkHead (3x): {sum(h['params'] for h in heads_info['LandmarkHead']):,} params
- **Optimization potential**: HIGH - Can unify into shared multi-head

## Top Parameter Consumers

### Top 10 Modules by Parameter Count
"""
    
    # Add top modules to report
    for i, (module_name, params) in enumerate(top_modules[:10], 1):
        percentage = (params / total_params) * 100
        report += f"\n{i}. **{module_name}**: {params:,} params ({percentage:.1f}%)"
    
    report += f"""

## Optimization Recommendations

### Priority 1: BiFPN Optimization (Target: 40-45% reduction)
- **Current**: 64 channels, 3 repetitions
- **Proposed**: 32 channels, 2 repetitions
- **Expected savings**: ~{int(bifpn_params * 0.45):,} parameters

### Priority 2: Unified Detection Heads (Target: 60% reduction)
- **Current**: 3 separate heads × 3 scales = 9 head modules
- **Proposed**: Single SharedMultiHead with shared trunk
- **Expected savings**: ~{int(heads_total * 0.6):,} parameters

### Priority 3: CBAM Optimization (Target: 50% reduction)
- **Current**: 6 instances with reduction=16
- **Proposed**: 3 shared instances with reduction=32
- **Expected savings**: ~{int(cbam_total_params * 0.5):,} parameters

### Priority 4: SSH Optimization (Target: 20% reduction)
- **Current**: Standard convolutions
- **Proposed**: More grouped convolutions and channel reduction
- **Expected savings**: ~{int((ssh_params + ssh_cs_params) * 0.2):,} parameters

## Expected V2 Architecture

### Projected Parameter Distribution
- **Total parameters**: ~250,000 (0.25M)
- **Reduction**: {((total_params - 250000) / total_params * 100):.1f}%

### Component Targets
| Component | Current | Target | Reduction |
|-----------|---------|--------|-----------|
| Backbone | {backbone_params:,} | {backbone_params:,} | 0% (unchanged) |
| BiFPN | {bifpn_params:,} | ~{int(bifpn_params * 0.55):,} | 45% |
| SSH | {ssh_params + ssh_cs_params:,} | ~{int((ssh_params + ssh_cs_params) * 0.8):,} | 20% |
| CBAM | {cbam_total_params:,} | ~{int(cbam_total_params * 0.5):,} | 50% |
| Heads | {heads_total:,} | ~{int(heads_total * 0.4):,} | 60% |

## Implementation Strategy

1. **Phase 1**: Create optimized modules (CBAM++, SharedMultiHead)
2. **Phase 2**: Implement lightweight BiFPN and SSH
3. **Phase 3**: Integrate into RetinaFaceV2 architecture
4. **Phase 4**: Knowledge distillation training
5. **Phase 5**: Evaluation and fine-tuning

## Visualization

![Parameter Distribution](architecture_params_distribution.png)

## Conclusion

The analysis confirms that FeatherFace baseline has **{total_params/1e6:.3f}M parameters**, with BiFPN being the largest consumer at {(bifpn_params/total_params*100):.1f}% of total parameters. The proposed optimizations targeting BiFPN, detection heads, and attention modules should achieve the goal of **0.25M parameters** while maintaining or improving performance through knowledge distillation and enhanced regularization.

---
*Generated on: {current_time}*
"""
    
    # Save report
    with open('analysis/baseline_architecture_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed breakdown as JSON
    breakdown_json = {
        'total_params': total_params,
        'components': components,
        'detailed_breakdown': {k: v['total_params'] for k, v in breakdown.items()},
        'optimization_targets': {
            'bifpn': {'current': components.get('BiFPN', 0), 'target_reduction': 0.45},
            'heads': {'current': components.get('Detection Heads', 0), 'target_reduction': 0.60},
            'cbam': {'current': components.get('CBAM modules', 0), 'target_reduction': 0.50},
            'ssh': {'current': components.get('SSH modules', 0), 'target_reduction': 0.20}
        }
    }
    
    with open('analysis/architecture_breakdown.json', 'w') as f:
        json.dump(breakdown_json, f, indent=2)
    
    print(f"\nAnalysis complete! Report saved to analysis/baseline_architecture_analysis.md")
    print(f"Detailed breakdown saved to analysis/architecture_breakdown.json")

if __name__ == "__main__":
    main()
