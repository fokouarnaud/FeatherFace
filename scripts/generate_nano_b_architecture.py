#!/usr/bin/env python3
"""
FeatherFace Nano-B Architecture Diagram Generator

Generates a professional publication-ready architecture diagram for FeatherFace Nano-B,
showcasing the Bayesian-Optimized Soft FPGM Pruning and Knowledge Distillation integration.

Features:
- Complete Nano-B architecture visualization
- B-FPGM pruning structure
- Knowledge distillation flow
- Parameter annotations
- Professional publication quality
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Ellipse
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_publication_style():
    """Setup matplotlib for publication-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'text.usetex': False,  # Set to True if LaTeX is available
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': True
    })

def create_featherface_nano_b_diagram():
    """Create comprehensive FeatherFace Nano-B architecture diagram"""
    
    # Create figure with golden ratio dimensions
    fig_width = 16
    fig_height = fig_width / 1.618  # Golden ratio
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Colors for different components (colorblind-friendly palette)
    colors = {
        'input': '#2E86AB',      # Blue
        'backbone': '#A23B72',   # Purple  
        'efficient': '#F18F01',  # Orange
        'detection': '#C73E1D',  # Red
        'pruning': '#4CAF50',    # Green
        'distillation': '#FF9800', # Amber
        'output': '#607D8B'      # Blue Grey
    }
    
    # Component dimensions and positions
    comp_height = 0.8
    comp_spacing = 1.5
    y_center = 4
    
    # Input layer
    input_box = FancyBboxPatch(
        (0.5, y_center - comp_height/2), 2, comp_height,
        boxstyle="round,pad=0.1", 
        facecolor=colors['input'], 
        edgecolor='black', 
        linewidth=1.5
    )
    ax.add_patch(input_box)
    ax.text(1.5, y_center, 'Input\n(3×H×W)', ha='center', va='center', 
            fontweight='bold', color='white', fontsize=10)
    
    # MobileNet Backbone
    backbone_x = 3.5
    backbone_box = FancyBboxPatch(
        (backbone_x, y_center - comp_height/2), 2.5, comp_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['backbone'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(backbone_box)
    ax.text(backbone_x + 1.25, y_center + 0.2, 'MobileNetV1 0.25×', 
            ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax.text(backbone_x + 1.25, y_center - 0.2, 'Backbone\n(~85K params)', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Efficient modules stack
    efficient_x = 7
    efficient_components = [
        ('Efficient CBAM\n(Reduction=8)', '~8K params'),
        ('Efficient BiFPN\n(72 channels)', '~45K params'),
        ('Grouped SSH\n(Groups=2)', '~25K params')
    ]
    
    for i, (name, params) in enumerate(efficient_components):
        y_pos = y_center + 1.5 - i * 1.2
        eff_box = FancyBboxPatch(
            (efficient_x, y_pos - comp_height/2), 2.8, comp_height*0.8,
            boxstyle="round,pad=0.08",
            facecolor=colors['efficient'],
            edgecolor='black',
            linewidth=1.2
        )
        ax.add_patch(eff_box)
        ax.text(efficient_x + 1.4, y_pos + 0.15, name, 
                ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax.text(efficient_x + 1.4, y_pos - 0.15, params, 
                ha='center', va='center', fontsize=8, color='white')
    
    # Detection heads
    detection_x = 11
    detection_box = FancyBboxPatch(
        (detection_x, y_center - comp_height/2), 2.5, comp_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['detection'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(detection_box)
    ax.text(detection_x + 1.25, y_center + 0.2, 'Detection Heads', 
            ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax.text(detection_x + 1.25, y_center - 0.2, 'BBox + Cls + Landmarks\n(~25K params)', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Output
    output_x = 14.5
    output_box = FancyBboxPatch(
        (output_x, y_center - comp_height/2), 2, comp_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(output_box)
    ax.text(output_x + 1, y_center, 'Output\nDetections', ha='center', va='center', 
            fontweight='bold', color='white', fontsize=10)
    
    # B-FPGM Pruning annotations
    pruning_y = 6.5
    
    # Pruning title
    ax.text(8.5, pruning_y + 0.5, 'B-FPGM Bayesian Pruning', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            color=colors['pruning'])
    
    # Pruning connections and rates
    pruning_targets = [
        (backbone_x + 1.25, y_center + comp_height/2 + 0.1, '15-25%'),
        (efficient_x + 1.4, y_center + 1.5 + comp_height/2 + 0.1, '20-30%'),
        (efficient_x + 1.4, y_center + comp_height/2 + 0.1, '25-35%'),
        (efficient_x + 1.4, y_center - 1.5 + comp_height/2 + 0.1, '10-20%'),
        (detection_x + 1.25, y_center + comp_height/2 + 0.1, '5-15%')
    ]
    
    for x, y, rate in pruning_targets:
        # Pruning rate annotation
        pruning_circle = plt.Circle((x, pruning_y), 0.3, 
                                  facecolor=colors['pruning'], 
                                  edgecolor='black', 
                                  linewidth=1)
        ax.add_patch(pruning_circle)
        ax.text(x, pruning_y, rate, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
        
        # Connection line
        connection = ConnectionPatch((x, y), (x, pruning_y - 0.3), 
                                   "data", "data",
                                   arrowstyle="->", 
                                   shrinkA=0, shrinkB=0,
                                   color=colors['pruning'], 
                                   linewidth=2)
        ax.add_artist(connection)
    
    # Knowledge Distillation flow
    kd_y = 1.5
    
    # Teacher model (V1) representation
    teacher_box = FancyBboxPatch(
        (2, kd_y - 0.4), 6, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors['distillation'],
        edgecolor='black',
        linewidth=1,
        alpha=0.7
    )
    ax.add_patch(teacher_box)
    ax.text(5, kd_y, 'Teacher Model (V1): 487K parameters → Knowledge Transfer', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Distillation arrows
    for x_pos in [backbone_x + 1.25, efficient_x + 1.4, detection_x + 1.25]:
        kd_arrow = ConnectionPatch((x_pos, kd_y + 0.4), (x_pos, y_center - comp_height/2 - 0.1),
                                 "data", "data",
                                 arrowstyle="->",
                                 shrinkA=0, shrinkB=0,
                                 color=colors['distillation'],
                                 linewidth=2.5,
                                 alpha=0.8)
        ax.add_artist(kd_arrow)
    
    # Main architecture flow arrows
    flow_positions = [
        (2.5, y_center),      # Input to backbone
        (6, y_center),        # Backbone to efficient
        (9.8, y_center),      # Efficient to detection
        (13.5, y_center)      # Detection to output
    ]
    
    for x_pos, y_pos in flow_positions:
        flow_arrow = patches.FancyArrowPatch(
            (x_pos - 0.2, y_pos), (x_pos + 0.2, y_pos),
            arrowstyle='->', 
            mutation_scale=20,
            color='black',
            linewidth=2
        )
        ax.add_patch(flow_arrow)
    
    # Parameter summary box
    summary_box = FancyBboxPatch(
        (11.5, 0.2), 5, 1,
        boxstyle="round,pad=0.1",
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(summary_box)
    
    summary_text = """FeatherFace Nano-B Summary:
• Total Parameters: 120-180K (Target: 48-65% reduction)
• B-FPGM: Bayesian-optimized pruning rates
• Knowledge Distillation: Weighted teacher guidance
• Architecture: Ultra-lightweight + competitive mAP"""
    
    ax.text(14, 0.7, summary_text, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Scientific foundation annotations
    foundation_y = 7.5
    ax.text(8.5, foundation_y, 
            'Scientific Foundation: Kaparinos & Mezaris (WACVW 2025) + Li et al. (CVPR 2023)', 
            ha='center', va='center', fontsize=10, fontweight='bold', 
            style='italic', color='darkblue')
    
    # Title and layout
    ax.set_title('FeatherFace Nano-B: Bayesian-Optimized Ultra-Lightweight Face Detection Architecture', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def save_architecture_diagram():
    """Generate and save the architecture diagram"""
    print("🎨 Generating FeatherFace Nano-B Architecture Diagram...")
    
    # Setup publication style
    setup_publication_style()
    
    # Create diagram
    fig = create_featherface_nano_b_diagram()
    
    # Save paths
    docs_dir = project_root / 'docs'
    docs_dir.mkdir(exist_ok=True)
    
    output_paths = [
        docs_dir / 'featherface_nano_b_architecture.png',
        docs_dir / 'featherface_nano_b_architecture.pdf',
        docs_dir / 'featherface_nano_b_architecture.svg'
    ]
    
    # Save in multiple formats
    for output_path in output_paths:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                       transparent=True, facecolor='white')
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save {output_path}: {e}")
    
    # Show file sizes
    for output_path in output_paths:
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"📊 {output_path.name}: {size_mb:.2f} MB")
    
    plt.close(fig)
    return output_paths

def create_architecture_documentation():
    """Create detailed documentation for the architecture"""
    docs_dir = project_root / 'docs'
    doc_path = docs_dir / 'NANO_B_ARCHITECTURE.md'
    
    documentation = """# FeatherFace Nano-B Architecture

## Overview

FeatherFace Nano-B is an ultra-lightweight face detection model that combines Bayesian-Optimized Soft FPGM Pruning with Weighted Knowledge Distillation to achieve 120-180K parameters (48-65% reduction from baseline) while maintaining competitive accuracy.

## Architecture Components

### 1. MobileNetV1 0.25× Backbone (~85K parameters)
- Lightweight feature extractor optimized for mobile deployment
- 0.25× width multiplier for ultra-efficiency
- Depthwise separable convolutions for parameter reduction

### 2. Efficient Modules Stack
- **Efficient CBAM** (~8K parameters): Channel and spatial attention with reduction ratio 8
- **Efficient BiFPN** (~45K parameters): Bi-directional feature pyramid with 72 channels
- **Grouped SSH** (~25K parameters): Grouped single-shot hierarchical detection with 2 groups

### 3. Detection Heads (~25K parameters)
- Bounding box regression
- Classification (face/background)
- Facial landmark detection (5 points)

## Scientific Innovations

### B-FPGM Bayesian Pruning
Based on Kaparinos & Mezaris (WACVW 2025):
- **Adaptive pruning rates**: 15-25% (backbone), 20-30% (CBAM), 25-35% (BiFPN), 10-20% (SSH), 5-15% (heads)
- **Bayesian optimization**: Automated rate selection for optimal accuracy-efficiency trade-off
- **Soft pruning**: Gradual weight reduction during training

### Weighted Knowledge Distillation
Based on Li et al. (CVPR 2023) and 2025 edge computing research:
- **Teacher model**: FeatherFace V1 (487K parameters)
- **Temperature**: 4.0 for optimal knowledge transfer
- **Alpha**: 0.7 (70% distillation, 30% task loss)
- **Adaptive weights**: Learnable coefficients for different output types

## Training Pipeline

### Phase 1: Weighted Knowledge Distillation (Epochs 1-100)
- Transfer knowledge from V1 teacher to Nano-B student
- Establish base capabilities before pruning

### Phase 2: Bayesian Pruning Optimization (Epochs 101-200)
- Apply B-FPGM with Bayesian-optimized rates
- Target 50% parameter reduction with minimal accuracy loss

### Phase 3: Fine-tuning (Epochs 201-300)
- Stabilize pruned weights
- Recover accuracy after structural changes

## Performance Targets

- **Parameters**: 120-180K (48-65% reduction from V1)
- **Model size**: <1 MB (ultra-lightweight)
- **WIDERFace mAP**: >78% overall (competitive with larger models)
- **Inference speed**: <50ms on mobile devices

## Deployment

The model supports:
- **Dynamic ONNX export**: Flexible input sizes (320×320 to 832×832)
- **Mobile optimization**: TorchScript for iOS/Android
- **Cross-platform**: Web deployment via ONNX.js
- **Edge devices**: Optimized for low-power inference

## Files Generated

- `featherface_nano_b_architecture.png`: High-resolution diagram
- `featherface_nano_b_architecture.pdf`: Vector format for publications
- `featherface_nano_b_architecture.svg`: Scalable web format

---

*Generated by FeatherFace Nano-B Architecture Generator*
"""
    
    with open(doc_path, 'w') as f:
        f.write(documentation)
    
    print(f"📝 Documentation created: {doc_path}")
    return doc_path

def main():
    """Main function to generate complete architecture package"""
    print("🎯 FeatherFace Nano-B Architecture Generator")
    print("="*50)
    
    try:
        # Generate architecture diagram
        diagram_paths = save_architecture_diagram()
        
        # Create documentation
        doc_path = create_architecture_documentation()
        
        print(f"\n✅ Architecture package completed!")
        print(f"📂 Files generated:")
        for path in diagram_paths + [doc_path]:
            if path.exists():
                print(f"  ✅ {path}")
        
        print(f"\n🎨 The architecture diagram showcases:")
        print(f"  • Complete Nano-B architecture flow")
        print(f"  • B-FPGM pruning structure and rates")
        print(f"  • Knowledge distillation from teacher model")
        print(f"  • Parameter distribution and targets")
        print(f"  • Scientific foundation and innovations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating architecture: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()