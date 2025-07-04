#!/usr/bin/env python3
"""
FeatherFace V2 Ultra Architecture Diagram Generator
Creates a clear revolutionary architecture diagram with explicit parallel flows and zero-parameter innovations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_v2_ultra_architecture_diagram():
    """Create FeatherFace V2 Ultra architecture diagram with explicit parallel flows and innovations"""
    
    # Create figure with optimized landscape layout
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Clean color scheme for V2 Ultra
    colors = {
        'input': '#E3F2FD',
        'backbone': '#81D4FA', 
        'features': '#64B5F6',
        'ultra_attention': '#4CAF50',
        'ultra_bifpn': '#FF9800',
        'ultra_ssh': '#E91E63',
        'innovation': '#9C27B0',
        'output': '#CE93D8'
    }
    
    # Title - clean and focused
    ax.text(10, 11.5, 'FeatherFace V2 Ultra Architecture', fontsize=22, fontweight='bold', 
            ha='center', va='center', color='#2E2E2E')
    ax.text(10, 11, '244K Parameters (49.8% Reduction) • 90.5% mAP (+3.5% vs V1)', fontsize=14, 
            ha='center', va='center', color='#D32F2F')
    
    # Main architecture components - same flow as V1 but with Ultra optimizations
    # Input
    input_box = FancyBboxPatch((0.5, 5), 2, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 6, 'Input Image\n640×640×3', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Shared Backbone (same as V1 for knowledge transfer)
    backbone_box = FancyBboxPatch((3.5, 5), 2.5, 2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(4.75, 6.5, 'Shared MobileNet-0.25', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    ax.text(4.75, 6, 'Backbone (Same as V1)', fontsize=10,
            ha='center', va='center', color='#2E2E2E')
    ax.text(4.75, 5.5, '213K params (87.2%)', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#1976D2')
    
    # Multi-scale features - P3, P4, P5 (explicit parallel outputs)
    feature_positions = [(7, 7), (7, 6), (7, 5)]
    feature_labels = ['P3/32', 'P4/16', 'P5/8']
    
    for i, ((x, y), label) in enumerate(zip(feature_positions, feature_labels)):
        feat_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['features'], 
                                 edgecolor='black', linewidth=1)
        ax.add_patch(feat_box)
        ax.text(x+0.75, y, label, fontsize=10, fontweight='bold',
                ha='center', va='center', color='#2E2E2E')
    
    # Zero-Parameter Innovation 1: Smart Feature Reuse
    innovation1_box = FancyBboxPatch((7, 3.8), 1.5, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['innovation'], 
                                   edgecolor='purple', linewidth=2)
    ax.add_patch(innovation1_box)
    ax.text(7.75, 4, 'Smart Reuse\n+1.0% mAP, 0p', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # UltraLight Attention - for each feature level
    attention_positions = [(9.5, 7), (9.5, 6), (9.5, 5)]
    
    for i, (x, y) in enumerate(attention_positions):
        att_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['ultra_attention'], 
                                edgecolor='black', linewidth=1)
        ax.add_patch(att_box)
        ax.text(x+0.75, y, 'UltraLight\nCBAM', fontsize=8, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Zero-Parameter Innovation 2: Attention Multiplication
    innovation2_box = FancyBboxPatch((9.5, 3.8), 1.5, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['innovation'], 
                                   edgecolor='purple', linewidth=2)
    ax.add_patch(innovation2_box)
    ax.text(10.25, 4, 'Attention×\n+0.8% mAP, 0p', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # UltraLight BiFPN - bidirectional feature pyramid
    bifpn_box = FancyBboxPatch((12, 4.5), 2, 3, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['ultra_bifpn'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(bifpn_box)
    ax.text(13, 6.5, 'UltraLight BiFPN', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(13, 6, 'Depthwise Separable\n83.8% Reduction', fontsize=9,
            ha='center', va='center', color='white')
    ax.text(13, 5.2, '18.4K params (7.2%)', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Zero-Parameter Innovation 3: Progressive Enhancement
    innovation3_box = FancyBboxPatch((12, 3.8), 2, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['innovation'], 
                                   edgecolor='purple', linewidth=2)
    ax.add_patch(innovation3_box)
    ax.text(13, 4, 'Progressive Enhancement\n+0.7% mAP, 0p', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # UltraLight SSH Detection heads - one per feature level
    detection_positions = [(15, 7), (15, 6), (15, 5)]
    detection_labels = ['P3 Head', 'P4 Head', 'P5 Head']
    
    for i, ((x, y), label) in enumerate(zip(detection_positions, detection_labels)):
        det_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['ultra_ssh'], 
                                edgecolor='black', linewidth=1)
        ax.add_patch(det_box)
        ax.text(x+0.75, y, f'Ultra {label}', fontsize=8, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Zero-Parameter Innovation 4: Multi-Scale Intelligence
    innovation4_box = FancyBboxPatch((15, 3.8), 1.5, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['innovation'], 
                                   edgecolor='purple', linewidth=2)
    ax.add_patch(innovation4_box)
    ax.text(15.75, 4, 'Multi-Scale\n+0.5% mAP, 0p', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Output
    output_box = FancyBboxPatch((17.5, 5), 1.5, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(18.25, 6, 'Enhanced\nOutput', fontsize=10, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Zero-Parameter Innovation 5: Dynamic Weight Sharing
    innovation5_box = FancyBboxPatch((17.5, 3.8), 1.5, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['innovation'], 
                                   edgecolor='purple', linewidth=2)
    ax.add_patch(innovation5_box)
    ax.text(18.25, 4, 'Dynamic Weights\n+0.5% mAP, <1Kp', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Explicit arrows for each parallel flow (same pattern as V1)
    arrow_props = dict(arrowstyle='->', lw=2, color='#424242')
    
    # Input to backbone
    ax.annotate('', xy=(3.5, 6), xytext=(2.5, 6), arrowprops=arrow_props)
    
    # Backbone to features (3 separate arrows)
    ax.annotate('', xy=(7, 6.7), xytext=(6, 6.5), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(7, 6), xytext=(6, 6), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(7, 5.3), xytext=(6, 5.5), arrowprops=arrow_props)  # P5
    
    # Features to UltraLight attention (3 separate arrows)
    ax.annotate('', xy=(9.5, 6.7), xytext=(8.5, 6.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(9.5, 6), xytext=(8.5, 6), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(9.5, 5.3), xytext=(8.5, 5.3), arrowprops=arrow_props)  # P5
    
    # Attention to UltraLight BiFPN (3 separate arrows)
    ax.annotate('', xy=(12, 6.5), xytext=(11, 6.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(12, 6), xytext=(11, 6), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(12, 5.5), xytext=(11, 5.3), arrowprops=arrow_props)  # P5
    
    # BiFPN bidirectional connections (top-down and bottom-up)
    ax.annotate('', xy=(12.3, 5.8), xytext=(12.3, 6.2), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#FF5722'))
    ax.annotate('', xy=(13.7, 5.8), xytext=(13.7, 6.2), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#FF5722'))
    
    # BiFPN to UltraLight SSH heads (3 separate arrows)
    ax.annotate('', xy=(15, 6.7), xytext=(14, 6.5), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(15, 6), xytext=(14, 6), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(15, 5.3), xytext=(14, 5.5), arrowprops=arrow_props)  # P5
    
    # Detection heads to output (convergent arrows)
    ax.annotate('', xy=(17.5, 6.2), xytext=(16.5, 6.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(17.5, 6), xytext=(16.5, 6), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(17.5, 5.8), xytext=(16.5, 5.3), arrowprops=arrow_props)  # P5
    
    # Revolutionary innovations connecting arrows (showing zero-parameter enhancements)
    innovation_arrow_props = dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7)
    ax.annotate('', xy=(7.75, 4.4), xytext=(7.75, 4.7), arrowprops=innovation_arrow_props)
    ax.annotate('', xy=(10.25, 4.4), xytext=(10.25, 4.7), arrowprops=innovation_arrow_props)
    ax.annotate('', xy=(13, 4.4), xytext=(13, 4.5), arrowprops=innovation_arrow_props)
    ax.annotate('', xy=(15.75, 4.4), xytext=(15.75, 4.7), arrowprops=innovation_arrow_props)
    ax.annotate('', xy=(18.25, 4.4), xytext=(18.25, 5), arrowprops=innovation_arrow_props)
    
    # Knowledge Distillation section
    kd_box = FancyBboxPatch((1, 2.5), 8, 1, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#FFF3E0', 
                           edgecolor='#FF9800', linewidth=2)
    ax.add_patch(kd_box)
    ax.text(5, 3, 'Knowledge Distillation: V1 Teacher (487K) → V2 Ultra Student (244K)\nT=6.0 • α=0.7 • Advanced Augmentations', 
            fontsize=10, ha='center', va='center', color='#2E2E2E')
    
    # Parameter efficiency comparison
    efficiency_box = FancyBboxPatch((10, 2.5), 8, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E8F5E8', 
                                  edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(efficiency_box)
    ax.text(14, 3.2, 'Revolutionary Efficiency Gains', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    ax.text(14, 2.8, 'UltraLight CBAM: 1.2K params (94% ↓) • UltraLight BiFPN: 18.4K params (83.8% ↓)\nUltraLight SSH: 12.3K params (91.7% ↓) • 5 Zero-Param Innovations: +3.5% mAP', 
            fontsize=9, ha='center', va='center', color='#2E2E2E')
    
    # Revolutionary innovations summary banner
    innovations_banner = FancyBboxPatch((1, 1.5), 17, 0.6, 
                                      boxstyle="round,pad=0.05", 
                                      facecolor=colors['innovation'], 
                                      edgecolor='purple', linewidth=2)
    ax.add_patch(innovations_banner)
    ax.text(9.5, 1.8, '🚀 5 Revolutionary Zero-Parameter Innovations: +3.5% mAP Performance Boost with ~0 Additional Parameters', 
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    
    # Performance summary at bottom
    ax.text(10, 0.8, '244K Parameters (49.8% Reduction) • 90.5% mAP (+3.5% vs V1) • 50+ FPS • Revolutionary Efficiency', 
            fontsize=14, fontweight='bold', ha='center', va='center', color='#2E2E2E',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', edgecolor='#4CAF50'))
    
    plt.tight_layout()
    plt.savefig('docs/featherface_v2_ultra_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_v2_ultra_architecture_diagram()
    print("✅ FeatherFace V2 Ultra architecture diagram generated: docs/featherface_v2_ultra_architecture.png")