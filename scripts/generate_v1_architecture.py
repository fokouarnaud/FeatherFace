#!/usr/bin/env python3
"""
FeatherFace V1 Architecture Diagram Generator
Creates a clear architectural diagram following reference model with explicit parallel flows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_v1_architecture_diagram():
    """Create FeatherFace V1 architecture diagram with explicit parallel flows"""
    
    # Create figure with optimized landscape layout
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Clean color scheme matching reference
    colors = {
        'input': '#E3F2FD',
        'backbone': '#81D4FA', 
        'features': '#64B5F6',
        'attention': '#66BB6A',
        'bifpn': '#FFB74D',
        'detection': '#F06292',
        'output': '#CE93D8'
    }
    
    # Title - minimal
    ax.text(9, 9.5, 'FeatherFace V1 Architecture', fontsize=20, fontweight='bold', 
            ha='center', va='center', color='#2E2E2E')
    
    # Main architecture components positioning
    # Input
    input_box = FancyBboxPatch((0.5, 4), 2, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 5, 'Input Image\n640×640×3', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Backbone (MobileNet-0.25)
    backbone_box = FancyBboxPatch((3.5, 4), 2.5, 2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(4.75, 5.5, 'MobileNet-0.25', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    ax.text(4.75, 5, 'Backbone', fontsize=10,
            ha='center', va='center', color='#2E2E2E')
    ax.text(4.75, 4.5, '213K params', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#1976D2')
    
    # Multi-scale features - P3, P4, P5 (explicit parallel outputs)
    feature_positions = [(7, 6), (7, 5), (7, 4)]
    feature_labels = ['P3/32', 'P4/16', 'P5/8']
    
    for i, ((x, y), label) in enumerate(zip(feature_positions, feature_labels)):
        feat_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['features'], 
                                 edgecolor='black', linewidth=1)
        ax.add_patch(feat_box)
        ax.text(x+0.75, y, label, fontsize=10, fontweight='bold',
                ha='center', va='center', color='#2E2E2E')
    
    # Attention mechanism (CBAM) - for each feature level
    attention_positions = [(9.5, 6), (9.5, 5), (9.5, 4)]
    
    for i, (x, y) in enumerate(attention_positions):
        att_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['attention'], 
                                edgecolor='black', linewidth=1)
        ax.add_patch(att_box)
        ax.text(x+0.75, y, 'Attention', fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
    
    # BiFPN - bidirectional feature pyramid
    bifpn_box = FancyBboxPatch((12, 3.5), 2, 3, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['bifpn'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(bifpn_box)
    ax.text(13, 5.5, 'BiFPN', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(13, 5, 'Bidirectional\nFeature Fusion', fontsize=9,
            ha='center', va='center', color='white')
    ax.text(13, 4.2, '114K params', fontsize=9, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Detection heads - one per feature level
    detection_positions = [(15, 6), (15, 5), (15, 4)]
    detection_labels = ['P3 Head', 'P4 Head', 'P5 Head']
    
    for i, ((x, y), label) in enumerate(zip(detection_positions, detection_labels)):
        det_box = FancyBboxPatch((x, y-0.3), 1.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['detection'], 
                                edgecolor='black', linewidth=1)
        ax.add_patch(det_box)
        ax.text(x+0.75, y, label, fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Output
    output_box = FancyBboxPatch((16.5, 4), 1.2, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(17.1, 5, 'Output', fontsize=10, fontweight='bold',
            ha='center', va='center', color='#2E2E2E')
    
    # Explicit arrows for each parallel flow
    arrow_props = dict(arrowstyle='->', lw=2, color='#424242')
    
    # Input to backbone
    ax.annotate('', xy=(3.5, 5), xytext=(2.5, 5), arrowprops=arrow_props)
    
    # Backbone to features (3 separate arrows)
    ax.annotate('', xy=(7, 5.7), xytext=(6, 5.5), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(7, 5), xytext=(6, 5), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(7, 4.3), xytext=(6, 4.5), arrowprops=arrow_props)  # P5
    
    # Features to attention (3 separate arrows)
    ax.annotate('', xy=(9.5, 5.7), xytext=(8.5, 5.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(9.5, 5), xytext=(8.5, 5), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(9.5, 4.3), xytext=(8.5, 4.3), arrowprops=arrow_props)  # P5
    
    # Attention to BiFPN (3 separate arrows)
    ax.annotate('', xy=(12, 5.5), xytext=(11, 5.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(12, 5), xytext=(11, 5), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(12, 4.5), xytext=(11, 4.3), arrowprops=arrow_props)  # P5
    
    # BiFPN bidirectional connections (top-down and bottom-up)
    ax.annotate('', xy=(12.3, 4.8), xytext=(12.3, 5.2), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#FF5722'))
    ax.annotate('', xy=(13.7, 4.8), xytext=(13.7, 5.2), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#FF5722'))
    
    # BiFPN to detection heads (3 separate arrows)
    ax.annotate('', xy=(15, 5.7), xytext=(14, 5.5), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(15, 5), xytext=(14, 5), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(15, 4.3), xytext=(14, 4.5), arrowprops=arrow_props)  # P5
    
    # Detection heads to output (convergent arrows)
    ax.annotate('', xy=(16.5, 5.2), xytext=(16.5, 5.7), arrowprops=arrow_props)  # P3
    ax.annotate('', xy=(16.5, 5), xytext=(16.5, 5), arrowprops=arrow_props)      # P4
    ax.annotate('', xy=(16.5, 4.8), xytext=(16.5, 4.3), arrowprops=arrow_props)  # P5
    
    # Additional architecture details section
    detail_y = 2.5
    
    # CBAM detail
    cbam_detail = FancyBboxPatch((1, detail_y-0.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['attention'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(cbam_detail)
    ax.text(2.5, detail_y, 'CBAM: Channel + Spatial Attention\n44K params (9.2%)', 
            fontsize=9, ha='center', va='center', color='white')
    
    # DCN detail
    dcn_detail = FancyBboxPatch((5, detail_y-0.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFCDD2', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(dcn_detail)
    ax.text(6.5, detail_y, 'DCN Context Module\n148K params (30.4%)', 
            fontsize=9, ha='center', va='center', color='#2E2E2E')
    
    # Channel Shuffle detail
    shuffle_detail = FancyBboxPatch((9, detail_y-0.5), 3, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#E1F5FE', 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(shuffle_detail)
    ax.text(10.5, detail_y, 'Channel Shuffle\n0 params (Zero-cost)', 
            fontsize=9, ha='center', va='center', color='#2E2E2E')
    
    # Detection heads detail
    heads_detail = FancyBboxPatch((13, detail_y-0.5), 3.5, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['detection'], 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(heads_detail)
    ax.text(14.75, detail_y, 'Detection Heads: Cls + BBox + Landmarks\n7K params (1.5%)', 
            fontsize=9, ha='center', va='center', color='white')
    
    # Performance summary at bottom
    ax.text(9, 0.8, '487K Parameters • 87.0% mAP on WIDERFace • 30+ FPS on Mobile', 
            fontsize=14, fontweight='bold', ha='center', va='center', color='#2E2E2E',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', edgecolor='#4CAF50'))
    
    plt.tight_layout()
    plt.savefig('docs/featherface_v1_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_v1_architecture_diagram()
    print("✅ FeatherFace V1 architecture diagram generated: docs/featherface_v1_architecture.png")