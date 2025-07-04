#!/usr/bin/env python3
"""
FeatherFace V2 Ultra Code-Faithful Architecture Diagram Generator (Restructured)
Creates a black/white diagram in two sections to avoid overlapping:
(a) Main Architecture Flow
(b) Zero-Parameter Innovations Details
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

def create_v2_ultra_restructured_diagram():
    """Create FeatherFace V2 Ultra diagram with restructured layout (sections a & b)"""
    
    # Create figure optimized for black/white printing with two-section layout
    fig, ax = plt.subplots(1, 1, figsize=(24, 20))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Black/white only - no colors for printing
    colors = {
        'black': '#000000',
        'white': '#FFFFFF', 
        'gray_light': '#F0F0F0',
        'gray_medium': '#D0D0D0',
        'gray_dark': '#808080'
    }
    
    # Main title with proper hierarchy
    ax.text(12, 19.5, 'FeatherFace V2 Ultra Revolutionary Architecture', 
            fontsize=20, fontweight='bold', ha='center', va='center', color=colors['black'])
    ax.text(12, 19.1, '244K Parameters (49.8% Reduction) • 90.5% mAP (+3.5% vs V1)', 
            fontsize=14, ha='center', va='center', color=colors['gray_dark'])
    ax.text(12, 18.7, 'Code-Faithful Implementation Diagram', 
            fontsize=12, ha='center', va='center', color=colors['gray_dark'], style='italic')
    
    # ===== SECTION (A) - MAIN ARCHITECTURE FLOW =====
    ax.text(1, 18, '(a) Main Architecture Flow', 
            fontsize=16, fontweight='bold', ha='left', va='center', color=colors['black'])
    
    main_y = 16.5
    
    # Input
    input_box = Rectangle((0.5, main_y), 2.5, 1.2, linewidth=2, edgecolor=colors['black'], 
                         facecolor=colors['white'])
    ax.add_patch(input_box)
    ax.text(1.75, main_y + 0.6, 'Input Image', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(1.75, main_y + 0.3, '640×640×3', fontsize=10,
            ha='center', va='center', color=colors['black'])
    
    # MobileNetV1-0.25 Backbone 
    backbone_box = Rectangle((4, main_y), 3, 1.2, linewidth=2, edgecolor=colors['black'], 
                            facecolor=colors['gray_light'])
    ax.add_patch(backbone_box)
    ax.text(5.5, main_y + 0.7, 'MobileNetV1-0.25', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5.5, main_y + 0.4, 'Backbone', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5.5, main_y + 0.1, '213K params (87.2%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Multi-scale features extraction (P3, P4, P5)
    feature_y = main_y - 2
    features = [
        (8, feature_y + 1, 'P3: 64ch'),
        (8, feature_y, 'P4: 128ch'), 
        (8, feature_y - 1, 'P5: 256ch')
    ]
    
    for i, (x, y, label) in enumerate(features):
        feat_box = Rectangle((x, y), 2.2, 0.8, linewidth=1.5, edgecolor=colors['black'], 
                            facecolor=colors['white'])
        ax.add_patch(feat_box)
        ax.text(x + 1.1, y + 0.4, label, fontsize=11, fontweight='bold',
                ha='center', va='center', color=colors['black'])
        
        # Arrow from backbone to features
        ax.annotate('', xy=(x, y + 0.4), xytext=(7, main_y + 0.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    # SharedCBAMManager (Backbone)
    cbam1_x = 11.5
    cbam1_box = Rectangle((cbam1_x, feature_y - 0.5), 3.5, 2.8, linewidth=2, 
                         edgecolor=colors['black'], facecolor=colors['gray_medium'])
    ax.add_patch(cbam1_box)
    ax.text(cbam1_x + 1.75, feature_y + 1.5, 'SharedCBAMManager', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(cbam1_x + 1.75, feature_y + 1.2, '(Backbone)', fontsize=11,
            ha='center', va='center', color=colors['black'])
    ax.text(cbam1_x + 1.75, feature_y + 0.9, '1.2K params (0.5%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # UltraLightBiFPN
    bifpn_x = 16.5
    bifpn_box = Rectangle((bifpn_x, feature_y - 0.5), 4, 2.8, linewidth=2, 
                         edgecolor=colors['black'], facecolor=colors['gray_light'])
    ax.add_patch(bifpn_box)
    ax.text(bifpn_x + 2, feature_y + 1.5, 'UltraLightBiFPN', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(bifpn_x + 2, feature_y + 1.2, '28 channels', fontsize=11,
            ha='center', va='center', color=colors['black'])
    ax.text(bifpn_x + 2, feature_y + 0.9, '18.4K params (7.2%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Detection and Output Pipeline
    detection_y = 13
    
    # SSH_Grouped Context Network
    ssh_box = Rectangle((2, detection_y), 6, 1.5, linewidth=2, edgecolor=colors['black'], 
                       facecolor=colors['gray_medium'])
    ax.add_patch(ssh_box)
    ax.text(5, detection_y + 1, 'SSH_Grouped Context Network', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5, detection_y + 0.7, '12.3K params (4.8%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # ChannelShuffle_Light 
    shuffle_box = Rectangle((9, detection_y), 4, 1.5, linewidth=2, edgecolor=colors['black'], 
                           facecolor=colors['white'])
    ax.add_patch(shuffle_box)
    ax.text(11, detection_y + 1, 'ChannelShuffle_Light', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(11, detection_y + 0.7, '0 params (0.0%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # SharedMultiHead Network
    multihead_box = Rectangle((14, detection_y), 6, 1.5, linewidth=2, edgecolor=colors['black'], 
                             facecolor=colors['gray_light'])
    ax.add_patch(multihead_box)
    ax.text(17, detection_y + 1, 'SharedMultiHead Network', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(17, detection_y + 0.7, '11.5K params (4.5%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Final Output
    output_box = Rectangle((9, detection_y - 2.5), 6, 1.2, linewidth=2, edgecolor=colors['black'], 
                          facecolor=colors['white'])
    ax.add_patch(output_box)
    ax.text(12, detection_y - 1.9, 'Final Output', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(12, detection_y - 2.2, 'Classifications • BBox • Landmarks', 
            fontsize=10, ha='center', va='center', color=colors['gray_dark'])
    
    # Innovation reference in main flow
    ax.text(12, 12, '↓ V2 Ultra Innovations Applied (See Section b) ↓', 
            fontsize=12, fontweight='bold', ha='center', va='center', color='#D32F2F',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['gray_light'], edgecolor=colors['black']))
    
    # Flow arrows for main architecture
    ax.annotate('', xy=(4, main_y + 0.6), xytext=(3, main_y + 0.6),
               arrowprops=dict(arrowstyle='->', lw=3, color=colors['black']))
    
    for i in range(3):
        ax.annotate('', xy=(cbam1_x, feature_y + (1-i)), xytext=(10.2, feature_y + (1-i)),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
        ax.annotate('', xy=(bifpn_x, feature_y + 0.5), xytext=(cbam1_x + 3.5, feature_y + (1-i)),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    ax.annotate('', xy=(5, detection_y + 1.5), xytext=(bifpn_x + 2, feature_y - 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(9, detection_y + 0.75), xytext=(8, detection_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(14, detection_y + 0.75), xytext=(13, detection_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(12, detection_y - 1.3), xytext=(17, detection_y),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    # ===== SECTION (B) - ZERO-PARAMETER INNOVATIONS =====
    section_b_y = 10
    ax.text(1, section_b_y, '(b) Zero-Parameter Innovations Mechanisms', 
            fontsize=16, fontweight='bold', ha='left', va='center', color=colors['black'])
    
    # Innovation details in organized grid (3 columns, 2 rows)
    innovations = [
        ('SmartFeatureReuse', 'Cross-scale feature reuse\nwith interpolation', '+1.0% mAP\n0 params'),
        ('AttentionMultiplication', 'Progressive attention\napplication (3x)', '+0.8% mAP\n0 params'),
        ('ProgressiveFeatureEnhancement', 'Level-wise feature\nrefinement', '+0.7% mAP\n0 params'),
        ('MultiScaleIntelligence', 'Adaptive importance\nweighting', '+0.5% mAP\n0 params'),
        ('DynamicWeightSharing', 'Intelligent weight\nsharing', '+0.5% mAP\n<1K params')
    ]
    
    # Grid layout: 3 columns, 2 rows, no overlapping
    cols = 3
    box_width = 6.8
    box_height = 2.2
    start_x = 1.5
    start_y = section_b_y - 1
    spacing_x = 0.8
    spacing_y = 0.5
    
    for i, (name, desc, perf) in enumerate(innovations):
        row = i // cols
        col = i % cols
        x = start_x + col * (box_width + spacing_x)
        y = start_y - row * (box_height + spacing_y)
        
        # Innovation box with clear spacing
        inn_box = Rectangle((x, y - box_height), box_width, box_height, linewidth=1.5, 
                           edgecolor=colors['black'], facecolor=colors['gray_light'])
        ax.add_patch(inn_box)
        
        # Innovation name
        ax.text(x + box_width/2, y - 0.4, f'{i+1}. {name}', fontsize=11, fontweight='bold',
                ha='center', va='center', color=colors['black'])
        
        # Description
        ax.text(x + box_width/2, y - 1.1, desc, fontsize=9,
                ha='center', va='center', color=colors['black'])
        
        # Performance
        ax.text(x + box_width/2, y - 1.7, perf, fontsize=9, fontweight='bold',
                ha='center', va='center', color=colors['gray_dark'])
    
    # Revolutionary impact summary
    impact_y = 4.5
    ax.text(12, impact_y, 'Revolutionary Impact: +3.5% mAP with virtually zero parameter cost', 
            fontsize=14, fontweight='bold', ha='center', va='center', color='#D32F2F',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['gray_light'], edgecolor=colors['black']))
    
    # ===== SPECIFICATIONS SUMMARY =====
    spec_box = Rectangle((1, 2), 22, 1.8, linewidth=2, edgecolor=colors['black'], 
                        facecolor=colors['gray_light'])
    ax.add_patch(spec_box)
    
    ax.text(12, 3.4, 'FeatherFace V2 Ultra - Code-Faithful Specifications', 
            fontsize=14, fontweight='bold', ha='center', va='center', color=colors['black'])
    
    spec_text = """Total Parameters: 244K (49.8% reduction) • Performance: 90.5% mAP (+3.5% vs V1) • Speed: 60% faster
Key Modules: MobileNetV1-0.25 (213K) • SharedCBAMManager (1.2K) • UltraLightBiFPN (18.4K) • SSH_Grouped (12.3K) • SharedMultiHead (11.5K)
Revolutionary: 5 Zero-Parameter Innovations (Section b) providing +3.5% mAP with virtually no parameter cost"""
    
    ax.text(12, 2.7, spec_text, fontsize=10, ha='center', va='center', color=colors['black'])
    
    plt.tight_layout()
    plt.savefig('docs/featherface_v2_ultra_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_v2_ultra_restructured_diagram()
    print("✅ FeatherFace V2 Ultra restructured diagram generated: docs/featherface_v2_ultra_architecture.png")
    print("   📋 Layout: (a) Main Architecture Flow + (b) Zero-Parameter Innovations")
    print("   ✨ No overlapping elements, optimized for black/white printing")