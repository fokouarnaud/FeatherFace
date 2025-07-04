#!/usr/bin/env python3
"""
FeatherFace V1 Academic Architecture Diagram Generator
Creates a clean, professional diagram following academic standards with CORRECT SEQUENCE
Sequence: Input → Backbone → Features → CBAM1 → BiFPN → CBAM2 → Detection → Context → Output
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_v1_academic_diagram():
    """Create FeatherFace V1 academic architecture diagram with correct sequence"""
    
    # Create figure with landscape academic layout
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Academic color palette - professional and accessible
    colors = {
        'input': '#E3F2FD',           # Light blue
        'backbone': '#BBDEFB',        # Medium blue  
        'features': '#90CAF9',        # Blue
        'cbam1': '#66BB6A',          # Green (First CBAM)
        'bifpn': '#FFB74D',          # Orange
        'cbam2': '#4CAF50',          # Dark Green (Second CBAM)
        'detection': '#F06292',       # Pink
        'context': '#FFCDD2',        # Light pink
        'output': '#CE93D8',         # Purple
        'text': '#263238',           # Dark blue-grey
        'arrow': '#546E7A'           # Medium grey
    }
    
    # Title and subtitle - academic format
    ax.text(10, 11.5, 'FeatherFace V1 Architecture', fontsize=22, fontweight='bold', 
            ha='center', va='center', color=colors['text'])
    ax.text(10, 11.1, '487K Parameters • 87.0% mAP • Double CBAM Pipeline', fontsize=16, 
            ha='center', va='center', color='#D32F2F')
    
    # CORRECT SEQUENCE IMPLEMENTATION
    
    # Stage 1: Input
    input_box = FancyBboxPatch((0.5, 9.5), 2.5, 1.2, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['input'], 
                              edgecolor='#424242', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.75, 10.3, 'Input', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(1.75, 9.9, '640×640×3', fontsize=10,
            ha='center', va='center', color=colors['text'])
    
    # Stage 2: MobileNet-0.25 Backbone
    backbone_box = FancyBboxPatch((3.5, 9.5), 3, 1.2, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='#424242', linewidth=1.5)
    ax.add_patch(backbone_box)
    ax.text(5, 10.3, 'MobileNet-0.25 Backbone', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(5, 9.9, '213K params', fontsize=10,
            ha='center', va='center', color=colors['text'])
    
    # Stage 3: Multi-scale Features [P3:64ch, P4:128ch, P5:256ch]
    feature_y = 8.5
    features = [
        (7, feature_y + 0.6, 'P3', '64ch'),
        (7, feature_y, 'P4', '128ch'), 
        (7, feature_y - 0.6, 'P5', '256ch')
    ]
    
    for x, y, label, channels in features:
        feat_box = FancyBboxPatch((x, y - 0.3), 2, 0.6, 
                                 boxstyle="round,pad=0.03", 
                                 facecolor=colors['features'], 
                                 edgecolor='#424242', linewidth=1.5)
        ax.add_patch(feat_box)
        ax.text(x + 1, y, f'{label}: {channels}', fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 4: First Attention Mechanisms (CBAM) → Enhanced Features [A3, A4, A5]
    cbam1_x = 9.5
    ax.text(cbam1_x + 1, 9.2, 'First CBAM', fontsize=11, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    enhanced_features = [
        (cbam1_x, feature_y + 0.6, 'A3', 'Enhanced'),
        (cbam1_x, feature_y, 'A4', 'Enhanced'), 
        (cbam1_x, feature_y - 0.6, 'A5', 'Enhanced')
    ]
    
    for x, y, label, desc in enhanced_features:
        cbam_box = FancyBboxPatch((x, y - 0.3), 2, 0.6, 
                                 boxstyle="round,pad=0.03", 
                                 facecolor=colors['cbam1'], 
                                 edgecolor='#424242', linewidth=1.5)
        ax.add_patch(cbam_box)
        ax.text(x + 1, y, f'{label}: {desc}', fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 5: BiFPN → Fused Features [F3:74ch, F4:74ch, F5:74ch]
    bifpn_x = 12
    bifpn_box = FancyBboxPatch((bifpn_x, 7.5), 3, 2, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['bifpn'], 
                              edgecolor='#424242', linewidth=2)
    ax.add_patch(bifpn_box)
    ax.text(bifpn_x + 1.5, 8.8, 'BiFPN', fontsize=14, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(bifpn_x + 1.5, 8.4, 'Feature Fusion', fontsize=11,
            ha='center', va='center', color='white')
    ax.text(bifpn_x + 1.5, 8, '114K params', fontsize=10,
            ha='center', va='center', color='white')
    
    # BiFPN internal structure
    for i, (label, y_offset) in enumerate([('TD-5', 0.4), ('TD-4', 0.0), ('TD-3', -0.4)]):
        td_box = FancyBboxPatch((bifpn_x + 0.2, 8.5 + y_offset - 0.15), 0.8, 0.3,
                               boxstyle="round,pad=0.02",
                               facecolor='#FFA726', edgecolor='white', linewidth=1)
        ax.add_patch(td_box)
        ax.text(bifpn_x + 0.6, 8.5 + y_offset, label, fontsize=8, fontweight='bold',
                ha='center', va='center', color='white')
    
    for i, (label, y_offset) in enumerate([('BU-3', -0.4), ('BU-4', 0.0), ('BU-5', 0.4)]):
        bu_box = FancyBboxPatch((bifpn_x + 2, 8.5 + y_offset - 0.15), 0.8, 0.3,
                               boxstyle="round,pad=0.02",
                               facecolor='#FF8A65', edgecolor='white', linewidth=1)
        ax.add_patch(bu_box)
        ax.text(bifpn_x + 2.4, 8.5 + y_offset, label, fontsize=8, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 6: Fused Features [F3:74ch, F4:74ch, F5:74ch]
    fused_x = 15.5
    fused_features = [
        (fused_x, feature_y + 0.6, 'F3', '74ch'),
        (fused_x, feature_y, 'F4', '74ch'), 
        (fused_x, feature_y - 0.6, 'F5', '74ch')
    ]
    
    for x, y, label, channels in fused_features:
        fused_box = FancyBboxPatch((x, y - 0.3), 2, 0.6, 
                                  boxstyle="round,pad=0.03", 
                                  facecolor='#FF8F00', 
                                  edgecolor='#424242', linewidth=1.5)
        ax.add_patch(fused_box)
        ax.text(x + 1, y, f'{label}: {channels}', fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 7: Second CBAM → Refined Features [R3, R4, R5]
    cbam2_x = 18
    ax.text(cbam2_x + 1, 9.2, 'Second CBAM', fontsize=11, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    refined_features = [
        (cbam2_x, feature_y + 0.6, 'R3', 'Refined'),
        (cbam2_x, feature_y, 'R4', 'Refined'), 
        (cbam2_x, feature_y - 0.6, 'R5', 'Refined')
    ]
    
    for x, y, label, desc in refined_features:
        cbam2_box = FancyBboxPatch((x, y - 0.3), 2, 0.6, 
                                  boxstyle="round,pad=0.03", 
                                  facecolor=colors['cbam2'], 
                                  edgecolor='#424242', linewidth=1.5)
        ax.add_patch(cbam2_box)
        ax.text(x + 1, y, f'{label}: {desc}', fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 8: Detection Heads
    detection_y = 6
    detection_heads = [
        (7, detection_y, 'Head P3'),
        (10, detection_y, 'Head P4'),
        (13, detection_y, 'Head P5')
    ]
    
    for x, y, label in detection_heads:
        det_box = FancyBboxPatch((x, y), 2.5, 0.8, 
                                boxstyle="round,pad=0.03", 
                                facecolor=colors['detection'], 
                                edgecolor='#424242', linewidth=1.5)
        ax.add_patch(det_box)
        ax.text(x + 1.25, y + 0.4, label, fontsize=11, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Stage 9: Context Enhancement
    context_y = 4.5
    
    # DCN Context
    dcn_box = FancyBboxPatch((6, context_y), 4, 1, 
                            boxstyle="round,pad=0.03", 
                            facecolor=colors['context'], 
                            edgecolor='#424242', linewidth=1.5)
    ax.add_patch(dcn_box)
    ax.text(8, context_y + 0.6, 'DCN Context', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(8, context_y + 0.2, '148K params', fontsize=10,
            ha='center', va='center', color=colors['text'])
    
    # Channel Shuffle
    shuffle_box = FancyBboxPatch((11, context_y), 4, 1, 
                                boxstyle="round,pad=0.03", 
                                facecolor=colors['context'], 
                                edgecolor='#424242', linewidth=1.5)
    ax.add_patch(shuffle_box)
    ax.text(13, context_y + 0.6, 'Channel Shuffle', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(13, context_y + 0.2, '0 params', fontsize=10,
            ha='center', va='center', color=colors['text'])
    
    # Stage 10: Final Output
    output_box = FancyBboxPatch((8, 2.5), 6, 1.2, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['output'], 
                               edgecolor='#424242', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(11, 3.3, 'Final Output', fontsize=14, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(11, 2.9, 'bbox_reg • classifications • landmarks', fontsize=11,
            ha='center', va='center', color='white')
    
    # Flow arrows - correct sequence
    arrow_style = dict(arrowstyle='->', lw=2.5, color=colors['arrow'])
    
    # Main horizontal flow
    main_y = 10.1
    ax.annotate('', xy=(3.5, main_y), xytext=(3, main_y), arrowprops=arrow_style)
    ax.annotate('', xy=(7, main_y), xytext=(6.5, main_y), arrowprops=arrow_style)
    
    # Features to First CBAM
    for y_offset in [0.6, 0, -0.6]:
        ax.annotate('', xy=(9.5, feature_y + y_offset), xytext=(9, feature_y + y_offset), 
                   arrowprops=arrow_style)
    
    # First CBAM to BiFPN
    for y_offset in [0.6, 0, -0.6]:
        ax.annotate('', xy=(12, 8.5), xytext=(11.5, feature_y + y_offset), 
                   arrowprops=arrow_style)
    
    # BiFPN to Fused Features
    for y_offset in [0.6, 0, -0.6]:
        ax.annotate('', xy=(15.5, feature_y + y_offset), xytext=(15, 8.5), 
                   arrowprops=arrow_style)
    
    # Fused Features to Second CBAM
    for y_offset in [0.6, 0, -0.6]:
        ax.annotate('', xy=(18, feature_y + y_offset), xytext=(17.5, feature_y + y_offset), 
                   arrowprops=arrow_style)
    
    # Second CBAM to Detection Heads
    for i, (x, y, _) in enumerate(detection_heads):
        ax.annotate('', xy=(x + 1.25, y + 0.8), xytext=(19, feature_y + (0.6 - 0.6*i)), 
                   arrowprops=arrow_style)
    
    # Detection to Context
    ax.annotate('', xy=(8, context_y + 1), xytext=(8.25, detection_y), arrowprops=arrow_style)
    ax.annotate('', xy=(13, context_y + 1), xytext=(11.25, detection_y), arrowprops=arrow_style)
    
    # Context to Output
    ax.annotate('', xy=(10, 3.7), xytext=(8, context_y), arrowprops=arrow_style)
    ax.annotate('', xy=(12, 3.7), xytext=(13, context_y), arrowprops=arrow_style)
    
    # Academic specification panel
    spec_box = FancyBboxPatch((0.5, 0.5), 19, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1)
    ax.add_patch(spec_box)
    
    ax.text(10, 1.6, 'Architecture Specifications', fontsize=14, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(10, 1.2, 'Total Parameters: 487K • Performance: 87.0% mAP on WIDERFace • Inference: 30+ FPS', 
            fontsize=12, ha='center', va='center', color=colors['text'])
    ax.text(10, 0.8, 'Double CBAM Pipeline: Features → CBAM₁ → BiFPN → CBAM₂ → Detection → Context → Output', 
            fontsize=11, ha='center', va='center', color='#D32F2F')
    
    plt.tight_layout()
    plt.savefig('docs/featherface_v1_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_v1_academic_diagram()
    print("✅ FeatherFace V1 academic architecture diagram with CORRECT SEQUENCE generated: docs/featherface_v1_architecture.png")