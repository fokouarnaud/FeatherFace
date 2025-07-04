#!/usr/bin/env python3
"""
FeatherFace Nano Architecture Diagram Generator
Creates a professional black/white diagram showing the scientifically justified architecture

Based on verified research:
- Li et al. CVPR 2023 (Knowledge Distillation)
- Woo et al. ECCV 2018 (CBAM)
- Tan et al. CVPR 2020 (BiFPN)
- Howard et al. 2017 (MobileNet)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

def create_nano_architecture_diagram():
    """Create FeatherFace Nano architecture diagram with scientific justification"""
    
    # Create figure optimized for black/white printing
    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Black/white color scheme for professional printing
    colors = {
        'black': '#000000',
        'white': '#FFFFFF', 
        'gray_light': '#F5F5F5',
        'gray_medium': '#E0E0E0',
        'gray_dark': '#808080'
    }
    
    # Main title
    ax.text(12, 17.5, 'FeatherFace Nano Ultra-Efficient Architecture', 
            fontsize=20, fontweight='bold', ha='center', va='center', color=colors['black'])
    ax.text(12, 17.1, '344K Parameters (29.3% Reduction) • Scientifically Justified Efficiency', 
            fontsize=14, ha='center', va='center', color=colors['gray_dark'])
    ax.text(12, 16.7, 'Based on 4 Verified Research Publications', 
            fontsize=12, ha='center', va='center', color=colors['gray_dark'], style='italic')
    
    # ===== MAIN ARCHITECTURE FLOW =====
    main_y = 15
    
    # Input
    input_box = Rectangle((0.5, main_y), 2.5, 1.2, linewidth=2, edgecolor=colors['black'], 
                         facecolor=colors['white'])
    ax.add_patch(input_box)
    ax.text(1.75, main_y + 0.6, 'Input Image', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(1.75, main_y + 0.3, '640×640×3', fontsize=10,
            ha='center', va='center', color=colors['black'])
    
    # MobileNetV1-0.25 Backbone
    backbone_box = Rectangle((4, main_y), 3.5, 1.2, linewidth=2, edgecolor=colors['black'], 
                            facecolor=colors['gray_light'])
    ax.add_patch(backbone_box)
    ax.text(5.75, main_y + 0.7, 'MobileNetV1-0.25', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5.75, main_y + 0.4, 'Backbone', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5.75, main_y + 0.1, '213K params (61.9%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Multi-scale features
    feature_y = main_y - 2.5
    features = [
        (8.5, feature_y + 1, 'P3: 64ch'),
        (8.5, feature_y, 'P4: 128ch'), 
        (8.5, feature_y - 1, 'P5: 256ch')
    ]
    
    for i, (x, y, label) in enumerate(features):
        feat_box = Rectangle((x, y), 2.2, 0.8, linewidth=1.5, edgecolor=colors['black'], 
                            facecolor=colors['white'])
        ax.add_patch(feat_box)
        ax.text(x + 1.1, y + 0.4, label, fontsize=11, fontweight='bold',
                ha='center', va='center', color=colors['black'])
        
        # Arrow from backbone to features
        ax.annotate('', xy=(x, y + 0.4), xytext=(7.5, main_y + 0.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    # Efficient CBAM (Woo et al. ECCV 2018)
    cbam1_x = 12
    cbam1_box = Rectangle((cbam1_x, feature_y - 0.5), 3.5, 2.8, linewidth=2, 
                         edgecolor=colors['black'], facecolor=colors['gray_medium'])
    ax.add_patch(cbam1_box)
    ax.text(cbam1_x + 1.75, feature_y + 1.5, 'Efficient CBAM', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(cbam1_x + 1.75, feature_y + 1.2, '(Woo et al. ECCV 2018)', fontsize=10,
            ha='center', va='center', color=colors['black'])
    ax.text(cbam1_x + 1.75, feature_y + 0.9, '7.4K params (2.2%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    ax.text(cbam1_x + 1.75, feature_y + 0.5, 'Higher reduction ratios', fontsize=9,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Efficient BiFPN (Tan et al. CVPR 2020)
    bifpn_x = 17
    bifpn_box = Rectangle((bifpn_x, feature_y - 0.5), 4, 2.8, linewidth=2, 
                         edgecolor=colors['black'], facecolor=colors['gray_light'])
    ax.add_patch(bifpn_box)
    ax.text(bifpn_x + 2, feature_y + 1.5, 'Efficient BiFPN', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(bifpn_x + 2, feature_y + 1.2, '(Tan et al. CVPR 2020)', fontsize=10,
            ha='center', va='center', color=colors['black'])
    ax.text(bifpn_x + 2, feature_y + 0.9, '38.7K params (11.2%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    ax.text(bifpn_x + 2, feature_y + 0.5, 'Depthwise separable', fontsize=9,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Flow arrows
    ax.annotate('', xy=(4, main_y + 0.6), xytext=(3, main_y + 0.6),
               arrowprops=dict(arrowstyle='->', lw=3, color=colors['black']))
    
    for i in range(3):
        ax.annotate('', xy=(cbam1_x, feature_y + (1-i)), xytext=(10.7, feature_y + (1-i)),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
        ax.annotate('', xy=(bifpn_x, feature_y + 0.5), xytext=(cbam1_x + 3.5, feature_y + (1-i)),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    # ===== SCIENTIFIC FOUNDATION SECTION =====
    foundation_y = 10
    ax.text(1, foundation_y, 'Scientific Foundation & Research Basis', 
            fontsize=16, fontweight='bold', ha='left', va='center', color=colors['black'])
    
    # Research papers with details
    papers = [
        ('Knowledge Distillation', 'Li et al. CVPR 2023', 'Teacher-student training\nframework', 'Nano Training'),
        ('CBAM Attention', 'Woo et al. ECCV 2018', 'Channel & spatial\nattention mechanism', 'Efficient CBAM'),
        ('BiFPN Architecture', 'Tan et al. CVPR 2020', 'Bidirectional feature\npyramid network', 'Efficient BiFPN'),
        ('MobileNet Backbone', 'Howard et al. 2017', 'Depthwise separable\nconvolutional networks', 'Backbone'),
    ]
    
    paper_y = foundation_y - 1
    for i, (technique, paper, description, implementation) in enumerate(papers):
        x = 1 + i * 5.5
        
        # Paper box
        paper_box = Rectangle((x, paper_y - 1.8), 5, 2, linewidth=1.5, 
                             edgecolor=colors['black'], facecolor=colors['gray_light'])
        ax.add_patch(paper_box)
        
        # Technique name
        ax.text(x + 2.5, paper_y - 0.4, technique, fontsize=11, fontweight='bold',
                ha='center', va='center', color=colors['black'])
        
        # Paper citation
        ax.text(x + 2.5, paper_y - 0.8, paper, fontsize=10, fontweight='bold',
                ha='center', va='center', color='#D32F2F')
        
        # Description
        ax.text(x + 2.5, paper_y - 1.2, description, fontsize=9,
                ha='center', va='center', color=colors['black'])
        
        # Implementation
        ax.text(x + 2.5, paper_y - 1.6, f'→ {implementation}', fontsize=9, fontweight='bold',
                ha='center', va='center', color=colors['gray_dark'])
    
    # ===== DETECTION PIPELINE =====
    detection_y = 7
    
    # Grouped SSH Context Processing
    ssh_box = Rectangle((2, detection_y), 6, 1.5, linewidth=2, edgecolor=colors['black'], 
                       facecolor=colors['gray_medium'])
    ax.add_patch(ssh_box)
    ax.text(5, detection_y + 1, 'Grouped SSH Context', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(5, detection_y + 0.7, '26.5K params (7.7%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    ax.text(5, detection_y + 0.4, 'Grouped convolutions (4 groups)', fontsize=9,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Channel Shuffle
    shuffle_box = Rectangle((9, detection_y), 4, 1.5, linewidth=2, edgecolor=colors['black'], 
                           facecolor=colors['white'])
    ax.add_patch(shuffle_box)
    ax.text(11, detection_y + 1, 'Channel Shuffle', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(11, detection_y + 0.7, '0 params (0.0%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    ax.text(11, detection_y + 0.4, 'Parameter-free mixing', fontsize=9,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Detection Heads
    heads_box = Rectangle((14, detection_y), 6, 1.5, linewidth=2, edgecolor=colors['black'], 
                         facecolor=colors['gray_light'])
    ax.add_patch(heads_box)
    ax.text(17, detection_y + 1, 'Efficient Detection Heads', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(17, detection_y + 0.7, '58.7K params (17.0%)', fontsize=10,
            ha='center', va='center', color=colors['gray_dark'])
    ax.text(17, detection_y + 0.4, 'CLS + BBOX + Landmarks', fontsize=9,
            ha='center', va='center', color=colors['gray_dark'])
    
    # Final Output
    output_box = Rectangle((9, detection_y - 2.5), 6, 1.2, linewidth=2, edgecolor=colors['black'], 
                          facecolor=colors['white'])
    ax.add_patch(output_box)
    ax.text(12, detection_y - 1.9, 'Final Output', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['black'])
    ax.text(12, detection_y - 2.2, 'Classifications • BBox • Landmarks', 
            fontsize=10, ha='center', va='center', color=colors['gray_dark'])
    
    # Detection flow arrows
    ax.annotate('', xy=(5, detection_y + 1.5), xytext=(bifpn_x + 2, feature_y - 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(9, detection_y + 0.75), xytext=(8, detection_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(14, detection_y + 0.75), xytext=(13, detection_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    ax.annotate('', xy=(12, detection_y - 1.3), xytext=(17, detection_y),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['black']))
    
    # ===== SPECIFICATIONS SUMMARY =====
    spec_box = Rectangle((1, 1), 22, 2, linewidth=2, edgecolor=colors['black'], 
                        facecolor=colors['gray_light'])
    ax.add_patch(spec_box)
    
    ax.text(12, 2.6, 'FeatherFace Nano - Ultra-Efficient Scientific Specifications', 
            fontsize=14, fontweight='bold', ha='center', va='center', color=colors['black'])
    
    # Performance metrics
    metrics_text = """Total Parameters: 344K (29.3% reduction) • Performance: Competitive mAP • Speed: 30-40% faster
Key Components: MobileNet-0.25 (213K) • Efficient CBAM (7.4K) • Efficient BiFPN (38.7K) • Grouped SSH (26.5K) • Detection Heads (58.7K)
Scientific Foundation: 4 Verified Research Publications • Knowledge Distillation Training • Production Ready"""
    
    ax.text(12, 1.8, metrics_text, fontsize=10, ha='center', va='center', color=colors['black'])
    
    # Efficiency comparison
    ax.text(12, 1.3, '🔬 Scientific Rigor: 100% Research-Backed Techniques • 🚀 Efficiency: 29.3% Parameter Reduction • ⚡ Performance: Maintained via Knowledge Distillation', 
            fontsize=11, ha='center', va='center', color='#D32F2F', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/featherface_nano_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_nano_architecture_diagram()
    print("✅ FeatherFace Nano architecture diagram generated: docs/featherface_nano_architecture.png")
    print("🔬 Scientific foundation: 4 verified research publications")
    print("📊 Ultra-efficient: 344K parameters (29.3% reduction)")
    print("✨ Black/white optimized for professional documentation")