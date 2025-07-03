#!/usr/bin/env python3
"""
Create FeatherFace architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create a visual architecture diagram for FeatherFace"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'backbone': '#B8E6B8', 
        'fpn': '#FFE4B5',
        'attention': '#FFB6C1',
        'detection': '#DDA0DD',
        'output': '#F0E68C'
    }
    
    # Title
    ax.text(7, 11.5, 'FeatherFace Architecture', fontsize=20, fontweight='bold', ha='center')
    
    # Input
    input_box = FancyBboxPatch((5.5, 10), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 10.4, 'Input Image\n640×640×3', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Backbone
    backbone_box = FancyBboxPatch((5, 8.5), 4, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['backbone'], 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(7, 9, 'MobileNetV1 0.25x\n(Backbone)\n213K parameters', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Feature maps P3, P4, P5
    for i, (name, ch) in enumerate([('P3', '32ch'), ('P4', '64ch'), ('P5', '128ch')]):
        x_pos = 3 + i * 4
        p_box = FancyBboxPatch((x_pos, 7), 1.5, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['backbone'], 
                               edgecolor='gray')
        ax.add_patch(p_box)
        ax.text(x_pos + 0.75, 7.3, f'{name}\n{ch}', ha='center', va='center', fontsize=8)
    
    # BiFPN
    bifpn_box = FancyBboxPatch((4.5, 5.5), 5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fpn'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(bifpn_box)
    ax.text(7, 6, 'BiFPN\n(Bidirectional FPN)\n84K→10K params (V2)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Enhanced features F3, F4, F5
    for i, name in enumerate(['F3', 'F4', 'F5']):
        x_pos = 3 + i * 4
        f_box = FancyBboxPatch((x_pos, 4.5), 1.5, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['fpn'], 
                               edgecolor='gray')
        ax.add_patch(f_box)
        ax.text(x_pos + 0.75, 4.8, f'{name}\n52ch', ha='center', va='center', fontsize=8)
    
    # CBAM Attention
    cbam_box = FancyBboxPatch((4.5, 3), 5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['attention'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(cbam_box)
    ax.text(7, 3.5, 'CBAM Attention\n(Channel + Spatial)\n12K→4K params (V2)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Attention features A3, A4, A5
    for i, name in enumerate(['A3', 'A4', 'A5']):
        x_pos = 3 + i * 4
        a_box = FancyBboxPatch((x_pos, 2), 1.5, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['attention'], 
                               edgecolor='gray')
        ax.add_patch(a_box)
        ax.text(x_pos + 0.75, 2.3, f'{name}\n52ch', ha='center', va='center', fontsize=8)
    
    # SSH Detection
    ssh_box = FancyBboxPatch((4.5, 0.5), 5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['detection'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(ssh_box)
    ax.text(7, 1, 'SSH Detection\n(Context Enhancement)\n174K→18K params (V2)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output boxes
    output_names = ['Classifications\n[16800, 2]', 'Bounding Boxes\n[16800, 4]', 'Landmarks\n[16800, 10]']
    for i, name in enumerate(output_names):
        x_pos = 1.5 + i * 4
        out_box = FancyBboxPatch((x_pos, -1), 3, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['output'], 
                                 edgecolor='black', linewidth=1.5)
        ax.add_patch(out_box)
        ax.text(x_pos + 1.5, -0.6, name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows for data flow
    arrow_props = dict(arrowstyle='->', lw=2, color='blue')
    
    # Input to backbone
    ax.annotate('', xy=(7, 8.5), xytext=(7, 9.2), arrowprops=arrow_props)
    
    # Backbone to features
    for i in range(3):
        x_pos = 3.75 + i * 4
        ax.annotate('', xy=(x_pos, 7.6), xytext=(7, 8.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # P features to BiFPN
    ax.annotate('', xy=(7, 5.5), xytext=(7, 7), arrowprops=arrow_props)
    
    # BiFPN to F features
    for i in range(3):
        x_pos = 3.75 + i * 4
        ax.annotate('', xy=(x_pos, 5.1), xytext=(7, 5.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # F features to CBAM
    ax.annotate('', xy=(7, 3), xytext=(7, 4.5), arrowprops=arrow_props)
    
    # CBAM to A features  
    for i in range(3):
        x_pos = 3.75 + i * 4
        ax.annotate('', xy=(x_pos, 2.6), xytext=(7, 3), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # A features to SSH
    ax.annotate('', xy=(7, 0.5), xytext=(7, 2), arrowprops=arrow_props)
    
    # SSH to outputs
    for i in range(3):
        x_pos = 3 + i * 4
        ax.annotate('', xy=(x_pos, -0.2), xytext=(7, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Add parameter comparison table
    table_x, table_y = 10.5, 7
    ax.text(table_x, table_y + 1, 'Parameter Comparison', fontsize=12, fontweight='bold')
    
    table_data = [
        ['Component', 'V1', 'V2'],
        ['MobileNet', '213K', '213K'],
        ['BiFPN', '84K', '10K'],  
        ['CBAM', '12K', '4K'],
        ['SSH', '174K', '18K'],
        ['Total', '488K', '256K'],
        ['Reduction', '-', '56.7%']
    ]
    
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            bg_color = 'lightgray' if i == 0 else 'white'
            font_weight = 'bold' if i == 0 or j == 0 else 'normal'
            ax.text(table_x + j * 1.2, table_y - i * 0.3, cell, 
                   fontsize=9, fontweight=font_weight, ha='center',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor=bg_color, alpha=0.7))
    
    # Add legend
    legend_x, legend_y = 0.5, 8
    ax.text(legend_x, legend_y + 1, 'Legend:', fontsize=12, fontweight='bold')
    
    legend_items = [
        ('Input/Output', colors['input']),
        ('Backbone', colors['backbone']),
        ('Feature Pyramid', colors['fpn']),
        ('Attention', colors['attention']),
        ('Detection', colors['detection'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        rect = FancyBboxPatch((legend_x, legend_y - i * 0.4), 0.3, 0.25, 
                              boxstyle="round,pad=0.02", 
                              facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + 0.4, legend_y - i * 0.4 + 0.125, label, 
               fontsize=9, va='center')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and save diagram
    fig = create_architecture_diagram()
    
    # Save as high-resolution PNG
    output_path = 'docs/architecture_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Architecture diagram saved to: {output_path}")
    
    # Show the plot
    plt.show()