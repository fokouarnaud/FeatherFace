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
    """Setup matplotlib for publication-quality black and white figures with improved typography"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,  # Increased base font size
        'axes.linewidth': 1.5,
        'axes.labelsize': 12,
        'axes.titlesize': 16,  # Larger titles
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 18,  # Much larger figure title
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white'
    })

def create_featherface_nano_b_diagram():
    """Create FeatherFace Nano-B architecture diagram - Landscape Multi-Level Style"""
    
    # Create figure with optimized landscape dimensions
    fig_width = 28  # Wider for better balance
    fig_height = 18  # Taller for better proportion
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Enhanced black and white styling with stronger visual hierarchy
    colors = {
        'main_border': '#000000',  # Pure black for main elements
        'component_fill': '#FFFFFF',  # Pure white for main components
        'component_border': '#000000',  # Black borders
        'secondary_fill': '#F8F8F8',  # Very light gray for secondary
        'critical_fill': '#F0F0F0',  # Light gray for critical components
        'text_main': '#000000',  # Pure black for main text
        'text_secondary': '#333333',  # Dark gray for secondary text
        'text_detail': '#555555',  # Medium gray for details
        'arrow_main': '#000000',  # Black for main data flow
        'arrow_bifpn': '#FF0000',  # Red for BiFPN connections
        'arrow_distill': '#666666',  # Gray for distillation
        'background': '#FFFFFF',  # Pure white background
        'highlight': '#E8E8E8',  # Light gray for highlights
        'accent': '#DDDDDD',  # Accent for special elements
        'grid': '#F0F0F0'  # Very light gray for grid
    }
    
    # Typography hierarchy
    fonts = {
        'title_main': {'size': 18, 'weight': 'bold'},
        'title_section': {'size': 14, 'weight': 'bold'},
        'title_module': {'size': 12, 'weight': 'bold'},
        'text_normal': {'size': 11, 'weight': 'normal'},
        'text_detail': {'size': 10, 'weight': 'normal'},
        'text_annotation': {'size': 9, 'weight': 'normal'}
    }
    
    # Border hierarchy for visual importance
    borders = {
        'critical': 3,    # Backbone, BiFPN
        'important': 2,   # CBAM, SSH, Heads
        'standard': 1.5,  # Other modules
        'auxiliary': 1    # Labels, annotations
    }
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Precise grid-based layout for perfect alignment
    # Grid system: 28x18 with 0.5 unit precision
    grid_unit = 0.5
    
    # Component dimensions optimized for visual balance
    dimensions = {
        'input': {'w': 6, 'h': 1.6},
        'backbone': {'w': 6, 'h': 2.2},  # Larger for critical importance
        'level_module': {'w': 4.2, 'h': 2.0},  # Slightly larger for readability
        'teacher': {'w': 7, 'h': 4.0},  # Properly sized for importance
        'output': {'w': 6, 'h': 1.6},
        'head_box': {'w': 4.2, 'h': 0.65},
        'stage_box': {'w': 1.8, 'h': 0.8}  # New: stage connection boxes
    }
    
    # Grid-aligned positions for optimal balance
    positions = {
        # Left side - input flow (better centered)
        'input': {'x': 1, 'y': 15, 'w': dimensions['input']['w'], 'h': dimensions['input']['h']},
        'backbone': {'x': 1, 'y': 12, 'w': dimensions['backbone']['w'], 'h': dimensions['backbone']['h']},
        
        # Center - multi-level processing (better spaced)
        'p3_x': 10, 'p4_x': 15, 'p5_x': 20,  # Wider spacing for clarity
        'level_spacing': 5,  # Increased spacing
        
        # Y positions for processing layers (better distributed)
        'cbam1_y': 14.5,
        'bifpn_y': 12,
        'cbam2_y': 9.5,
        'ssh_y': 7,
        'heads_y': 4,
        
        # Right side - output (better positioned)
        'output': {'x': 21, 'y': 1.5, 'w': dimensions['output']['w'], 'h': dimensions['output']['h']},
        
        # Bottom left - teacher model (better positioned)
        'teacher': {'x': 1, 'y': 0.5, 'w': dimensions['teacher']['w'], 'h': dimensions['teacher']['h']},
        
        # Right side - pruning annotations (repositioned)
        'pruning_x': 25.5,
        'pruning_start_y': 13.5
    }
    
    # Simplified title - less cluttered
    ax.text(14, 17.2, 'FeatherFace Nano-B: Ultra-Lightweight Face Detection',
            ha='center', va='center', fontsize=fonts['title_main']['size'], 
            fontweight=fonts['title_main']['weight'], color=colors['text_main'])
    ax.text(14, 16.7, '120K-180K Parameters | Bayesian Pruning + Knowledge Distillation',
            ha='center', va='center', fontsize=fonts['title_section']['size'], 
            fontweight=fonts['title_section']['weight'], color=colors['text_secondary'])
    
    # Simplified level labels - cleaner
    level_positions = [positions['p3_x'], positions['p4_x'], positions['p5_x']]
    level_labels = ['P3\n40×40', 'P4\n20×20', 'P5\n10×10']
    
    for x_pos, label in zip(level_positions, level_labels):
        ax.text(x_pos + dimensions['level_module']['w']/2, 16.2, label, 
                ha='center', va='center', fontsize=fonts['text_normal']['size'], 
                fontweight=fonts['text_normal']['weight'], color=colors['text_main'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['accent'], 
                         edgecolor=colors['component_border'], linewidth=1))
    
    # === LEFT SIDE: INPUT AND BACKBONE ===
    
    # 1. Input Layer with improved styling
    input_pos = positions['input']
    input_box = FancyBboxPatch(
        (input_pos['x'], input_pos['y']), input_pos['w'], input_pos['h'],
        boxstyle="round,pad=0.1", 
        facecolor=colors['component_fill'], 
        edgecolor=colors['component_border'], 
        linewidth=borders['important']
    )
    ax.add_patch(input_box)
    ax.text(input_pos['x'] + input_pos['w']/2, input_pos['y'] + input_pos['h']/2, 
            'INPUT\n640×640', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    
    # 2. MobileNet Backbone with critical importance styling
    backbone_pos = positions['backbone']
    backbone_box = FancyBboxPatch(
        (backbone_pos['x'], backbone_pos['y']), backbone_pos['w'], backbone_pos['h'],
        boxstyle="round,pad=0.1",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=borders['critical']  # Thickest border for critical component
    )
    ax.add_patch(backbone_box)
    ax.text(backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y'] + backbone_pos['h'] - 0.3, 
            'MOBILENET V1-0.25', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    ax.text(backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y'] + backbone_pos['h']/2, 
            '60K params', 
            ha='center', va='center', fontsize=fonts['text_detail']['size'], 
            fontweight=fonts['text_detail']['weight'], color=colors['text_secondary'])
    ax.text(backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y'] + 0.3, 
            'Bayesian Pruning', 
            ha='center', va='center', fontsize=fonts['text_annotation']['size'], 
            style='italic', color=colors['text_detail'])
    
    # Simplified backbone stages
    stage_data = [
        ('32ch', positions['p3_x']),
        ('64ch', positions['p4_x']),
        ('128ch', positions['p5_x'])
    ]
    
    stage_y = backbone_pos['y'] - 0.5
    for stage_text, x_pos in stage_data:
        # Stage box with improved styling
        stage_box = FancyBboxPatch(
            (x_pos + dimensions['level_module']['w']/2 - 0.8, stage_y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=colors['highlight'],
            edgecolor=colors['component_border'],
            linewidth=borders['auxiliary']
        )
        ax.add_patch(stage_box)
        ax.text(x_pos + dimensions['level_module']['w']/2, stage_y, stage_text, 
                ha='center', va='center', fontsize=fonts['text_detail']['size'], 
                fontweight=fonts['text_detail']['weight'], color=colors['text_main'])
        
        # Connection from backbone to stage with improved styling
        connection = patches.ConnectionPatch(
            (backbone_pos['x'] + backbone_pos['w'], backbone_pos['y'] + backbone_pos['h']/2),
            (x_pos + dimensions['level_module']['w']/2, stage_y + 0.3),
            "data", "data", arrowstyle="->", shrinkA=0, shrinkB=0,
            color=colors['arrow_main'], linewidth=1.5
        )
        ax.add_patch(connection)
    
    # === CENTER: MULTI-LEVEL PROCESSING ===
    
    def create_level_module(x, y, width, height, title, details, module_type='standard', 
                           params='', flops=''):
        """Create a module box for each level with enhanced styling and information"""
        # Determine styling based on module type with enhanced hierarchy
        if module_type == 'critical':
            border_width = borders['critical']
            fill_color = colors['critical_fill']
        elif module_type == 'important':
            border_width = borders['important']
            fill_color = colors['component_fill']
        elif module_type == 'cbam':
            border_width = borders['important']
            fill_color = colors['secondary_fill']
        elif module_type == 'bifpn':
            border_width = borders['critical']
            fill_color = colors['highlight']
        else:
            border_width = borders['standard']
            fill_color = colors['secondary_fill']
        
        # Main box
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=fill_color,
            edgecolor=colors['component_border'],
            linewidth=border_width
        )
        ax.add_patch(box)
        
        # Title with proper typography - simplified
        ax.text(x + width/2, y + height - 0.3, title, 
                ha='center', va='center', fontsize=fonts['title_module']['size'], 
                fontweight=fonts['title_module']['weight'], color=colors['text_main'])
        
        # Main details - concise
        ax.text(x + width/2, y + height/2 + 0.1, details, 
                ha='center', va='center', fontsize=fonts['text_annotation']['size'], 
                fontweight=fonts['text_annotation']['weight'], color=colors['text_secondary'])
        
        # Simplified parameters annotation
        if params:
            ax.text(x + width/2, y + 0.25, params, 
                    ha='center', va='bottom', fontsize=8, 
                    color=colors['text_detail'], weight='bold')
        
        # Enhanced pattern for differentiation
        if module_type == 'cbam':
            # Add attention pattern with stronger visibility
            for i in range(3):
                line_y = y + 0.4 + i * 0.4
                ax.plot([x + 0.4, x + width - 0.4], [line_y, line_y], 
                       color=colors['text_detail'], linewidth=1, linestyle='--', alpha=0.7)
        elif module_type == 'bifpn':
            # Add bidirectional arrows pattern
            mid_y = y + height/2
            ax.annotate('', xy=(x + width - 0.3, mid_y + 0.2), xytext=(x + 0.3, mid_y + 0.2),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow_bifpn'], lw=1.5))
            ax.annotate('', xy=(x + 0.3, mid_y - 0.2), xytext=(x + width - 0.3, mid_y - 0.2),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow_bifpn'], lw=1.5))
        
        return box
    
    # Create modules for each level (P3, P4, P5) with improved styling
    levels_x = [positions['p3_x'], positions['p4_x'], positions['p5_x']]
    
    # 1. Efficient CBAM (Pre-BiFPN) - Simplified
    cbam1_y = positions['cbam1_y']
    cbam_params = ['2.4K', '4.8K', '9.6K']
    
    for i, x in enumerate(levels_x):
        create_level_module(x, cbam1_y, dimensions['level_module']['w'], dimensions['level_module']['h'], 
                          'CBAM', 
                          'Attention\nR=8', 
                          module_type='cbam',
                          params=cbam_params[i])
    
    # Simplified section label
    ax.text(2, cbam1_y + 0.9, 'PRE\nATTN', ha='center', va='center', 
            fontsize=fonts['text_normal']['size'], fontweight=fonts['text_normal']['weight'], 
            color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # 2. BiFPN + MSE Standard
    bifpn_y = positions['bifpn_y']
    bifpn_params = ['16.8K', '20.4K', '27.6K']
    
    for i, x in enumerate(levels_x):
        create_level_module(x, bifpn_y, dimensions['level_module']['w'], dimensions['level_module']['h'], 
                          'BiFPN+MSE', 
                          'MSE-FPN\n72ch', 
                          module_type='bifpn',
                          params=bifpn_params[i])
    
    # Simplified BiFPN arrows - cleaner and contained
    arrow_y = bifpn_y + 1.0
    
    # Top-down arrows (P5→P4→P3) - simplified
    for i in range(len(levels_x) - 1):
        arrow = patches.FancyArrowPatch(
            (levels_x[i+1] + dimensions['level_module']['w']/2, arrow_y + 0.2),
            (levels_x[i] + dimensions['level_module']['w']/2, arrow_y + 0.2),
            arrowstyle='->', mutation_scale=20, color=colors['arrow_bifpn'], linewidth=2
        )
        ax.add_patch(arrow)
    
    # Bottom-up arrows (P3→P4→P5) - simplified
    for i in range(len(levels_x) - 1):
        arrow = patches.FancyArrowPatch(
            (levels_x[i] + dimensions['level_module']['w']/2, arrow_y - 0.2),
            (levels_x[i+1] + dimensions['level_module']['w']/2, arrow_y - 0.2),
            arrowstyle='->', mutation_scale=20, color=colors['arrow_bifpn'], linewidth=2
        )
        ax.add_patch(arrow)
    
    # Single central label for BiFPN
    ax.text(levels_x[1] + dimensions['level_module']['w']/2, arrow_y, 
            'BiFPN', ha='center', va='center', fontsize=fonts['text_normal']['size'], 
            fontweight='bold', color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.15", facecolor=colors['background'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    ax.text(2, bifpn_y + 0.9, 'FPN', ha='center', va='center', 
            fontsize=fonts['text_normal']['size'], fontweight=fonts['text_normal']['weight'], 
            color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # 3. Simplified CBAM (Post-BiFPN)
    cbam2_y = positions['cbam2_y']
    for i, x in enumerate(levels_x):
        create_level_module(x, cbam2_y, dimensions['level_module']['w'], dimensions['level_module']['h'], 
                          'CBAM', 
                          'Attention\nR=8', 
                          module_type='cbam',
                          params=cbam_params[i])
    
    ax.text(2, cbam2_y + 0.9, 'POST\nATTN', ha='center', va='center', 
            fontsize=fonts['text_normal']['size'], fontweight=fonts['text_normal']['weight'], 
            color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # 4. Simplified SSH
    ssh_y = positions['ssh_y']
    ssh_params = ['8.4K', '10.8K', '13.2K']
    
    for i, x in enumerate(levels_x):
        create_level_module(x, ssh_y, dimensions['level_module']['w'], dimensions['level_module']['h'], 
                          'SSH', 
                          'Groups=2\nContext', 
                          module_type='important',
                          params=ssh_params[i])
    
    ax.text(2, ssh_y + 0.9, 'SSH', ha='center', va='center', 
            fontsize=fonts['text_normal']['size'], fontweight=fonts['text_normal']['weight'], 
            color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # 5. Simplified Channel Shuffle
    shuffle_y = ssh_y - 1.2
    shuffle_box = FancyBboxPatch(
        (levels_x[0], shuffle_y), levels_x[-1] + dimensions['level_module']['w'] - levels_x[0], 0.6,
        boxstyle="round,pad=0.1",
        facecolor=colors['highlight'],
        edgecolor=colors['component_border'],
        linewidth=1
    )
    ax.add_patch(shuffle_box)
    ax.text((levels_x[0] + levels_x[-1] + dimensions['level_module']['w'])/2, shuffle_y + 0.3, 
            'CHANNEL SHUFFLE', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    
    # === RIGHT SIDE: DETECTION HEADS AND OUTPUT ===
    
    # Simplified detection heads
    heads_y = positions['heads_y']
    head_types = ['Class', 'BBox', 'Landmark']
    
    for i, x in enumerate(levels_x):
        for j, head_type in enumerate(head_types):
            head_box = FancyBboxPatch(
                (x, heads_y + j * 0.5), 
                dimensions['head_box']['w'], 0.45,
                boxstyle="round,pad=0.05",
                facecolor=colors['component_fill'],
                edgecolor=colors['component_border'],
                linewidth=1
            )
            ax.add_patch(head_box)
            ax.text(x + dimensions['head_box']['w']/2, heads_y + j * 0.5 + 0.225, 
                    head_type, 
                    ha='center', va='center', fontsize=fonts['text_annotation']['size'], 
                    fontweight=fonts['text_annotation']['weight'], color=colors['text_main'])
    
    ax.text(2, heads_y + 0.75, 'HEADS', ha='center', va='center', 
            fontsize=fonts['text_normal']['size'], fontweight=fonts['text_normal']['weight'], 
            color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # Final output with improved styling
    output_pos = positions['output']
    output_box = FancyBboxPatch(
        (output_pos['x'], output_pos['y']), output_pos['w'], output_pos['h'],
        boxstyle="round,pad=0.1",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=borders['important']
    )
    ax.add_patch(output_box)
    ax.text(output_pos['x'] + output_pos['w']/2, output_pos['y'] + output_pos['h']/2, 
            'OUTPUT\nFaces + BBoxes + Landmarks', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    
    # === TEACHER MODEL AND DISTILLATION ===
    
    teacher_pos = positions['teacher']
    teacher_box = FancyBboxPatch(
        (teacher_pos['x'], teacher_pos['y']), teacher_pos['w'], teacher_pos['h'],
        boxstyle="round,pad=0.2",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=borders['critical']
    )
    ax.add_patch(teacher_box)
    ax.text(teacher_pos['x'] + teacher_pos['w']/2, teacher_pos['y'] + teacher_pos['h'] - 0.5, 
            'TEACHER V1', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    ax.text(teacher_pos['x'] + teacher_pos['w']/2, teacher_pos['y'] + teacher_pos['h']/2, 
            '487K params\nKnowledge\nDistillation', 
            ha='center', va='center', fontsize=fonts['text_detail']['size'], 
            fontweight=fonts['text_detail']['weight'], color=colors['text_secondary'])
    
    # Distillation arrows to key components with improved styling
    distill_targets = [
        (backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y']),
        (levels_x[1] + dimensions['level_module']['w']/2, bifpn_y),
        (levels_x[1] + dimensions['level_module']['w']/2, heads_y + 1)
    ]
    
    for target_x, target_y in distill_targets:
        kd_arrow = patches.ConnectionPatch(
            (teacher_pos['x'] + teacher_pos['w'], teacher_pos['y'] + teacher_pos['h']/2),
            (target_x, target_y),
            "data", "data", arrowstyle="->", shrinkA=0, shrinkB=0,
            color=colors['arrow_distill'], linewidth=2, linestyle='dashed'
        )
        ax.add_patch(kd_arrow)
    
    # Simplified pruning rates
    pruning_x = positions['pruning_x']
    pruning_rates = [
        (backbone_pos['y'] + backbone_pos['h']/2, '20%'),
        (bifpn_y + 0.9, '30%'), 
        (ssh_y + 0.9, '15%'),
        (heads_y + 0.5, '10%')
    ]
    
    ax.text(pruning_x, positions['pruning_start_y'], 'PRUNING', 
            ha='center', va='center', fontsize=fonts['text_normal']['size'], 
            fontweight=fonts['text_normal']['weight'], color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    for y_pos, rate in pruning_rates:
        # Simplified rate annotation
        rate_box = FancyBboxPatch(
            (pruning_x - 0.5, y_pos - 0.2), 1, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=colors['component_fill'],
            edgecolor=colors['component_border'],
            linewidth=1
        )
        ax.add_patch(rate_box)
        ax.text(pruning_x, y_pos, rate, 
                ha='center', va='center', fontsize=fonts['text_annotation']['size'], 
                fontweight='bold', color=colors['text_main'])
    
    # Simplified scientific foundation
    foundation_text = "B-FPGM Pruning | Knowledge Distillation | CBAM | BiFPN | MobileNet"
    
    ax.text(14, 0.8, foundation_text, ha='center', va='center', 
            fontsize=fonts['text_detail']['size'], fontweight=fonts['text_detail']['weight'], 
            color=colors['text_main'], 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], 
                     edgecolor=colors['component_border'], linewidth=1))
    
    # Simplified vertical flow arrows - contained within bounds
    for x in levels_x:
        arrow_positions = [
            (stage_y - 0.2, cbam1_y + dimensions['level_module']['h']),
            (cbam1_y, bifpn_y + dimensions['level_module']['h']),
            (bifpn_y, cbam2_y + dimensions['level_module']['h']),
            (cbam2_y, ssh_y + dimensions['level_module']['h']),
            (ssh_y, heads_y + 1.3)
        ]
        
        for start_y, end_y in arrow_positions:
            arrow = patches.FancyArrowPatch(
                (x + dimensions['level_module']['w']/2, start_y), 
                (x + dimensions['level_module']['w']/2, end_y),
                arrowstyle='->', mutation_scale=15, color=colors['arrow_main'], linewidth=1.5
            )
            ax.add_patch(arrow)
    
    # Simplified convergence arrows - cleaner paths
    for i, x in enumerate(levels_x):
        conv_arrow = patches.FancyArrowPatch(
            (x + dimensions['level_module']['w']/2, heads_y - 0.2), 
            (output_pos['x'] + output_pos['w']/2, output_pos['y'] + output_pos['h']),
            arrowstyle='->', mutation_scale=15, color=colors['arrow_main'], linewidth=1.5
        )
        ax.add_patch(conv_arrow)
    
    # Set axis limits for optimized landscape layout
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 18)
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