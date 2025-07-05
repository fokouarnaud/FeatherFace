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
    
    # Create figure with landscape dimensions for multi-level representation
    fig_width = 24
    fig_height = 16  # Landscape layout for P3/P4/P5 parallel representation
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Enhanced black and white styling with visual hierarchy
    colors = {
        'main_border': '#000000',  # Pure black for main elements
        'component_fill': '#FFFFFF',  # Pure white for main components
        'component_border': '#000000',  # Black borders
        'secondary_fill': '#F8F8F8',  # Very light gray for secondary
        'text_main': '#000000',  # Pure black for main text
        'text_secondary': '#333333',  # Dark gray for secondary text
        'text_detail': '#555555',  # Medium gray for details
        'arrow_main': '#000000',  # Black for main data flow
        'arrow_distill': '#666666',  # Gray for distillation
        'background': '#FFFFFF',  # Pure white background
        'highlight': '#E8E8E8',  # Light gray for highlights
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
    # Grid system: 24x16 with 0.5 unit precision
    grid_unit = 0.5
    
    # Component dimensions optimized for hierarchy
    dimensions = {
        'input': {'w': 5.5, 'h': 1.4},
        'backbone': {'w': 5.5, 'h': 1.6},  # Slightly larger for importance
        'level_module': {'w': 3.8, 'h': 1.8},
        'teacher': {'w': 6, 'h': 3.2},  # Reduced size for better balance
        'output': {'w': 5.5, 'h': 1.4},
        'head_box': {'w': 3.8, 'h': 0.55}
    }
    
    # Grid-aligned positions for perfect layout
    positions = {
        # Left side - input flow (perfectly aligned)
        'input': {'x': 0.5, 'y': 13.5, 'w': dimensions['input']['w'], 'h': dimensions['input']['h']},
        'backbone': {'x': 0.5, 'y': 11, 'w': dimensions['backbone']['w'], 'h': dimensions['backbone']['h']},
        
        # Center - multi-level processing (perfectly spaced)
        'p3_x': 8.5, 'p4_x': 13, 'p5_x': 17.5,  # Evenly spaced at 4.5 unit intervals
        'level_spacing': 4.5,  # Consistent spacing
        
        # Y positions for processing layers (evenly distributed)
        'cbam1_y': 13.2,
        'bifpn_y': 10.8,
        'cbam2_y': 8.4,
        'ssh_y': 6.0,
        'heads_y': 3.5,
        
        # Right side - output 
        'output': {'x': 18, 'y': 1.2, 'w': dimensions['output']['w'], 'h': dimensions['output']['h']},
        
        # Bottom left - teacher model (repositioned)
        'teacher': {'x': 0.5, 'y': 0.5, 'w': dimensions['teacher']['w'], 'h': dimensions['teacher']['h']},
        
        # Right side - pruning annotations
        'pruning_x': 22.5,
        'pruning_start_y': 12
    }
    
    # Main title with improved typography
    ax.text(12, 15.5, 'FeatherFace Nano-B: Bayesian-Optimized Ultra-Lightweight Face Detection',
            ha='center', va='center', fontsize=fonts['title_main']['size'], 
            fontweight=fonts['title_main']['weight'], color=colors['text_main'])
    ax.text(12, 15, '120,000-180,000 Parameters (48-65% Reduction) | 7 Scientific Techniques',
            ha='center', va='center', fontsize=fonts['title_section']['size'], 
            fontweight=fonts['title_section']['weight'], color=colors['text_secondary'])
    
    # Level labels with consistent spacing and improved typography
    level_positions = [positions['p3_x'], positions['p4_x'], positions['p5_x']]
    level_labels = [
        'P3 Level\n(32→72 channels)\n320×320→40×40',
        'P4 Level\n(64→72 channels)\n160×160→20×20', 
        'P5 Level\n(128→72 channels)\n80×80→10×10'
    ]
    
    for x_pos, label in zip(level_positions, level_labels):
        ax.text(x_pos + dimensions['level_module']['w']/2, 14.8, label, 
                ha='center', va='center', fontsize=fonts['text_normal']['size'], 
                fontweight=fonts['text_normal']['weight'], color=colors['text_main'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['secondary_fill'], 
                         edgecolor=colors['component_border'], linewidth=borders['auxiliary']))
    
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
            'INPUT IMAGE\n(640×640×3)', 
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
            'MOBILENET V1-0.25 BACKBONE', 
            ha='center', va='center', fontsize=fonts['title_module']['size'], 
            fontweight=fonts['title_module']['weight'], color=colors['text_main'])
    ax.text(backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y'] + backbone_pos['h']/2, 
            '~60,000 parameters (40% of total)', 
            ha='center', va='center', fontsize=fonts['text_detail']['size'], 
            fontweight=fonts['text_detail']['weight'], color=colors['text_secondary'])
    ax.text(backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y'] + 0.3, 
            'Bayesian-Optimized Pruning Applied', 
            ha='center', va='center', fontsize=fonts['text_annotation']['size'], 
            style='italic', color=colors['text_detail'])
    
    # Show backbone stages with connections to levels
    stage_data = [
        ('32ch\n320×320', positions['p3_x']),
        ('64ch\n160×160', positions['p4_x']),
        ('128ch\n80×80', positions['p5_x'])
    ]
    
    stage_y = backbone_pos['y'] - 0.5
    for stage_text, x_pos in stage_data:
        # Stage box
        stage_box = FancyBboxPatch(
            (x_pos + level_width/2 - 0.8, stage_y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=colors['highlight'],
            edgecolor=colors['component_border'],
            linewidth=1
        )
        ax.add_patch(stage_box)
        ax.text(x_pos + level_width/2, stage_y, stage_text, 
                ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text_main'])
        
        # Connection from backbone to stage
        connection = patches.ConnectionPatch(
            (backbone_pos['x'] + backbone_pos['w'], backbone_pos['y'] + backbone_pos['h']/2),
            (x_pos + level_width/2, stage_y + 0.3),
            "data", "data", arrowstyle="->", shrinkA=0, shrinkB=0,
            color=colors['arrow'], linewidth=1.5
        )
        ax.add_patch(connection)
    
    # === CENTER: MULTI-LEVEL PROCESSING ===
    
    def create_level_module(x, y, width, height, title, details, pattern=''):
        """Create a module box for each level"""
        # Main box
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=colors['component_fill'],
            edgecolor=colors['component_border'],
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Title
        ax.text(x + width/2, y + height - 0.3, title, 
                ha='center', va='center', fontweight='bold', fontsize=10, color=colors['text_main'])
        
        # Details
        ax.text(x + width/2, y + height/2 - 0.1, details, 
                ha='center', va='center', fontsize=9, color=colors['text_main'])
        
        # Pattern for differentiation (using different line styles)
        if pattern == 'dashed':
            for i in range(3):
                line_y = y + 0.3 + i * 0.3
                ax.plot([x + 0.2, x + width - 0.2], [line_y, line_y], 
                       color=colors['component_border'], linewidth=1, linestyle='--')
        elif pattern == 'dotted':
            for i in range(3):
                line_y = y + 0.3 + i * 0.3
                ax.plot([x + 0.2, x + width - 0.2], [line_y, line_y], 
                       color=colors['component_border'], linewidth=1, linestyle=':')
        
        return box
    
    # Create modules for each level (P3, P4, P5)
    levels_x = [positions['p3_x'], positions['p4_x'], positions['p5_x']]
    
    # 1. Efficient CBAM (Pre-BiFPN)
    cbam1_y = positions['cbam1_y']
    for i, x in enumerate(levels_x):
        create_level_module(x, cbam1_y, level_width, 1.8, 
                          'EFFICIENT CBAM', 
                          'Channel + Spatial\nReduction: 8\n~2.7K params', 
                          'dashed')
    
    ax.text(2, cbam1_y + 0.9, 'PRE-BIFPN\nATTENTION', ha='center', va='center', 
            fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # 2. Efficient BiFPN with bidirectional connections
    bifpn_y = positions['bifpn_y']
    for i, x in enumerate(levels_x):
        create_level_module(x, bifpn_y, level_width, 1.8, 
                          'EFFICIENT BIFPN', 
                          'DWSConv\n72 channels\n~15K params', 
                          'dotted')
    
    # BiFPN bidirectional arrows
    arrow_y = bifpn_y + 0.9
    # Top-down arrows (P5→P4→P3)
    for i in range(len(levels_x) - 1):
        arrow = patches.FancyArrowPatch(
            (levels_x[i+1] + level_width/2, arrow_y + 0.2),
            (levels_x[i] + level_width/2, arrow_y + 0.2),
            arrowstyle='->', mutation_scale=15, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow)
        ax.text((levels_x[i] + levels_x[i+1] + level_width)/2, arrow_y + 0.4, 'TD', 
                ha='center', va='center', fontsize=8, color=colors['text_main'])
    
    # Bottom-up arrows (P3→P4→P5)
    for i in range(len(levels_x) - 1):
        arrow = patches.FancyArrowPatch(
            (levels_x[i] + level_width/2, arrow_y - 0.2),
            (levels_x[i+1] + level_width/2, arrow_y - 0.2),
            arrowstyle='->', mutation_scale=15, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow)
        ax.text((levels_x[i] + levels_x[i+1] + level_width)/2, arrow_y - 0.4, 'BU', 
                ha='center', va='center', fontsize=8, color=colors['text_main'])
    
    ax.text(2, bifpn_y + 0.9, 'BIFPN\nFEATURE\nPYRAMID', ha='center', va='center', 
            fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # 3. Efficient CBAM (Post-BiFPN)
    cbam2_y = positions['cbam2_y']
    for i, x in enumerate(levels_x):
        create_level_module(x, cbam2_y, level_width, 1.8, 
                          'EFFICIENT CBAM', 
                          'Same as Pre\n~2.7K params\nPost-processing', 
                          'dashed')
    
    ax.text(2, cbam2_y + 0.9, 'POST-BIFPN\nATTENTION', ha='center', va='center', 
            fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # 4. Grouped SSH Context Module
    ssh_y = positions['ssh_y']
    for i, x in enumerate(levels_x):
        create_level_module(x, ssh_y, level_width, 2.2, 
                          'GROUPED SSH', 
                          'Groups: 2\nMulti-scale\nContext Agg\n~11.7K params')
    
    ax.text(2, ssh_y + 1.1, 'SSH\nCONTEXT\nMODULE', ha='center', va='center', 
            fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # 5. Channel Shuffle (spans all levels)
    shuffle_y = ssh_y - 1.5
    shuffle_box = FancyBboxPatch(
        (levels_x[0], shuffle_y), levels_x[-1] + level_width - levels_x[0], 0.8,
        boxstyle="round,pad=0.1",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=2
    )
    ax.add_patch(shuffle_box)
    ax.text((levels_x[0] + levels_x[-1] + level_width)/2, shuffle_y + 0.4, 
            'CHANNEL SHUFFLE (0 params) - Groups = 2', 
            ha='center', va='center', fontweight='bold', fontsize=11, color=colors['text_main'])
    
    # === RIGHT SIDE: DETECTION HEADS AND OUTPUT ===
    
    # Detection heads for each level
    heads_y = 3
    head_types = ['Classification', 'Regression', 'Landmarks']
    head_outputs = ['72→2', '72→4', '72→10']
    
    for i, x in enumerate(levels_x):
        for j, (head_type, output) in enumerate(zip(head_types, head_outputs)):
            head_box = FancyBboxPatch(
                (x, heads_y + j * 0.7), level_width, 0.6,
                boxstyle="round,pad=0.05",
                facecolor=colors['component_fill'],
                edgecolor=colors['component_border'],
                linewidth=1
            )
            ax.add_patch(head_box)
            ax.text(x + level_width/2, heads_y + j * 0.7 + 0.3, 
                    f'{head_type}\n{output}', 
                    ha='center', va='center', fontsize=9, color=colors['text_main'])
    
    ax.text(2, heads_y + 1, 'DETECTION\nHEADS\n(~15K params)', ha='center', va='center', 
            fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # Final output
    output_pos = positions['output']
    output_box = FancyBboxPatch(
        (output_pos['x'], output_pos['y']), output_pos['w'], output_pos['h'],
        boxstyle="round,pad=0.1",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(output_pos['x'] + output_pos['w']/2, output_pos['y'] + output_pos['h']/2, 
            'OUTPUT\nFace Classifications • BBox Regressions • Landmarks', 
            ha='center', va='center', fontweight='bold', fontsize=11, color=colors['text_main'])
    
    # === TEACHER MODEL AND DISTILLATION ===
    
    teacher_pos = positions['teacher']
    teacher_box = FancyBboxPatch(
        (teacher_pos['x'], teacher_pos['y']), teacher_pos['w'], teacher_pos['h'],
        boxstyle="round,pad=0.2",
        facecolor=colors['component_fill'],
        edgecolor=colors['component_border'],
        linewidth=2
    )
    ax.add_patch(teacher_box)
    ax.text(teacher_pos['x'] + teacher_pos['w']/2, teacher_pos['y'] + teacher_pos['h'] - 0.5, 
            'TEACHER MODEL (V1)', 
            ha='center', va='center', fontweight='bold', fontsize=12, color=colors['text_main'])
    ax.text(teacher_pos['x'] + teacher_pos['w']/2, teacher_pos['y'] + teacher_pos['h']/2, 
            '487,103 parameters\nTemperature: 4.0\nAlpha: 0.7\nWeighted Knowledge\nDistillation', 
            ha='center', va='center', fontsize=10, color=colors['text_main'])
    
    # Distillation arrows to key components
    distill_targets = [
        (backbone_pos['x'] + backbone_pos['w']/2, backbone_pos['y']),
        (levels_x[1] + level_width/2, bifpn_y),
        (levels_x[1] + level_width/2, heads_y + 1)
    ]
    
    for target_x, target_y in distill_targets:
        kd_arrow = patches.ConnectionPatch(
            (teacher_pos['x'] + teacher_pos['w'], teacher_pos['y'] + teacher_pos['h']/2),
            (target_x, target_y),
            "data", "data", arrowstyle="->", shrinkA=0, shrinkB=0,
            color=colors['arrow'], linewidth=2, linestyle='dashed'
        )
        ax.add_patch(kd_arrow)
    
    # B-FPGM Pruning rates (right side)
    pruning_x = 23
    pruning_rates = [
        (backbone_pos['y'] + backbone_pos['h']/2, '15-25%', 'Backbone'),
        (cbam1_y + 0.9, '20-30%', 'CBAM'),
        (bifpn_y + 0.9, '25-35%', 'BiFPN'), 
        (ssh_y + 1.1, '10-20%', 'SSH'),
        (heads_y + 1, '5-15%', 'Heads')
    ]
    
    ax.text(pruning_x, 12, 'B-FPGM\nBAYESIAN\nPRUNING\nRATES', 
            ha='center', va='center', fontweight='bold', fontsize=11, color=colors['text_main'],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    for y_pos, rate, component in pruning_rates:
        # Rate annotation
        rate_box = FancyBboxPatch(
            (pruning_x - 0.8, y_pos - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=colors['component_fill'],
            edgecolor=colors['component_border'],
            linewidth=1
        )
        ax.add_patch(rate_box)
        ax.text(pruning_x, y_pos, f'{rate}\n{component}', 
                ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text_main'])
    
    # Scientific foundation
    foundation_text = "Scientific Foundation (7 Techniques):\n1. B-FPGM: Kaparinos & Mezaris, WACVW 2025\n2. Knowledge Distillation: Li et al. CVPR 2023\n3. CBAM: Woo et al. ECCV 2018\n4. BiFPN: Tan et al. CVPR 2020\n5. MobileNet: Howard et al. 2017\n6. Bayesian Optimization: Mockus, 1989\n7. Channel Shuffle: Parameter-free optimization"
    
    ax.text(12, 0.5, foundation_text, ha='center', va='center', fontsize=10, 
            color=colors['text_main'], 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], edgecolor=colors['component_border']))
    
    # Add vertical flow arrows connecting levels
    for x in levels_x:
        # Input to CBAM1
        arrow1 = patches.FancyArrowPatch(
            (x + level_width/2, stage_y - 0.3), (x + level_width/2, cbam1_y + 1.8),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow1)
        
        # CBAM1 to BiFPN
        arrow2 = patches.FancyArrowPatch(
            (x + level_width/2, cbam1_y), (x + level_width/2, bifpn_y + 1.8),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow2)
        
        # BiFPN to CBAM2
        arrow3 = patches.FancyArrowPatch(
            (x + level_width/2, bifpn_y), (x + level_width/2, cbam2_y + 1.8),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow3)
        
        # CBAM2 to SSH
        arrow4 = patches.FancyArrowPatch(
            (x + level_width/2, cbam2_y), (x + level_width/2, ssh_y + 2.2),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow4)
        
        # SSH to Heads
        arrow5 = patches.FancyArrowPatch(
            (x + level_width/2, ssh_y), (x + level_width/2, heads_y + 2.1),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow5)
    
    # Heads to output (convergence)
    for i, x in enumerate(levels_x):
        conv_arrow = patches.FancyArrowPatch(
            (x + level_width/2, heads_y), 
            (output_pos['x'] + output_pos['w']/2, output_pos['y'] + output_pos['h']),
            arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2
        )
        ax.add_patch(conv_arrow)
    
    # Set axis limits for landscape layout
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
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