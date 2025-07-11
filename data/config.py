# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 350,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 56,  # CALIBRATION: SSH architecture (divisible by 4) → targeting ~494K parameters
    'lr' : 1e-3,
    'optim' : 'adamw'
}


# Configuration for FeatherFace V2 (V1 + Coordinate Attention)
# INNOVATION: Replace CBAM with Coordinate Attention for mobile-optimized face detection
# Scientific foundation: Hou et al. CVPR 2021 + simplified training approach
cfg_v2 = {
    # Base configuration IDENTICAL to V1 for controlled comparison
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 350,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 56,  # Identical to V1 for direct comparison
    'lr': 1e-3,
    'optim': 'adamw',
    
    # V2 Innovation: Coordinate Attention Configuration
    'attention_mechanism': 'coordinate_attention',  # NEW: Replace CBAM
    'coordinate_attention_config': {
        'reduction_ratio': 32,          # Mobile optimization (vs 16 in CBAM)
        'mobile_optimized': True,       # Enable mobile-specific optimizations
        'preserve_spatial': True,       # Key advantage vs CBAM
        'use_depthwise': False,         # Standard conv for stability
    },
    
    # Performance targets based on research (simplified training)
    'performance_targets': {
        'widerface_easy': 0.9200,      # Maintain V1 level
        'widerface_medium': 0.9000,    # Maintain V1 level
        'widerface_hard': 0.8500,      # +8-10% vs V1 (realistic with simple training)
        'mobile_speedup': 2.0,         # 2x faster than CBAM
        'parameter_budget': 500000,    # ~493K parameters (4K more than V1)
    },
    
    # Scientific validation references
    'scientific_basis': {
        'coordinate_attention': 'Hou et al. CVPR 2021',
        'training_methodology': 'Simplified direct training (like V1)',
        'architectural_innovation': 'Post-SSH coordinate attention',
        'controlled_experiment': 'Single variable change (attention mechanism)'
    },
    
    # Experimental configuration
    'experiment_config': {
        'baseline_model': 'v1',         # Comparison baseline
        'training_approach': 'direct_supervision',  # No knowledge distillation
        'controlled_variables': ['attention_mechanism'],  # Only change attention
        'evaluation_metrics': ['widerface_ap', 'inference_time', 'parameters'],
        'validation_protocol': 'widerface_standard',
    }
}

# Note: cfg_mnet (V1 baseline) and cfg_v2 (V2 innovation) configurations supported

