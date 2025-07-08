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
# Scientific foundation: Hou et al. CVPR 2021 + 2024-2025 applications
cfg_v2 = {
    # Base configuration identical to V1 for controlled comparison
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
    
    # Training configuration optimized for V2
    'knowledge_distillation': {
        'enabled': True,                # V1 → V2 knowledge transfer
        'temperature': 4.0,             # Standard distillation temperature
        'alpha': 0.7,                   # Distillation vs task loss weight
        'teacher_model': 'v1',          # Use V1 as teacher
    },
    
    # Performance targets based on research
    'performance_targets': {
        'widerface_easy': 0.9300,      # +0.43% vs V1 (maintain)
        'widerface_medium': 0.9150,    # +1.26% vs V1 (slight improvement)
        'widerface_hard': 0.8800,      # +10.85% vs V1 (major improvement)
        'mobile_speedup': 2.0,         # 2x faster than CBAM
        'parameter_budget': 500000,    # Maintain ~489K parameters
    },
    
    # Scientific validation references
    'scientific_basis': {
        'coordinate_attention': 'Hou et al. CVPR 2021',
        'mobile_applications': 'EfficientFace 2024, FasterMLP 2025',
        'face_detection': 'Dense Face Detection 2024',
        'methodology': 'FeatherFace V2 Methodology 2025'
    },
    
    # Experimental configuration
    'experiment_config': {
        'baseline_model': 'v1',         # Comparison baseline
        'controlled_variables': ['attention_mechanism'],  # Only change attention
        'evaluation_metrics': ['widerface_ap', 'inference_time', 'parameters'],
        'validation_protocol': 'widerface_standard',
    }
}

# Note: cfg_mnet (V1 baseline) and cfg_v2 (V2 innovation) configurations supported

