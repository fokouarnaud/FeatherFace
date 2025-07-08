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


# Configuration for FeatherFace Nano-B (V1 Clone + Bayesian Pruning)
# STRATEGY: Start IDENTICAL to V1, then let Bayesian pruning decide what to cut intelligently
# This preserves V1's proven architecture optimizations and ensures 100% compatibility
cfg_nano_b = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,        # Optimized for H100 80GB HBM3
    'ngpu': 1,
    'epoch': 300,  # Optimized for 3-phase training pipeline
    'decay1': 150,
    'decay2': 250,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,            # MobileNet backbone input channels (IDENTICAL to V1)
    'out_channel': 56,           # SSH detection head channels (IDENTICAL to V1 for 100% compatibility)
    'num_classes': 2,            # Binary classification (background/face) - CORRECT for face detection
                                 # Note: bbox_regression=4 coords, landmarks=10 coords, but classification=2 classes
    
    # Learning configuration (OPTIMIZED for H100 training)
    'lr': 1e-4,              # Proper learning rate for effective gradient flow
    'optim': 'adamw',
    'weight_decay': 5e-4,
    
    # Nano-B specific efficiency techniques (scientifically validated)
    'cbam_reduction': 8,              # CBAM attention bottleneck ratio (Woo et al. ECCV 2018)
    'ssh_groups': 2,                  # SSH multi-scale convolution groups for efficiency
    # NOTE: PAS de bifpn_channels séparé - utiliser out_channel comme V1 pour compatibilité 100%
    'use_pruned_conv': True,          # Enable B-FPGM pruning-aware convolutions
    
    # Weighted Knowledge Distillation (2025 research) - STABILIZED
    'knowledge_distillation': True,
    'distillation_temperature': 2.0,  # Stabilized for numerical stability and convergence
    'distillation_alpha': 0.8,        # Increased distillation weight for better knowledge transfer
    'adaptive_weights': True,          # Adaptive distillation weights
    
    # B-FPGM Bayesian-Optimized Pruning (Kaparinos & Mezaris, WACVW 2025) - ENHANCED STRATEGY
    'pruning_enabled': True,
    'target_reduction': 0.8,                 # Target: 80% reduction (619K → 124K average)
    'stabilization_epochs': 30,              # Phase 1: Enhanced stabilization (modules 2024 + V1 adaptation)
    'pruning_start_epoch': 30,               # Phase 2 start: B-FPGM analysis on stabilized Enhanced
    'pruning_epochs': 20,                    # Phase 2: BO pruning rate optimization (epochs 30-50)
    'full_training_epochs': 250,             # Phase 3: Full training on pruned Enhanced (epochs 50-300)
    'bayesian_iterations': 25,               # Bayesian optimization iterations
    'acquisition_function': 'ei',            # Expected Improvement (Mockus 1989)
    'distance_type': 'l2',                   # FPGM geometric median distance
    'sparsity_schedule': 'polynomial',       # Gradual sparsity introduction
    'num_groups': 6,                         # Enhanced architecture groups (backbone + 3 modules 2024 + V1 components)
    'eval_batches': 100,                     # Speed up BO evaluation
    
    # Scientific foundation (Extended for Nano-B)
    'scientific_basis': {
        'cbam': 'Woo et al. ECCV 2018',
        'bifpn': 'Tan et al. CVPR 2020', 
        'knowledge_distillation': 'Li et al. CVPR 2023',
        'mobilenet': 'Howard et al. 2017',
        'b_fpgm': 'Kaparinos & Mezaris, WACVW 2025',
        'weighted_distillation': '2025 Edge Computing Research',
        'bayesian_optimization': 'Mockus, 1989'
    },
    
    # Performance targets (ENHANCED-FIRST BAYESIAN PRUNING)
    'target_parameter_range': {
        'enhanced_start': 619000,         # Enhanced with all 2024 modules active (CORRECT TARGET)
        'aggressive_pruning': 120000,     # 80% reduction (ultra-efficient post-pruning)
        'balanced_pruning': 150000,       # 76% reduction (optimal balance post-pruning)  
        'conservative_pruning': 180000,   # 71% reduction (safe performance post-pruning)
        'pruning_philosophy': 'Start Enhanced-complete (619K all modules), let Bayesian pruning optimize intelligently',
        'advantage': 'Best-first approach + intelligent automated reduction vs manual optimization'
    },
    
    # Training phases scientific justification (Enhanced Strategy)
    'phase_justification': {
        'phase_1_stabilization': {
            'duration': '30 epochs',
            'goal': 'Enhanced modules adaptation and V1 base integration',
            'scientific_basis': 'Gradient flow stabilization (Frankle & Carbin ICLR 2019)',
            'modules_adaptation': 'ScaleDecoupling + ASSN + MSE-FPN learn to collaborate with V1',
            'knowledge_distillation': 'Basic Teacher → Enhanced Student transfer',
            'computational_cost': 'Minimal (10% of total training on full Enhanced)'
        },
        'phase_2_pruning': {
            'duration': '20 epochs (epochs 30-50)',
            'goal': 'B-FPGM analysis + Bayesian optimization on stabilized Enhanced',
            'scientific_basis': 'B-FPGM on trained weights (Kaparinos & Mezaris WACVW 2025)',
            'advantage': 'Better importance estimation than random initialization',
            'bayesian_optimization': '25 iterations for optimal pruning configuration',
            'output': 'Optimized pruned architecture (619K → 120-180K)'
        },
        'phase_3_training': {
            'duration': '250 epochs (epochs 50-300)',
            'goal': 'Full training on optimized pruned Enhanced architecture',
            'scientific_basis': 'Training on optimal structure (83% of total training)',
            'efficiency': 'Majority of computation on final architecture',
            'knowledge_distillation': 'Complete Teacher → Pruned Enhanced Student transfer',
            'performance_recovery': 'Compensation for structural pruning losses'
        }
    },
    
    # ABLATION STUDY FLAGS (2024 Modules) - All TRUE by default = Enhanced Nano-B 2024
    # Disable individually or in combination to test impact of removing each module
    'ablation_modules': {
        'small_face_optimization': True,   # ScaleDecoupling (SNLA 2024) - targets main V1 limitation: small faces <32x32
        'assn_enabled': True,              # ASSN P3 specialized attention (PMC/ScienceDirect 2024) - replaces generic CBAM on P3
        'mse_fpn_enabled': True,           # MSE-FPN semantic enhancement (Scientific Reports 2024) - targets semantic gap between scales
        
        # Ablation study configuration
        'ablation_mode': 'combined',       # 'individual' | 'progressive' | 'combined' | 'best_combination'
        'target_limitation': 'small_faces', # Primary V1 limitation to address
        'evaluation_metric': 'small_face_ap', # Focus metric for ablation comparison
        
        # Module interaction settings
        'p3_specialized_pipeline': True,   # True when small_face_optimization OR assn_enabled = True
        'differential_processing': True,   # True when P3 has different pipeline than P4/P5
        'preserve_v1_base': True,          # Always keep V1 architecture as foundation (NEVER change this)
    },
    
    # Module-specific configurations (active by default with Enhanced Nano-B 2024)
    'scale_decoupling_config': {
        'enabled': True,   # Controlled by ablation_modules['small_face_optimization']
        'target_layer': 'p3_only',         # Apply only to P3 (small faces)
        'suppression_strength': 0.7,       # Large object suppression factor
        'enhancement_factor': 1.3,         # Small object enhancement multiplier
        'frequency_threshold': 0.5,        # High-frequency features threshold
    },
    
    'assn_config': {
        'enabled': True,   # Controlled by ablation_modules['assn_enabled']
        'target_layer': 'p3_only',         # Replace CBAM on P3 post-BiFPN
        'scale_levels': [80, 40, 20],      # Multi-scale attention levels
        'attention_type': 'scale_sequence', # Specialized for small objects
        'replaces_cbam_on_p3': True,       # Replaces standard CBAM on P3 when enabled
    },
    
    'mse_fpn_config': {
        'enabled': True,   # Controlled by ablation_modules['mse_fpn_enabled']
        'target_layers': 'all_levels',     # Apply to P3, P4, P5
        'semantic_injection': True,        # Context enrichment
        'channel_guidance': True,          # Importance-based channel weighting
        'gated_fusion': True,              # Smart feature combination
        'enhancement_gain': 43.4,          # Validated performance gain from original research
    }
}

# Note: Legacy V2 configuration removed - only cfg_mnet (V1) and cfg_nano_b (Nano-B 2024) supported

