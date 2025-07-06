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


# Configuration for FeatherFace Nano-B (Bayesian-Optimized Ultra-Efficient)
# IMPORTANT: Le nombre final de paramètres sera VARIABLE (120K-180K) selon
# l'optimisation bayésienne. Cette variabilité est un AVANTAGE car elle permet
# de trouver automatiquement la meilleure configuration vs un nombre fixe suboptimal.
cfg_nano_b = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 300,  # Optimized for Bayesian pruning pipeline
    'decay1': 150,
    'decay2': 250,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 32,           # Nano-B optimized for 120-180K parameters (variable)
    'num_classes': 2,            # Binary classification (face/no-face)
    
    # Learning configuration  
    'lr': 1e-3,
    'optim': 'adamw',
    'weight_decay': 5e-4,
    
    # Nano-B specific efficiency techniques
    'cbam_reduction': 8,         # Efficient CBAM (Woo et al. ECCV 2018)
    'ssh_groups': 2,             # Grouped SSH (established technique)
    'bifpn_channels': 72,        # BiFPN output channels (divisible by 4)
    'use_pruned_conv': True,     # Enable pruning-aware convolutions
    
    # Weighted Knowledge Distillation (2025 research)
    'knowledge_distillation': True,
    'distillation_temperature': 4.0,  # Distillation temperature
    'distillation_alpha': 0.7,        # Distillation weight
    'adaptive_weights': True,          # Adaptive distillation weights
    
    # Bayesian-Optimized Pruning (B-FPGM - Kaparinos & Mezaris, WACVW 2025)
    'pruning_enabled': True,
    'target_reduction': 0.5,           # 50% parameter reduction target
    'pruning_start_epoch': 50,         # Start pruning after initial training
    'pruning_epochs': 20,              # Bayesian optimization epochs
    'fine_tune_epochs': 30,            # Fine-tuning after pruning
    'bayesian_iterations': 25,         # BO iterations for rate optimization
    'acquisition_function': 'ei',      # Expected Improvement
    'distance_type': 'l2',             # FPGM distance metric
    'sparsity_schedule': 'polynomial', # SFP schedule
    
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
    
    # Performance targets (PLAGE VARIABLE selon optimisation bayésienne)
    'target_parameters': {
        'nano_b_min': 120000,        # Minimum target (65% reduction from V1)
        'nano_b_max': 180000,        # Maximum target (48% reduction from V1)  
        'nano_b_optimal': 150000,    # Optimal target (56% reduction from V1)
        'variability_reason': 'Bayesian optimization finds best config automatically'
    }
}

# Note: Legacy V2 configuration removed - only cfg_mnet (V1) and cfg_nano_b (Enhanced 2024) supported

