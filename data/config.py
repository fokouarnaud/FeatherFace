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
    'out_channel': 74,  # CALIBRATION: DCN architecture → 487,103 parameters (very close to 488.7K target)
    'lr' : 1e-3,
    'optim' : 'adamw'
}

# Configuration for FeatherFace V2 (Ultra-Efficient)
cfg_mnet_v2 = {
    'name': 'mobilenet0.25',  # Fixed: Must match backbone loading condition
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 400,  # More epochs for knowledge distillation
    'decay1': 250,
    'decay2': 350,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,  # Original config (for compatibility)
    'out_channel_v2': 20,  # V2 Ultra: 20 channels for <250K target
    'lr': 1e-3,
    'optim': 'adamw',
    # V2 Ultra Innovations (Zero/Low Parameter)
    'smart_features': True,      # Enabled with channel-aligned strategy
    'attention_multiply': 3,     # Attention Multiplication (0 params)  
    'cbam_reduction': 128,       # CBAM reduction ratio (75% param reduction)
    'ssh_groups': 1,             # SSH groups (1 for minimal parameters: 32//1=32, 16//1=16, 8//1=8)
    'dynamic_sharing': True,     # Dynamic Weight Sharing (<1K params)
    'progressive_enhance': True, # Progressive Feature Enhancement (0 params)
    'multi_teacher': True,       # Multi-teacher distillation
}

