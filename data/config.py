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

# Configuration for FeatherFace Nano (Scientifically Justified Ultra-Efficient)
cfg_nano = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 400,  # Extended for knowledge distillation
    'decay1': 250,
    'decay2': 350,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel_nano': 64,  # Nano optimized channels for 344K parameters
    
    # Learning configuration
    'lr': 1e-3,
    'optim': 'adamw',
    'weight_decay': 5e-4,
    
    # Scientifically justified efficiency techniques
    'cbam_reduction': 32,        # Efficient CBAM (Woo et al. ECCV 2018)
    'ssh_groups': 4,             # Grouped SSH (established technique)
    
    # Knowledge Distillation (Li et al. CVPR 2023)
    'knowledge_distillation': True,
    'temperature': 4.0,          # Distillation temperature
    'alpha': 0.7,                # Distillation weight
    'feature_weight': 0.1,       # Feature alignment weight
    
    # Scientific foundation
    'scientific_basis': {
        'cbam': 'Woo et al. ECCV 2018',
        'bifpn': 'Tan et al. CVPR 2020', 
        'knowledge_distillation': 'Li et al. CVPR 2023',
        'mobilenet': 'Howard et al. 2017'
    }
}

# Legacy V2 configuration (deprecated - use cfg_nano instead)
cfg_mnet_v2 = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 400,
    'decay1': 250,
    'decay2': 350,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'lr': 1e-3,
    'optim': 'adamw',
    'cbam_reduction': 32,
    'ssh_groups': 4,
}

