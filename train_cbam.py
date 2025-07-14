#!/usr/bin/env python3
"""
FeatherFace CBAM Baseline Training Script

This script trains the CBAM baseline model for scientific comparison with ECA-Net.
Uses cfg_cbam_paper_exact configuration to achieve 488,664 parameters.

Scientific Foundation:
- CBAM: Convolutional Block Attention Module (Woo et al. ECCV 2018)
- Target: 92.7% Easy, 90.7% Medium, 78.3% Hard WIDERFace AP
- Parameters: 488,664 (baseline for ECA-Net comparison)
"""

from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_cbam_paper_exact
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.featherface_cbam_exact import FeatherFaceCBAMExact
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FeatherFace CBAM Baseline Training')
    
    # Dataset arguments
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', 
                       help='Training dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, 
                       help='Number of workers used in dataloading')
    
    # Model arguments
    parser.add_argument('--network', default='cbam', 
                       help='Network version: cbam for CBAM baseline')
    parser.add_argument('--resume_net', default=None, 
                       help='Resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, 
                       help='Resume iter for retraining')
    
    # Training arguments
    parser.add_argument('-b', '--batch_size', default=32, type=int, 
                       help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, 
                       help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                       help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                       help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                       help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/cbam/', 
                       help='Directory for saving checkpoint models')
    
    return parser.parse_args()

def safe_model_loading(model, checkpoint_path):
    """Safely load model checkpoint, filtering out thop profiling keys"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Filter out thop profiling keys that can corrupt model saving
        if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
            # Direct state dict
            filtered_state_dict = {}
            for key, value in checkpoint.items():
                if not (key.endswith('total_ops') or key.endswith('total_params')):
                    # Remove 'module.' prefix if present
                    clean_key = key[7:] if key.startswith('module.') else key
                    filtered_state_dict[clean_key] = value
            
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Successfully loaded model from {checkpoint_path}")
        else:
            print(f"Warning: Unexpected checkpoint format in {checkpoint_path}")
    
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
    
    return model

def main():
    args = parse_args()
    
    # Configuration
    cfg = cfg_cbam_paper_exact
    
    # Create save directory
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # CUDA setup
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # Initialize model
    net = FeatherFaceCBAMExact(cfg=cfg)
    print("Printing net...")
    print(net)
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Target: 488,664 parameters (CBAM baseline)")
    
    if args.resume_net is not None:
        print(f'Resuming training, loading {args.resume_net}...')
        net = safe_model_loading(net, args.resume_net)
    
    if cfg['ngpu'] > 1 and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(cfg['ngpu'])))
    
    if torch.cuda.is_available():
        net = net.cuda()
        cudnn.benchmark = True
    
    # Optimizer
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss
    criterion = MultiBoxLoss(cfg['num_classes'], 0.35, True, 0, True, 7, 0.35, False)
    
    # Prior boxes
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if torch.cuda.is_available():
            priors = priors.cuda()
    
    # Dataset
    print("Loading Dataset...")
    dataset = WiderFaceDetection(args.training_dataset, preproc(cfg['image_size'], cfg['rgb_mean']))
    
    # Learning rate scheduling
    def adjust_learning_rate(optimizer, epoch, initial_lr, gamma, stepsize):
        """Sets the learning rate"""
        lr = initial_lr * (gamma ** (epoch // stepsize))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    print('Loading Dataset...')
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    
    # Training loop
    net.train()
    epoch = 0 + args.resume_epoch
    print('Training FeatherFace CBAM baseline on:', dataset.name)
    print('Using the specified args:')
    print(args)
    
    step_index = 0
    for iteration in range(cfg['max_epoch']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, iteration, args.lr, args.gamma, cfg['lr_steps'][step_index-1])
        
        load_t0 = time.time()
        for batch_idx, (images, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            
            # Forward
            t0 = time.time()
            out = net(images)
            
            # Backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            t1 = time.time()
            
            if batch_idx % 10 == 0:
                print(f'Epoch:{epoch} || epochiter: {batch_idx}/{len(data_loader)} || Loss: {loss.item():.4f} || LR: {optimizer.param_groups[0]["lr"]:.6f} || Batchtime: {t1-t0:.4f} s')
        
        load_t1 = time.time()
        epoch += 1
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_name = os.path.join(args.save_folder, f'featherface_cbam_epoch_{epoch}.pth')
            
            # Clean state dict before saving (remove any thop profiling keys)
            state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
            clean_state_dict = {}
            for key, value in state_dict.items():
                if not (key.endswith('total_ops') or key.endswith('total_params')):
                    clean_state_dict[key] = value
            
            torch.save(clean_state_dict, save_name)
            print(f'Saved checkpoint: {save_name}')
    
    # Save final model
    final_save_name = os.path.join(args.save_folder, 'featherface_cbam_final.pth')
    state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
    clean_state_dict = {}
    for key, value in state_dict.items():
        if not (key.endswith('total_ops') or key.endswith('total_params')):
            clean_state_dict[key] = value
    
    torch.save(clean_state_dict, final_save_name)
    print(f'Final model saved: {final_save_name}')

if __name__ == '__main__':
    main()