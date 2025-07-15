#!/usr/bin/env python3
"""
FeatherFace CBAM Baseline Training Script

This script trains the CBAM baseline model for scientific comparison with ODConv.
Uses cfg_cbam_paper_exact configuration to achieve 488,664 parameters.

Scientific Foundation:
- CBAM: Convolutional Block Attention Module (Woo et al. ECCV 2018)
- Target: 92.7% Easy, 90.7% Medium, 78.3% Hard WIDERFace AP
- Parameters: 488,664 (baseline for ODConv comparison)
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
    # Get centralized training configuration
    training_cfg = cfg_cbam_paper_exact['training_config']
    
    parser = argparse.ArgumentParser(description='FeatherFace CBAM Baseline Training')
    
    # Dataset arguments (use centralized config as defaults)
    parser.add_argument('--training_dataset', default=training_cfg['training_dataset'], 
                       help='Training dataset directory')
    parser.add_argument('--num_workers', default=training_cfg['num_workers'], type=int, 
                       help='Number of workers used in dataloading')
    
    # Model arguments (use centralized config as defaults)
    parser.add_argument('--network', default=training_cfg['network'], 
                       help='Network version: cbam for CBAM baseline')
    parser.add_argument('--resume_net', default=training_cfg['resume_net'], 
                       help='Resume net for retraining')
    parser.add_argument('--resume_epoch', default=training_cfg['resume_epoch'], type=int, 
                       help='Resume iter for retraining')
    
    # Training arguments (use centralized config as defaults)
    parser.add_argument('-b', '--batch_size', default=cfg_cbam_paper_exact['batch_size'], type=int, 
                       help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', default=cfg_cbam_paper_exact['lr'], type=float, 
                       help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                       help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                       help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                       help='Gamma update for SGD')
    parser.add_argument('--save_folder', default=training_cfg['save_folder'], 
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
    def adjust_learning_rate(optimizer, epoch, initial_lr, gamma, decay_epochs):
        """Adjust learning rate based on epoch"""
        lr = initial_lr
        for decay_epoch in decay_epochs:
            if epoch >= decay_epoch:
                lr *= gamma
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    print('Loading Dataset...')
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    
    # Training loop
    net.train()
    print('Training FeatherFace CBAM baseline on:', dataset.name)
    print('Using the specified args:')
    print(args)
    
    # Use decay epochs from config
    decay_epochs = cfg['lr_steps']
    best_loss = float('inf')
    
    for epoch in range(args.resume_epoch, cfg['max_epoch']):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, args.lr, args.gamma, decay_epochs)
        
        print(f"\n🔄 Epoch {epoch}/{cfg['max_epoch']} | LR: {current_lr:.2e}")
        
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move to device
            if torch.cuda.is_available():
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            
            # Forward pass
            out = net(images)
            
            # Compute loss
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Verbose logging
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(data_loader)} | '
                      f'Loss: {loss.item():.4f} | Loc: {loss_l.item():.4f} | '
                      f'Cls: {loss_c.item():.4f} | Landm: {loss_landm.item():.4f}')
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(data_loader)
        
        print(f'📈 Epoch {epoch} completed: Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s')
        
        # Save checkpoint every 10 epochs or if best model
        if epoch % 10 == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_name = f"featherface_cbam_best.pth"
                print(f"🎉 New best model! Loss: {best_loss:.4f}")
            else:
                save_name = f"featherface_cbam_epoch_{epoch}.pth"
            
            save_path = os.path.join(args.save_folder, save_name)
            
            # Clean state dict before saving (remove any thop profiling keys)
            state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
            clean_state_dict = {}
            for key, value in state_dict.items():
                if not (key.endswith('total_ops') or key.endswith('total_params')):
                    clean_state_dict[key] = value
            
            torch.save(clean_state_dict, save_path)
            print(f'Saved checkpoint: {save_path}')
    
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