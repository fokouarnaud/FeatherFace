#!/usr/bin/env python3
"""
FeatherFace ECA-CBAM Parallel Training Script
===========================================

This script trains the FeatherFace model with ECA-CBAM hybrid attention mechanism.
Combines ECA-Net efficiency with CBAM spatial attention for optimal face detection.

Scientific Innovation:
- ECA-Net channel attention: 22 parameters vs 2000 (CBAM CAM)
- CBAM spatial attention: Preserved for face localization
- Sequential attention: ECA → SAM processing
- Parameter reduction: 8.1% vs CBAM baseline (449,017 vs 488,664)

Expected Performance:
- WIDERFace Easy: 94.0% AP (+1.3% vs CBAM)
- WIDERFace Medium: 92.0% AP (+1.3% vs CBAM)
- WIDERFace Hard: 80.0% AP (+1.7% vs CBAM)
- Overall: 88.7% AP (+1.5% vs CBAM)

Usage:
    python train_eca_cbam_parallel.py --training_dataset ./data/widerface/train/label.txt
    python train_eca_cbam_parallel.py --training_dataset ./data/widerface/train/label.txt --resume_net ./weights/eca_cbam_parallel_parallel/epoch_100.pth
"""

import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
sys.path.append('.')

from data import cfg_eca_cbam_parallel_parallel, cfg_mnet
from models.featherface_eca_cbam_parallel_parallel import FeatherFaceECAcbaMParallel
from data.wider_face import WiderFaceDetection, detection_collate
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import preproc


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FeatherFace ECA-CBAM Parallel Training')
    
    # Training configuration
    parser.add_argument('--training_dataset', 
                       default='./data/widerface/train/label.txt',
                       help='Training dataset path')
    parser.add_argument('--network', 
                       default='eca_cbam_parallel',
                       choices=['eca_cbam_parallel'],
                       help='Network architecture')
    parser.add_argument('--num_workers', 
                       default=8, 
                       type=int,
                       help='Number of workers for data loading')
    parser.add_argument('--lr', 
                       default=1e-3, 
                       type=float,
                       help='Learning rate')
    parser.add_argument('--momentum', 
                       default=0.9, 
                       type=float,
                       help='Momentum for SGD')
    parser.add_argument('--resume_net', 
                       default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--resume_epoch', 
                       default=0, 
                       type=int,
                       help='Resume training from epoch')
    parser.add_argument('--save_folder', 
                       default='./weights/eca_cbam_parallel_parallel/',
                       help='Folder to save checkpoints')
    parser.add_argument('--batch_size', 
                       default=32, 
                       type=int,
                       help='Batch size')
    parser.add_argument('--max_epoch', 
                       default=350, 
                       type=int,
                       help='Maximum training epochs')
    parser.add_argument('--gpu_train', 
                       action='store_true',
                       help='Use GPU for training')
    
    # ECA-CBAM specific parameters
    parser.add_argument('--eca_gamma', 
                       default=2, 
                       type=int,
                       help='ECA gamma parameter for adaptive kernel')
    parser.add_argument('--eca_beta', 
                       default=1, 
                       type=int,
                       help='ECA beta parameter for adaptive kernel')
    parser.add_argument('--sam_kernel_size', 
                       default=7, 
                       type=int,
                       help='CBAM SAM kernel size')
    parser.add_argument('--interaction_weight', 
                       default=0.1, 
                       type=float,
                       help='Cross-combined interaction weight')
    
    # Logging and validation
    parser.add_argument('--log_attention', 
                       action='store_true',
                       help='Log attention analysis during training')
    parser.add_argument('--validate_every', 
                       default=50, 
                       type=int,
                       help='Validation frequency (epochs)')
    parser.add_argument('--save_every', 
                       default=50, 
                       type=int,
                       help='Save frequency (epochs)')
    
    args = parser.parse_args()
    return args


def create_model(cfg, args):
    """Create ECA-CBAM FeatherFace model"""
    print(f"🔬 Creating FeatherFace ECA-CBAM Parallel Model...")
    print(f"📊 Configuration: {cfg['attention_mechanism']}")
    
    # Update configuration with command line arguments
    cfg.update({
        'eca_gamma': args.eca_gamma,
        'eca_beta': args.eca_beta,
        'sam_kernel_size': args.sam_kernel_size,
        'interaction_weight': args.interaction_weight,
    })
    
    # Create model
    model = FeatherFaceECAcbaMParallel(cfg=cfg, phase='train')
    
    # Validate model
    validation, param_info = model.validate_eca_cbam_parallel_hybrid()
    
    print(f"✅ Model created successfully!")
    print(f"📈 Total parameters: {param_info['total']:,}")
    print(f"📉 Parameter reduction: {param_info['parameter_reduction']:,} ({param_info['efficiency_gain']:.1f}%)")
    print(f"🎯 Attention efficiency: {param_info['attention_efficiency']:.0f} params/module")
    
    if not validation['parameter_target_achieved']:
        print(f"⚠️  WARNING: Parameter target not achieved!")
    
    if validation['hybrid_innovation']:
        print(f"🚀 Innovation: ECA-CBAM hybrid attention validated!")
    
    return model, param_info


def load_dataset(cfg, args):
    """Load WIDERFace dataset"""
    print(f"📂 Loading WIDERFace dataset...")
    print(f"📁 Training data: {args.training_dataset}")
    
    # Create dataset
    dataset = WiderFaceDetection(
        txt_path=args.training_dataset,
        preproc=preproc(cfg['image_size'], cfg['rgb_mean'])
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True
    )
    
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🔄 Workers: {args.num_workers}")
    
    return data_loader


def create_optimizer(model, cfg, args):
    """Create optimizer and scheduler"""
    print(f"⚙️  Creating optimizer...")
    
    # Choose optimizer (THESIS: Adam, not AdamW)
    if cfg['optim'] == 'adam' or cfg['optim'] == 'adamw':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4  # THESIS: 1e-4 (not 5e-4)
        )
        print(f"Adam optimizer created (lr={args.lr}, weight_decay=1e-4)")
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=1e-4
        )
        print(f"SGD optimizer created (lr={args.lr}, momentum={args.momentum})")
    
    # Learning rate scheduler (THESIS: CosineAnnealingWarmRestarts, not MultiStepLR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.max_epoch,  # Restart period = total epochs
        T_mult=1,            # No period increase
        eta_min=1e-6         # Minimum learning rate
    )
    
    print(f"LR scheduler: CosineAnnealingWarmRestarts(T_0={args.max_epoch}, eta_min=1e-6)")
    
    return optimizer, scheduler


def train_epoch(model, data_loader, criterion, priors, optimizer, epoch, cfg, args, writer=None):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0.0
    epoch_loss_l = 0.0
    epoch_loss_c = 0.0
    epoch_loss_landm = 0.0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(data_loader):
        if args.gpu_train:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        
        # Forward pass
        out = model(images)
        
        # Compute loss
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_loss_l += loss_l.item()
        epoch_loss_c += loss_c.item()
        epoch_loss_landm += loss_landm.item()
        
        # Log progress
        if i % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:03d}/{args.max_epoch:03d} | '
                  f'Batch {i:04d}/{len(data_loader):04d} | '
                  f'Loss {loss.item():.4f} | '
                  f'LR {current_lr:.6f}')
            
            # TensorBoard logging
            if writer is not None:
                global_step = epoch * len(data_loader) + i
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Localization', loss_l.item(), global_step)
                writer.add_scalar('Loss/Classification', loss_c.item(), global_step)
                writer.add_scalar('Loss/Landmarks', loss_landm.item(), global_step)
                writer.add_scalar('Learning_Rate', current_lr, global_step)
    
    # Epoch statistics
    epoch_time = time.time() - start_time
    num_batches = len(data_loader)
    
    avg_loss = epoch_loss / num_batches
    avg_loss_l = epoch_loss_l / num_batches
    avg_loss_c = epoch_loss_c / num_batches
    avg_loss_landm = epoch_loss_landm / num_batches
    
    print(f'📊 Epoch {epoch:03d} Summary:')
    print(f'   ⏱️  Time: {epoch_time:.1f}s')
    print(f'   📉 Avg Loss: {avg_loss:.4f}')
    print(f'   📍 Loc Loss: {avg_loss_l:.4f}')
    print(f'   🎯 Cls Loss: {avg_loss_c:.4f}')
    print(f'   🔍 Landm Loss: {avg_loss_landm:.4f}')
    
    return avg_loss


def analyze_attention(model, data_loader, epoch, args):
    """Analyze attention patterns during training"""
    if not args.log_attention:
        return
    
    print(f"🔍 Analyzing ECA-CBAM attention patterns...")
    
    model.eval()
    with torch.no_grad():
        # Get a batch for analysis
        images, _ = next(iter(data_loader))
        if args.gpu_train:
            images = images.cuda()
        
        # Analyze attention
        analysis = model.get_attention_analysis(images[:1])  # Use first image
        
        print(f"📊 Attention Analysis - Epoch {epoch}:")
        print(f"   🧠 Mechanism: {analysis['attention_summary']['mechanism']}")
        print(f"   📈 Modules: {analysis['attention_summary']['modules_count']}")
        print(f"   🔧 Channel: {analysis['attention_summary']['channel_attention']}")
        print(f"   📍 Spatial: {analysis['attention_summary']['spatial_attention']}")
        print(f"   🚀 Innovation: {analysis['attention_summary']['innovation']}")
        
        # Log attention statistics
        backbone_stats = []
        for stage, stats in analysis['backbone_attention'].items():
            backbone_stats.append(f"{stage}: {stats['eca_attention_mean']:.4f}")
        
        bifpn_stats = []
        for level, stats in analysis['bifpn_attention'].items():
            bifpn_stats.append(f"{level}: {stats['eca_attention_mean']:.4f}")
        
        print(f"   📊 Backbone attention: {', '.join(backbone_stats)}")
        print(f"   📊 BiFPN attention: {', '.join(bifpn_stats)}")


def save_checkpoint(model, optimizer, epoch, args, param_info):
    """Save model checkpoint"""
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'parameter_info': param_info,
        'eca_cbam_parallel_config': {
            'eca_gamma': args.eca_gamma,
            'eca_beta': args.eca_beta,
            'sam_kernel_size': args.sam_kernel_size,
            'interaction_weight': args.interaction_weight,
        }
    }
    
    checkpoint_path = os.path.join(args.save_folder, f'epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    if epoch == args.max_epoch or epoch % 50 == 0:
        final_path = os.path.join(args.save_folder, f'featherface_eca_cbam_parallel_parallel_epoch_{epoch}.pth')
        torch.save(model.state_dict(), final_path)
        print(f"💾 Model saved: {final_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("🚀 FeatherFace ECA-CBAM Parallel Training")
    print("=" * 60)
    print(f"📅 Started: {datetime.datetime.now()}")
    print(f"🔧 Arguments: {args}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # CUDA setup
    if args.gpu_train and torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        cudnn.benchmark = True
        print(f"🚀 CUDA enabled: {torch.cuda.device_count()} GPUs")
    else:
        args.gpu_train = False
        print(f"💻 CPU training")
    
    # Create save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Load configuration
    cfg = cfg_eca_cbam_parallel_parallel.copy()
    cfg.update(vars(args))
    
    # Create model
    model, param_info = create_model(cfg, args)
    
    if args.gpu_train:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"🔄 Multi-GPU training: {torch.cuda.device_count()} GPUs")
    
    # Load dataset
    data_loader = load_dataset(cfg, args)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, cfg, args)
    
    # Create loss function
    criterion = MultiBoxLoss(2, 0.5, True, 0, True, 7, 0.5, False)
    
    # Prior boxes
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if args.gpu_train:
            priors = priors.cuda()
    
    # Resume training if specified
    start_epoch = args.resume_epoch
    if args.resume_net is not None:
        print(f"🔄 Resuming from {args.resume_net}")
        checkpoint = torch.load(args.resume_net, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.save_folder, 'logs'))
    
    # Training loop
    print(f"🎯 Training ECA-CBAM hybrid for {args.max_epoch} epochs...")
    print(f"📊 Expected performance: +1.5% to +2.5% mAP vs CBAM baseline")
    
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        print(f"\n🔄 Epoch {epoch}/{args.max_epoch}")
        
        # Train epoch
        avg_loss = train_epoch(model, data_loader, criterion, priors, optimizer, epoch, cfg, args, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Analyze attention periodically
        if epoch % args.validate_every == 0:
            analyze_attention(model, data_loader, epoch, args)
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.max_epoch:
            save_checkpoint(model, optimizer, epoch, args, param_info)
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Epoch/Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Final model save
    final_model_path = os.path.join(args.save_folder, 'featherface_eca_cbam_parallel_parallel_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"🎉 Final model saved: {final_model_path}")
    
    # Training summary
    print(f"\n🎉 Training completed!")
    print(f"📊 Total parameters: {param_info['total']:,}")
    print(f"📉 Parameter reduction: {param_info['efficiency_gain']:.1f}% vs CBAM baseline")
    print(f"🎯 Expected performance: +1.5% to +2.5% mAP improvement")
    print(f"⏱️  Training time: {datetime.datetime.now()}")
    
    # Final comparison
    comparison = model.compare_with_cbam_baseline()
    print(f"\n🔬 Final Comparison with CBAM Baseline:")
    print(f"   📊 Parameter efficiency: {comparison['parameter_comparison']['efficiency_gain']}")
    print(f"   📈 Expected performance: {comparison['performance_prediction']['expected_performance']}")
    print(f"   🚀 Innovation: {comparison['performance_prediction']['deployment']}")
    
    writer.close()


if __name__ == '__main__':
    main()