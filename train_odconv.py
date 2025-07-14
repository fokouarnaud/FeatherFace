#!/usr/bin/env python3
"""
FeatherFace ODConv Training Script
=================================

This script trains the FeatherFace model with ODConv attention mechanism,
replacing CBAM baseline based on systematic literature review 2025.

Scientific Foundation:
- ODConv: Omni-Dimensional Dynamic Convolution (Li et al. ICLR 2022)
- Proven gains: +3.77-5.71% ImageNet, superior long-range dependencies
- Literature validated: Systematic review 2025 identifies ODConv > CBAM

Performance Targets (vs CBAM baseline):
- WIDERFace Easy: 94.0% AP (+1.3% vs 92.7%)
- WIDERFace Medium: 92.0% AP (+1.3% vs 90.7%)
- WIDERFace Hard: 80.5% AP (+2.2% vs 78.3%)
- Parameters: ~485,000 (vs 488,664 CBAM)

Key Innovations:
- 4D attention: spatial + input channel + output channel + kernel
- Multidimensional modeling vs CBAM 2D attention
- Parameter efficient with superior performance
- Mobile-optimized for real-world deployment

Usage:
    python train_odconv.py --training_dataset ./data/widerface/train/label.txt
    python train_odconv.py --resume_net weights/odconv/checkpoint.pth --resume_epoch 100
"""

from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_odconv
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.featherface_odconv import FeatherFaceODConv
import numpy as np

def parse_args():
    """Parse command line arguments for ODConv training"""
    parser = argparse.ArgumentParser(description='FeatherFace ODConv Training')
    
    # Dataset arguments
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', 
                       help='Training dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, 
                       help='Number of workers used in dataloading')
    
    # Model arguments  
    parser.add_argument('--network', default='odconv', 
                       help='Network version: odconv for ODConv innovation')
    parser.add_argument('--resume_net', default=None, 
                       help='Resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, 
                       help='Resume iter for retraining')
    
    # Training arguments
    parser.add_argument('-b', '--batch_size', default=32, type=int, 
                       help='Batch size for training')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, 
                       help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                       help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                       help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                       help='Gamma update for SGD')
    parser.add_argument('--num_epochs', default=350, type=int, 
                       help='Maximum epoch to train')
    
    # ODConv specific arguments
    parser.add_argument('--odconv_reduction', default=0.0625, type=float,
                       help='ODConv reduction ratio for attention mechanisms')
    parser.add_argument('--odconv_kernel_num', default=1, type=int,
                       help='Number of kernels in ODConv (1 for efficiency)')
    parser.add_argument('--odconv_temperature', default=31, type=int,
                       help='Temperature for ODConv attention softmax')
    
    # Output arguments
    parser.add_argument('--save_folder', default='./weights/odconv/', 
                       help='Directory for saving checkpoint models')
    parser.add_argument('--save_frequency', default=10, type=int,
                       help='Save model every N epochs')
    
    # System arguments
    parser.add_argument('--gpu_train', default=True, type=bool, 
                       help='Use GPU for training')
    parser.add_argument('--ngpu', default=1, type=int, 
                       help='Number of GPUs to use')
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed for reproducibility')
    
    # Monitoring arguments
    parser.add_argument('--verbose', default=True, type=bool,
                       help='Verbose logging including ODConv attention analysis')
    parser.add_argument('--log_attention', default=False, type=bool,
                       help='Log ODConv attention weights for analysis')
    
    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, initial_lr, gamma, decay_epochs):
    """Adjust learning rate based on epoch"""
    lr = initial_lr
    for decay_epoch in decay_epochs:
        if epoch >= decay_epoch:
            lr *= gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def save_checkpoint(model, optimizer, epoch, save_path, cfg, loss=None):
    """Save model checkpoint with ODConv configuration"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cfg': cfg,
        'loss': loss,
        'odconv_config': cfg.get('odconv_config', {}),
        'timestamp': datetime.datetime.now(),
        'model_type': 'FeatherFace_ODConv'
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if available
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✅ Checkpoint loaded: {checkpoint_path} (epoch {epoch})")
    
    return epoch

def analyze_odconv_attention(model, data_loader, device, max_batches=5):
    """
    Analyze ODConv attention patterns for monitoring training
    
    Provides insights into how 4D attention evolves during training.
    """
    print("\n🔍 Analyzing ODConv Attention Patterns...")
    
    model.eval()
    attention_stats = {
        'spatial_mean': [],
        'channel_in_mean': [],
        'channel_out_mean': [],
        'kernel_mean': []
    }
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            images = images.to(device)
            
            # Get attention analysis
            attention_analysis = model.get_attention_analysis(images)
            
            # Analyze backbone attention (first module as example)
            backbone_att = attention_analysis['backbone_attention']['odconv_0']
            
            attention_stats['spatial_mean'].append(backbone_att['spatial'].mean().item())
            attention_stats['channel_in_mean'].append(backbone_att['channel_in'].mean().item())
            attention_stats['channel_out_mean'].append(backbone_att['channel_out'].mean().item())
            attention_stats['kernel_mean'].append(backbone_att['kernel'].mean().item())
    
    model.train()
    
    # Compute statistics
    for key in attention_stats:
        attention_stats[key] = {
            'mean': np.mean(attention_stats[key]),
            'std': np.std(attention_stats[key])
        }
    
    print(f"  📊 Spatial attention: {attention_stats['spatial_mean']['mean']:.4f} ± {attention_stats['spatial_mean']['std']:.4f}")
    print(f"  📊 Input channel attention: {attention_stats['channel_in_mean']['mean']:.4f} ± {attention_stats['channel_in_mean']['std']:.4f}")
    print(f"  📊 Output channel attention: {attention_stats['channel_out_mean']['mean']:.4f} ± {attention_stats['channel_out_mean']['std']:.4f}")
    print(f"  📊 Kernel attention: {attention_stats['kernel_mean']['mean']:.4f} ± {attention_stats['kernel_mean']['std']:.4f}")
    
    return attention_stats

def train_epoch(model, data_loader, criterion, optimizer, device, epoch, cfg, args):
    """Train one epoch with ODConv monitoring"""
    model.train()
    
    epoch_loss = 0.0
    epoch_start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = images.to(device)
        targets = [ann.to(device) for ann in targets]
        
        # Forward pass
        out = model(images)
        
        # Compute loss
        priorbox = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
        priors = priorbox.forward()
        priors = priors.to(device)
        
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Verbose logging
        if args.verbose and batch_idx % 50 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(data_loader)} | '
                  f'Loss: {loss.item():.4f} | Loc: {loss_l.item():.4f} | '
                  f'Cls: {loss_c.item():.4f} | Landm: {loss_landm.item():.4f}')
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = epoch_loss / len(data_loader)
    
    print(f'📈 Epoch {epoch} completed: Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s')
    
    return avg_loss

def main():
    """Main training function for FeatherFace ODConv"""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("🚀 FeatherFace ODConv Training Started")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"  Network: {args.network}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  ODConv reduction: {args.odconv_reduction}")
    print(f"  ODConv kernels: {args.odconv_kernel_num}")
    print(f"  ODConv temperature: {args.odconv_temperature}")
    
    # Update configuration with ODConv parameters
    cfg = cfg_odconv.copy()
    cfg['batch_size'] = args.batch_size
    cfg['lr'] = args.lr
    cfg['epoch'] = args.num_epochs
    cfg['odconv_config'] = {
        'reduction': args.odconv_reduction,
        'kernel_num': args.odconv_kernel_num,
        'temperature': args.odconv_temperature,
        'init_weight': True,
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_train else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if device.type == 'cuda':
        cudnn.benchmark = True
    
    # Create save directory
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Create model
    print(f"\n🏗️  Creating FeatherFace ODConv model...")
    model = FeatherFaceODConv(cfg=cfg, phase='train')
    
    # Model analysis
    param_info = model.get_parameter_count()
    comparison = model.compare_with_cbam_baseline()
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  ODConv parameters: {param_info['total_odconv']:,}")
    print(f"  vs CBAM baseline: {param_info['improvement_vs_cbam']:+,} parameters")
    print(f"  Efficiency gain: {(param_info['improvement_vs_cbam']/param_info['cbam_baseline']*100):.2f}%")
    print(f"  Attention type: {comparison['attention_capability']['odconv_dimensions']}")
    
    # Move model to device
    model = model.to(device)
    
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup loss criterion
    criterion = MultiBoxLoss(cfg['num_classes'], 0.35, True, 0, True, 7, 0.35, False)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume_net:
        start_epoch = load_checkpoint(model, optimizer, args.resume_net)
        start_epoch = max(start_epoch, args.resume_epoch)
    
    # Setup dataset
    print(f"\n📁 Loading dataset: {args.training_dataset}")
    dataset = WiderFaceDetection(args.training_dataset, preproc(cfg['image_size'], cfg['rgb_mean']))
    
    data_loader = data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} images, {len(data_loader)} batches")
    
    # Training loop
    print(f"\n🎯 Starting ODConv training from epoch {start_epoch}...")
    decay_epochs = [cfg['decay1'], cfg['decay2']]
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, args.lr, args.gamma, decay_epochs)
        
        print(f"\n🔄 Epoch {epoch}/{args.num_epochs} | LR: {current_lr:.2e}")
        
        # Train one epoch
        avg_loss = train_epoch(model, data_loader, criterion, optimizer, device, epoch, cfg, args)
        
        # ODConv attention analysis (optional)
        if args.log_attention and epoch % 20 == 0:
            analyze_odconv_attention(model, data_loader, device)
        
        # Save checkpoint
        if epoch % args.save_frequency == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_name = f"featherface_odconv_best.pth"
                print(f"🎉 New best model! Loss: {best_loss:.4f}")
            else:
                save_name = f"featherface_odconv_epoch_{epoch}.pth"
            
            save_path = os.path.join(args.save_folder, save_name)
            save_checkpoint(model, optimizer, epoch, save_path, cfg, avg_loss)
    
    # Final save
    final_save_path = os.path.join(args.save_folder, "featherface_odconv_final.pth")
    save_checkpoint(model, optimizer, args.num_epochs, final_save_path, cfg, avg_loss)
    
    print(f"\n🎉 FeatherFace ODConv training completed!")
    print(f"📁 Models saved in: {args.save_folder}")
    print(f"🎯 Final performance analysis:")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  ODConv efficiency: {param_info['odconv_efficiency']:.1f}% of total")
    print(f"  Next step: Evaluate on WIDERFace with test_widerface.py")

if __name__ == '__main__':
    main()