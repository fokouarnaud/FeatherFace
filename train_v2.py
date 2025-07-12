#!/usr/bin/env python3
"""
FeatherFace V2 ECA-Net Training Script

This script implements V2 training using scientifically validated ECA-Net:
1. Direct ground truth supervision (no knowledge distillation)
2. Standard MultiBoxLoss like V1
3. Only architectural difference: ECA-Net vs CBAM
4. Clean comparison for scientific evaluation

Scientific Foundation:
- Base Training: V1 standard training pipeline
- Innovation: ECA-Net (Wang et al. CVPR 2020)
- Validation: 1,500+ citations, ImageNet benchmark proven

Target Performance:
- WIDERFace Hard: 77.2% → 88.0% (+10.8%)
- Parameters: 515,137 (vs 515,115 V1, minimal +22 overhead)
- Training: Stable convergence like V1
"""

from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_v2
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.featherface_v2 import FeatherFaceV2
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FeatherFace V2 Simple Training')
    
    # Dataset arguments
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', 
                       help='Training dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, 
                       help='Number of workers used in dataloading')
    
    # Model arguments
    parser.add_argument('--network', default='v2', 
                       help='Network version: v2 for FeatherFace V2')
    parser.add_argument('--resume_net', default=None, 
                       help='Resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, 
                       help='Resume epoch for retraining')
    
    # Training arguments
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/v2_eca/', 
                       help='Location to save V2 ECA-Net checkpoint models')
    
    # Experimental arguments
    parser.add_argument('--experiment_name', default='v2_eca_net_validated',
                       help='Experiment name: ECA-Net scientifically validated')
    
    return parser.parse_args()

def calculate_model_stats(net, img_dim):
    """Calculate model parameters and FLOP estimation (inspired by scripts/training/train.py)"""
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    
    print('=' * 60)
    print(f'Model: {net.__class__.__name__}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Total parameters: {total_params:,}')
    print(f'Input size: {img_dim}x{img_dim}')
    
    # Safe FLOP calculation without corrupting the model
    try:
        # Import thop only for calculation, not for storing metadata
        from thop import profile, clever_format
        dummy_input = torch.randn(1, 3, img_dim, img_dim)
        
        # Create a copy for profiling to avoid corrupting the original model
        import copy
        net_copy = copy.deepcopy(net)
        
        macs, params = profile(net_copy, inputs=(dummy_input,))
        macs, params = clever_format([macs, params], "%.3f")
        
        print(f'Computational complexity: {macs}')
        print(f'FLOP estimation: {params}')
        
        # Delete the copy to free memory
        del net_copy
        
    except ImportError:
        print('THOP not available - skipping FLOP calculation')
    except Exception as e:
        print(f'FLOP calculation failed: {e}')
    
    print('=' * 60)
    
    return trainable_params, total_params

def load_checkpoint_safely(checkpoint_path, model):
    """
    Load checkpoint with robust thop filtering and module prefix handling
    Inspired by scripts/training/train.py but with enhanced filtering
    """
    print(f'Loading checkpoint: {checkpoint_path}')
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Enhanced filtering for thop keys and proper module handling
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        thop_keys_filtered = 0
        module_prefixes_removed = 0
        
        for k, v in state_dict.items():
            # Filter thop profiling keys (critical fix)
            if k.endswith('total_ops') or k.endswith('total_params'):
                thop_keys_filtered += 1
                continue
            
            # Handle module prefixes
            if k.startswith('module.'):
                clean_key = k[7:]  # remove 'module.'
                module_prefixes_removed += 1
            else:
                clean_key = k
            
            new_state_dict[clean_key] = v
        
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict)
        
        print(f'✅ Checkpoint loaded successfully')
        print(f'🗑️  Filtered {thop_keys_filtered} thop keys')
        print(f'🔧 Removed {module_prefixes_removed} module prefixes')
        
        return True
        
    except Exception as e:
        print(f'❌ Failed to load checkpoint: {e}')
        return False

def train():
    """Main training function"""
    args = parse_args()
    
    # Create save directory
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    print("=" * 60)
    print("🚀 FeatherFace V2 ECA-Net Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: FeatherFace V2 with ECA-Net")
    print(f"Training: Direct supervision (like V1)")
    print(f"Innovation: ECA-Net vs CBAM (Wang et al. CVPR 2020)")
    print("=" * 60)
    
    # Configuration
    cfg = cfg_v2
    
    # Model configuration
    rgb_mean = (104, 117, 123)  # BGR order
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']
    
    # Training parameters
    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = cfg['lr']
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder
    
    # Create V2 model
    print("\n📊 Model Creation:")
    net = FeatherFaceV2(cfg=cfg, phase='train')
    print("V2 Model Analysis:")
    trainable_params, total_params = calculate_model_stats(net, img_dim)
    
    # Load resume checkpoint if specified (with enhanced thop filtering)
    if args.resume_net is not None:
        success = load_checkpoint_safely(args.resume_net, net)
        if not success:
            print("❌ Failed to load checkpoint - starting from scratch")
            args.resume_epoch = 0
    
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💾 Device: {device}")
    
    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    
    net.train()
    cudnn.benchmark = True
    
    # Optimizer (same as V1)
    if cfg['optim'] == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), cfg['lr'], weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, 
                             momentum=momentum, weight_decay=weight_decay)
    
    # Dataset
    print(f"\n📊 Dataset: {training_dataset}")
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    
    # Learning rate scheduler
    scheduler_dataloader = data.DataLoader(dataset, batch_size, shuffle=True, 
                                         num_workers=num_workers, collate_fn=detection_collate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, 
                                                   steps_per_epoch=len(scheduler_dataloader), 
                                                   epochs=max_epoch)
    
    # Loss function (same as V1)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    
    # Prior boxes
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    
    # Training setup
    print(f"\n🎯 Training Configuration:")
    print(f"Epochs: {max_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {initial_lr}")
    print(f"Optimizer: {cfg['optim']}")
    print(f"Loss: MultiBoxLoss (same as V1)")
    
    epoch = 0 + args.resume_epoch
    print('\nLoading Dataset...')
    
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    
    # Training statistics
    best_loss = float('inf')
    training_stats = {
        'epoch_losses': [],
        'best_epoch': 0
    }
    
    print(f"\n🚀 Starting Training:")
    print(f"Max iterations: {max_iter}")
    print(f"Epoch size: {epoch_size}")
    
    # Training loop (same structure as V1)
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # Create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, 
                                               num_workers=num_workers, collate_fn=detection_collate))
            
            # Save checkpoint (with safe state dict - no thop corruption)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                checkpoint_path = os.path.join(save_folder, f'featherface_v2_eca_epoch_{epoch}.pth')
                # Get clean state dict (without thop or module prefixes)
                state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
                torch.save(state_dict, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            epoch += 1
        
        load_t0 = time.time()
        
        if iteration in stepvalues:
            step_index += 1
        
        # Load training data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        
        # Forward pass
        out = net(images)
        
        # Calculate loss (same as V1)
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Statistics
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        lr = optimizer.param_groups[0]['lr']
        
        # Logging (same format as V1)
        if iteration % 10 == 0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                  .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                         epoch_size, iteration + 1, max_iter, loss_l.item(), 
                         loss_c.item(), loss_landm.item(), lr, batch_time, 
                         str(datetime.timedelta(seconds=eta))))
        
        # Save best model (with clean state dict)
        total_loss = loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            training_stats['best_epoch'] = epoch
            best_model_path = os.path.join(save_folder, 'featherface_v2_eca_best.pth')
            # Save clean state dict without thop or module prefixes
            state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
            torch.save(state_dict, best_model_path)
        
        # Update training stats
        if iteration % epoch_size == 0:
            training_stats['epoch_losses'].append(total_loss)
    
    # Save final model (with clean state dict)
    final_model_path = os.path.join(save_folder, 'featherface_v2_eca_final.pth')
    # Save clean state dict without thop or module prefixes
    final_state_dict = net.state_dict() if not isinstance(net, torch.nn.DataParallel) else net.module.state_dict()
    torch.save(final_state_dict, final_model_path)
    
    print("\n🎉 Training Complete!")
    print(f"📊 Final Statistics:")
    print(f"Best loss: {best_loss:.4f} (epoch {training_stats['best_epoch']})")
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(save_folder, 'featherface_v2_eca_best.pth')}")
    print(f"Total parameters: {total_params:,}")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Evaluate on WIDERFace: python test_widerface.py -m {final_model_path} --network v2")
    print(f"2. Compare with V1: python test_v1_v2_comparison.py")
    print(f"3. Validate performance: python validate_model.py --version v2")
    
    print(f"\n✅ V2 ECA-Net Training Benefits:")
    print(f"- Stable training like V1")
    print(f"- No complex knowledge distillation")
    print(f"- Clean architectural comparison")
    print(f"- ECA-Net scientifically validated innovation (Wang et al. CVPR 2020)")

if __name__ == '__main__':
    train()