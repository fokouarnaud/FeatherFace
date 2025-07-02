"""
FeatherFace V2 Training with Knowledge Distillation
Trains the lightweight V2 model using knowledge from the original model
"""

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_mnet_v2
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from layers.modules_distill import (
    DropBlock2D, DistillationLoss, FeatureExtractor,
    mixup_data, cutmix_data, cosine_annealing_with_warmup
)
from layers.advanced_training import (
    GradientClipper, DynamicDistillationLoss, SmartEarlyStopping, 
    TrainingMonitor, AdvancedAugmentation
)
import time
import datetime
import math
import numpy as np
from models.retinaface import RetinaFace
from models.retinaface_v2 import RetinaFaceV2, get_retinaface_v2, count_parameters

parser = argparse.ArgumentParser(description='FeatherFace V2 Training with Knowledge Distillation')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
# Knowledge Distillation arguments
parser.add_argument('--teacher_model', default='./weights/mobilenet0.25_Final.pth', help='Path to teacher model')
parser.add_argument('--teacher_backbone_only', action='store_true', help='Use only backbone weights from teacher')
parser.add_argument('--temperature', default=4.0, type=float, help='Temperature for distillation')
parser.add_argument('--alpha', default=0.7, type=float, help='Weight for distillation loss')
parser.add_argument('--feature_weight', default=0.1, type=float, help='Weight for feature matching loss')

# Augmentation arguments
parser.add_argument('--mixup_alpha', default=0.2, type=float, help='MixUp alpha parameter')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='CutMix probability')
parser.add_argument('--dropblock_prob', default=0.1, type=float, help='DropBlock probability')
parser.add_argument('--dropblock_size', default=3, type=int, help='DropBlock size')

# Advanced Training arguments - NEW
parser.add_argument('--gradient_clip_norm', default=1.0, type=float, help='Gradient clipping max norm')
parser.add_argument('--alpha_initial', default=0.8, type=float, help='Initial alpha for dynamic distillation')
parser.add_argument('--alpha_final', default=0.5, type=float, help='Final alpha for dynamic distillation')
parser.add_argument('--alpha_strategy', default='linear', choices=['linear', 'cosine', 'exponential'], 
                   help='Alpha decay strategy')
parser.add_argument('--early_stopping_patience', default=20, type=int, help='Early stopping patience')
parser.add_argument('--early_stopping_min_epoch', default=100, type=int, help='Minimum epoch before early stopping')
parser.add_argument('--optimal_window_start', default=100, type=int, help='Optimal training window start')
parser.add_argument('--optimal_window_end', default=120, type=int, help='Optimal training window end')
parser.add_argument('--monitor_gradients', action='store_true', help='Enable gradient monitoring')
parser.add_argument('--log_interval', default=10, type=int, help='Logging interval in epochs')

# Training arguments
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--epochs', default=400, type=int, help='Number of epochs')
parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

args = parser.parse_args()

# Setup device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create save folder
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

# Configuration
cfg = cfg_mnet_v2
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
batch_size = args.batch_size
max_epoch = args.epochs

# Setup random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def train_v2():
    # Initialize dataset
    print("Loading dataset...")
    dataset = WiderFaceDetection(args.training_dataset, preproc(img_dim, rgb_mean))
    
    # Create data loader
    train_loader = data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True
    )
    
    # Initialize student model (V2)
    print("Initializing student model (V2)...")
    student_net = get_retinaface_v2(cfg, phase='train')
    student_net = student_net.to(device)
    
    # Add DropBlock after backbone
    dropblock = DropBlock2D(drop_prob=args.dropblock_prob, block_size=args.dropblock_size)
    
    # Count parameters
    student_params = count_parameters(student_net)
    print(f"Student model parameters: {student_params:,} ({student_params/1e6:.3f}M)")
    
    # Initialize teacher model (original)
    print("Loading teacher model...")
    
    if args.teacher_backbone_only:
        print("Using backbone weights only from teacher model")
        # For backbone-only mode, we just need any working model
        teacher_net = student_net  # Use same architecture
        print("Teacher will use same architecture as student")
    else:
        teacher_net = RetinaFace(cfg=cfg_mnet, phase='train')
        teacher_net = teacher_net.to(device)
    
    # Load teacher weights
    if os.path.exists(args.teacher_model):
        print(f"Loading teacher weights from {args.teacher_model}")
        checkpoint = torch.load(args.teacher_model, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Try to load with strict=False to handle architecture differences
        try:
            teacher_net.load_state_dict(state_dict, strict=False)
            print("Teacher weights loaded (with some mismatched keys ignored)")
        except Exception as e:
            print(f"Warning: Could not load all teacher weights: {e}")
            print("Using partially loaded teacher model")
    else:
        print("Warning: Teacher model not found, using random initialization")
    
    teacher_net.eval()  # Teacher in eval mode
    teacher_params = sum(p.numel() for p in teacher_net.parameters())
    print(f"Teacher model parameters: {teacher_params:,} ({teacher_params/1e6:.3f}M)")    
    # Setup feature extraction layers for distillation
    feature_layers = ['body.stage1', 'body.stage2', 'body.stage3']  # Backbone features
    student_extractor = FeatureExtractor(student_net, feature_layers)
    teacher_extractor = FeatureExtractor(teacher_net, feature_layers)
    
    # Initialize losses
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    distill_criterion = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=args.feature_weight
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        student_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_net is not None:
        print(f"Resume training from {args.resume_net}")
        checkpoint = torch.load(args.resume_net, map_location=device)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Setup priorbox
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    priors = priorbox.forward()
    priors = priors.to(device)    
    # Training loop
    print("\nStarting training...")
    iteration = 0
    epoch_size = len(dataset) // batch_size
    
    for epoch in range(start_epoch, max_epoch):
        student_net.train()
        epoch_loss = 0
        task_loss_sum = 0
        distill_loss_sum = 0
        feature_loss_sum = 0
        
        # Adjust learning rate
        lr = cosine_annealing_with_warmup(
            optimizer, epoch, max_epoch,
            lr_min=1e-6, lr_max=args.lr, 
            warmup_epochs=args.warmup_epochs
        )
        
        # Training iterations
        for batch_idx, (images, targets) in enumerate(train_loader):
            iteration += 1
            
            # Move data to device
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            
            # Apply MixUp or CutMix augmentation
            if np.random.rand() < 0.5:  # 50% chance to apply augmentation
                if np.random.rand() < 0.5:
                    # Apply MixUp
                    images, _, _, lam, _ = mixup_data(images, targets, args.mixup_alpha)
                else:
                    # Apply CutMix
                    images, _, _, lam, _ = cutmix_data(images, targets, prob=args.cutmix_prob)
            
            # Apply DropBlock to features (would need to modify model)
            # For simplicity, we'll apply it after getting features
            
            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_outputs, teacher_features = teacher_extractor(images)
            
            # Forward pass through student
            student_outputs, student_features = student_extractor(images)
            
            # Apply DropBlock to student features
            student_features = [dropblock(feat) for feat in student_features]            
            # Compute losses
            # For detection task, we need to handle the outputs properly
            if isinstance(student_outputs, tuple):
                # Unpack in correct order: (bbox_regressions, classifications, landmarks)
                student_bbox, student_cls, student_ldm = student_outputs
                teacher_bbox, teacher_cls, teacher_ldm = teacher_outputs
                
                # MultiBoxLoss expects (loc_data, conf_data, landm_data)
                student_pred = (student_bbox, student_cls, student_ldm)
                teacher_pred = (teacher_bbox, teacher_cls, teacher_ldm)
                
                # Calculate task loss
                loss_cls, loss_box, loss_ldm = criterion(student_pred, priors, targets)
                task_loss = cfg['loc_weight'] * loss_box + loss_cls + loss_ldm
                
                # Calculate distillation loss (simplified for detection)
                # KL divergence on classification outputs
                distill_loss = 0
                for s_cls, t_cls in zip(student_cls, teacher_cls):
                    s_soft = F.log_softmax(s_cls / args.temperature, dim=-1)
                    t_soft = F.softmax(t_cls / args.temperature, dim=-1)
                    distill_loss += F.kl_div(s_soft, t_soft, reduction='batchmean')
                distill_loss *= (args.temperature ** 2)
                
                # Feature matching loss
                feature_loss = 0
                for s_feat, t_feat in zip(student_features, teacher_features):
                    if s_feat.shape[1] != t_feat.shape[1]:
                        # Skip if channels don't match
                        continue
                    feature_loss += F.mse_loss(s_feat, t_feat.detach())
                
                # Total loss
                total_loss = (1 - args.alpha) * task_loss + \
                           args.alpha * distill_loss + \
                           args.feature_weight * feature_loss
            else:
                # Fallback to simple loss
                total_loss = task_loss = criterion(student_outputs, priors, targets)
                distill_loss = feature_loss = torch.tensor(0.0)            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += total_loss.item()
            task_loss_sum += task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
            distill_loss_sum += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            feature_loss_sum += feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{max_epoch} | Iter: {batch_idx}/{epoch_size} | '
                      f'Total Loss: {total_loss.item():.4f} | Task: {task_loss:.4f} | '
                      f'Distill: {distill_loss:.4f} | Feature: {feature_loss:.4f} | '
                      f'LR: {lr:.6f}')
        
        # Epoch summary
        avg_loss = epoch_loss / epoch_size
        avg_task = task_loss_sum / epoch_size
        avg_distill = distill_loss_sum / epoch_size
        avg_feature = feature_loss_sum / epoch_size
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Average Total Loss: {avg_loss:.4f}')
        print(f'  Average Task Loss: {avg_task:.4f}')
        print(f'  Average Distill Loss: {avg_distill:.4f}')
        print(f'  Average Feature Loss: {avg_feature:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'cfg': cfg
            }
            checkpoint_path = os.path.join(args.save_folder, f'FeatherFaceV2_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')    
    # Save final model
    final_model = {
        'model_state_dict': student_net.state_dict(),
        'cfg': cfg,
        'epochs_trained': max_epoch,
        'final_loss': avg_loss
    }
    final_path = os.path.join(args.save_folder, 'FeatherFaceV2_final.pth')
    torch.save(final_model, final_path)
    print(f'\nTraining completed! Final model saved: {final_path}')
    print(f'Total parameters: {student_params:,} ({student_params/1e6:.3f}M)')
    print(f'Compression ratio: {teacher_params/student_params:.2f}x')


if __name__ == '__main__':
    # Set CUDA settings
    cudnn.benchmark = True
    
    # Print configuration
    print("=" * 60)
    print("FeatherFace V2 Training with Knowledge Distillation")
    print("=" * 60)
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (distillation weight): {args.alpha}")
    print(f"Feature weight: {args.feature_weight}")
    print(f"MixUp alpha: {args.mixup_alpha}")
    print(f"CutMix probability: {args.cutmix_prob}")
    print(f"DropBlock: prob={args.dropblock_prob}, size={args.dropblock_size}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Epochs: {max_epoch}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Start training
    train_v2()