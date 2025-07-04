#!/usr/bin/env python3
"""
FeatherFace Nano Training Script
Ultra-efficient face detection training with scientifically justified knowledge distillation

Based on: Li et al. "Rethinking Feature-Based Knowledge Distillation for Face Recognition" CVPR 2023
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from torch.utils.data import DataLoader
import time
import datetime
import math

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_nano
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from models.featherface_nano import FeatherFaceNano
from utils.training_utils import save_checkpoint, load_checkpoint
from utils.monitoring import setup_training_monitoring


def parse_args():
    parser = argparse.ArgumentParser(description='FeatherFace Nano Training')
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt',
                        help='Training dataset path')
    parser.add_argument('--teacher_model', type=str, required=True,
                        help='Path to teacher model (V1) for knowledge distillation')
    parser.add_argument('--network', default='nano', 
                        help='Network architecture (nano)')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--resume_net', default=None, 
                        help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, 
                        help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/nano/', 
                        help='Location to save checkpoint models')
    parser.add_argument('--temperature', default=4.0, type=float,
                        help='Knowledge distillation temperature')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='Knowledge distillation weight')
    parser.add_argument('--feature_weight', default=0.1, type=float,
                        help='Feature distillation weight')
    parser.add_argument('--epochs', default=400, type=int,
                        help='Number of training epochs')
    
    return parser.parse_args()


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for FeatherFace Nano
    Based on: Li et al. "Rethinking Feature-Based Knowledge Distillation for Face Recognition" CVPR 2023
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, feature_weight=0.1):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_outputs, teacher_outputs, targets, student_features=None, teacher_features=None):
        """
        Compute knowledge distillation loss
        
        Args:
            student_outputs: (bbox, cls, ldm) from student (Nano)
            teacher_outputs: (bbox, cls, ldm) from teacher (V1)
            targets: Ground truth targets
            student_features: Optional intermediate features from student
            teacher_features: Optional intermediate features from teacher
        """
        
        student_bbox, student_cls, student_ldm = student_outputs
        teacher_bbox, teacher_cls, teacher_ldm = teacher_outputs
        
        # 1. Task loss (standard training loss)
        criterion = MultiBoxLoss(num_classes=2, overlap_thresh=0.35, prior_for_matching=True,
                               bkg_label=0, neg_mining=True, neg_pos=7, neg_overlap=0.35,
                               encode_target=False, use_gpu=True)
        
        task_loss = criterion(student_outputs, targets)
        
        # 2. Knowledge distillation on classifications (Li et al. CVPR 2023)
        student_cls_soft = torch.log_softmax(student_cls / self.temperature, dim=-1)
        teacher_cls_soft = torch.softmax(teacher_cls / self.temperature, dim=-1)
        
        cls_distill_loss = self.kld_loss(student_cls_soft, teacher_cls_soft) * (self.temperature ** 2)
        
        # 3. Feature distillation (if features provided)
        feature_distill_loss = torch.tensor(0.0, device=student_bbox.device)
        if student_features is not None and teacher_features is not None:
            for s_feat, t_feat in zip(student_features, teacher_features):
                if s_feat.shape != t_feat.shape:
                    # Adapt feature dimensions if needed
                    s_feat = nn.functional.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                    if s_feat.shape[1] != t_feat.shape[1]:
                        s_feat = nn.functional.interpolate(s_feat, size=t_feat.shape[1:], mode='nearest')
                
                feature_distill_loss += self.mse_loss(s_feat, t_feat.detach())
        
        # 4. Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * cls_distill_loss + self.feature_weight * feature_distill_loss
        
        return total_loss, task_loss, cls_distill_loss, feature_distill_loss


def train():
    args = parse_args()
    
    # Create save directory
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Configuration
    cfg = cfg_nano
    
    # Dataset
    print("📊 Loading FeatherFace Nano training dataset...")
    rgb_mean = (104, 117, 123)  # BGR order
    img_dim = cfg['image_size']
    
    dataset = WiderFaceDetection(args.training_dataset, preproc(img_dim, rgb_mean))
    
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, 
                           num_workers=args.num_workers, collate_fn=detection_collate)
    
    print(f"📊 Training on {len(dataset)} images with batch size {cfg['batch_size']}")
    
    # Teacher Model (V1 - for knowledge distillation)
    print("👨‍🏫 Loading teacher model (FeatherFace V1)...")
    teacher_model = RetinaFace(cfg=cfg_mnet, phase='train')
    teacher_checkpoint = torch.load(args.teacher_model, map_location='cpu')
    teacher_model.load_state_dict(teacher_checkpoint, strict=False)
    teacher_model = teacher_model.cuda()
    teacher_model.eval()  # Teacher in eval mode
    print(f"✓ Teacher model loaded from {args.teacher_model}")
    
    # Student Model (Nano)
    print("🎓 Creating FeatherFace Nano student model...")
    student_model = FeatherFaceNano(cfg=cfg, phase='train')
    
    if args.resume_net is not None:
        print(f"📂 Resuming training from {args.resume_net}")
        student_checkpoint = torch.load(args.resume_net, map_location='cpu')
        student_model.load_state_dict(student_checkpoint['model_state_dict'], strict=False)
    
    student_model = student_model.cuda()
    
    # Count parameters
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"📊 Student (Nano): {student_params:,} parameters")
    print(f"📊 Teacher (V1): {teacher_params:,} parameters")
    print(f"📊 Parameter reduction: {((teacher_params - student_params) / teacher_params * 100):.1f}%")
    
    # Optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[cfg['decay1'], cfg['decay2']], gamma=args.gamma)
    
    # Knowledge Distillation Loss
    distillation_criterion = KnowledgeDistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=args.feature_weight
    )
    
    # Training monitoring
    monitor = setup_training_monitoring("featherface_nano_training")
    
    print(f"\n🚀 Starting FeatherFace Nano training with knowledge distillation")
    print(f"🔬 Scientific basis: Li et al. CVPR 2023")
    print(f"📈 Epochs: {args.epochs}, LR: {args.lr}, Temperature: {args.temperature}")
    print("=" * 80)
    
    student_model.train()
    epoch_size = len(dataset) // cfg['batch_size']
    max_iter = args.epochs * epoch_size
    stepvalues = [cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size]
    step_index = 0
    
    start_epoch = args.resume_epoch
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        student_model.train()
        total_loss = 0
        total_task_loss = 0
        total_distill_loss = 0
        total_feature_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            iteration = epoch * epoch_size + batch_idx
            
            if iteration in stepvalues:
                step_index += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * (args.gamma ** step_index)
                print(f"📉 Learning rate adjusted to: {param_group['lr']:.2e}")
            
            # Move to GPU
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
            
            # Forward pass - Student
            student_outputs = student_model(images)
            
            # Forward pass - Teacher (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            # Knowledge distillation loss
            loss, task_loss, distill_loss, feature_loss = distillation_criterion(
                student_outputs, teacher_outputs, targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_distill_loss += distill_loss.item()
            total_feature_loss += feature_loss.item()
            
            # Logging
            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch+1:3d}/{args.epochs} | "
                      f"Iter: {iteration:6d}/{max_iter} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Task: {task_loss.item():.4f} | "
                      f"Distill: {distill_loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
                
                # Monitor logging
                monitor.log_metrics({
                    'loss': loss.item(),
                    'task_loss': task_loss.item(),
                    'distillation_loss': distill_loss.item(),
                    'feature_loss': feature_loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch + 1
                }, step=iteration)
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_task_loss = total_task_loss / len(dataloader)
        avg_distill_loss = total_distill_loss / len(dataloader)
        avg_feature_loss = total_feature_loss / len(dataloader)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Task Loss: {avg_task_loss:.4f}")
        print(f"  Distillation Loss: {avg_distill_loss:.4f}")
        print(f"  Feature Loss: {avg_feature_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(args.save_folder, f'nano_epoch_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': cfg,
                'args': vars(args)
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_path = os.path.join(args.save_folder, 'nano_final.pth')
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'config': cfg,
        'args': vars(args)
    }, final_path)
    
    print(f"\n✅ FeatherFace Nano training completed!")
    print(f"💾 Final model saved: {final_path}")
    print(f"🔬 Scientific foundation: Knowledge distillation (Li et al. CVPR 2023)")
    print(f"📊 Total parameters: {student_params:,} ({((teacher_params - student_params) / teacher_params * 100):.1f}% reduction)")


if __name__ == '__main__':
    train()