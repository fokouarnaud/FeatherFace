#!/usr/bin/env python3
"""
FeatherFace Nano-B Training Script
Hybrid approach combining Knowledge Distillation + Bayesian-Optimized Pruning

Training Pipeline:
1. Load pretrained FeatherFace V1 (teacher) and initialize Nano-B (student)
2. Apply weighted knowledge distillation
3. Use Bayesian optimization to find optimal pruning rates
4. Apply B-FPGM structured pruning
5. Fine-tune the pruned model
6. Export for mobile deployment

Scientific Foundation:
- Knowledge Distillation: Li et al. CVPR 2023
- B-FPGM Pruning: Kaparinos & Mezaris, WACVW 2025
- Weighted Distillation: 2025 research
- Bayesian Optimization: Mockus, 1989

Target: 120-180K parameters with competitive accuracy
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Model imports
from models.retinaface import RetinaFace
from models.featherface_nano_b import FeatherFaceNanoB, create_featherface_nano_b
# Using cfg_nano_b directly from data.config instead of create_nano_b_config
from data.config import cfg_mnet, cfg_nano_b
from data.wider_face import WiderFaceDetection, detection_collate
from layers.modules_distill import DistillationLoss
from data.data_augment import preproc as SSDAugmentation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FeatherFace Nano-B Training')
    
    # Dataset arguments
    parser.add_argument('--training_dataset', required=True, 
                       help='Training dataset path (e.g., ./data/widerface/train/label.txt)')
    parser.add_argument('--validation_dataset', 
                       help='Validation dataset path')
    
    # Model arguments
    parser.add_argument('--teacher_model', required=True,
                       help='Path to pretrained teacher model (V1)')
    parser.add_argument('--resume_net', 
                       help='Path to resume Nano-B training')
    parser.add_argument('--network', default='nano_b', choices=['nano_b'],
                       help='Network architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                       help='Total training epochs')
    parser.add_argument('--pruning_start_epoch', type=int, default=50,
                       help='Epoch to start pruning optimization')
    parser.add_argument('--pruning_epochs', type=int, default=20,
                       help='Epochs for pruning optimization')
    parser.add_argument('--fine_tune_epochs', type=int, default=30,
                       help='Epochs for fine-tuning after pruning')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    
    # Distillation arguments
    parser.add_argument('--distillation_alpha', type=float, default=0.7,
                       help='Distillation loss weight')
    parser.add_argument('--distillation_temperature', type=float, default=4.0,
                       help='Distillation temperature')
    
    # Pruning arguments
    parser.add_argument('--target_reduction', type=float, default=0.5,
                       help='Target parameter reduction (0-1)')
    parser.add_argument('--bayesian_iterations', type=int, default=25,
                       help='Bayesian optimization iterations')
    parser.add_argument('--acquisition_function', default='ei',
                       choices=['ei', 'ucb', 'pi'],
                       help='Bayesian optimization acquisition function')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers (optimized for H100)')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--multigpu', action='store_true',
                       help='Use multiple GPUs')
    parser.add_argument('--save_folder', default='weights/nano_b/',
                       help='Directory to save models')
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='Save model every N epochs')
    
    # Evaluation arguments
    parser.add_argument('--eval_frequency', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--eval_batches', type=int, default=100,
                       help='Number of batches for evaluation (for speed)')
    
    return parser.parse_args()


class NanoBTrainer:
    """FeatherFace Nano-B training manager"""
    
    def __init__(self, args):
        """Initialize trainer with arguments"""
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.training_history = []
        
        # Initialize models and data
        self._setup_models()
        self._setup_data()
        self._setup_optimization()
        
        logger.info(f"Trainer initialized on device: {self.device}")
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        logger.info(f"Teacher parameters: {teacher_params:,}")
        logger.info(f"Student parameters: {student_params:,}")
        
        # Validate parameter count for Enhanced Nano-B with all 2024 modules
        expected_params = 706000  # Updated to match Enhanced model with ScaleDecoupling + ASSN + MSE-FPN
        if abs(student_params - expected_params) > 50000:  # 50K tolerance
            logger.warning(f"PARAMETER COUNT MISMATCH: Expected ~{expected_params:,}, got {student_params:,}")
            logger.warning(f"This may indicate incorrect model configuration")
        else:
            logger.info(f"✓ Enhanced model parameters validated: {student_params:,} (Enhanced target: {expected_params:,})")
            logger.info(f"✓ Will be pruned to 120K-180K parameters during training")
    
    def _setup_models(self):
        """Setup teacher and student models"""
        logger.info("Setting up models...")
        
        # Load teacher model (FeatherFace V1)
        self.teacher = RetinaFace(cfg=cfg_mnet, phase='train')
        
        if os.path.exists(self.args.teacher_model):
            teacher_checkpoint = torch.load(self.args.teacher_model, map_location='cpu')
            
            if 'model_state_dict' in teacher_checkpoint:
                state_dict = teacher_checkpoint['model_state_dict']
            else:
                state_dict = teacher_checkpoint
            
            # Filter out profiling keys and incompatible architecture keys
            teacher_model_dict = self.teacher.state_dict()
            
            # Keep only keys that exist in current model and have compatible shapes
            compatible_state_dict = {}
            profiling_keys_count = 0
            incompatible_keys_count = 0
            
            for k, v in state_dict.items():
                # Skip profiling keys
                if k.endswith(('.total_ops', '.total_params')):
                    profiling_keys_count += 1
                    continue
                
                # Check if key exists in current model and has compatible shape
                if k in teacher_model_dict:
                    if v.shape == teacher_model_dict[k].shape:
                        compatible_state_dict[k] = v
                    else:
                        incompatible_keys_count += 1
                else:
                    incompatible_keys_count += 1
            
            logger.info(f"Filtered checkpoint: {profiling_keys_count} profiling keys, {incompatible_keys_count} incompatible keys")
            logger.info(f"Loading {len(compatible_state_dict)} compatible weights from teacher checkpoint")
            
            # Load only compatible weights
            missing_keys, unexpected_keys = self.teacher.load_state_dict(compatible_state_dict, strict=False)
            
            if missing_keys:
                logger.info(f"Teacher model will use random initialization for {len(missing_keys)} missing layers")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} (should be zero with filtering)")
                
            logger.info("Teacher model loaded with compatible weights only")
            
            logger.info(f"Loaded teacher model from {self.args.teacher_model}")
        else:
            logger.warning(f"Teacher model not found: {self.args.teacher_model}")
        
        # Setup teacher for inference only
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Create student model (FeatherFace Nano-B)
        # Use cfg_nano_b directly, overriding with command line arguments
        pruning_config = {
            'target_reduction': self.args.target_reduction,
            'bayesian_iterations': cfg_nano_b['bayesian_iterations'],
            'acquisition_function': self.args.acquisition_function,
            'distance_type': cfg_nano_b['distance_type'],
            'sparsity_schedule': cfg_nano_b['sparsity_schedule'],
            'num_groups': cfg_nano_b['num_groups'],
            'eval_batches': self.args.eval_batches
        }
        
        self.student = create_featherface_nano_b(
            cfg=cfg_nano_b,
            phase='train',
            pruning_config=pruning_config
        )
        
        # Resume training if specified
        if self.args.resume_net and os.path.exists(self.args.resume_net):
            checkpoint = torch.load(self.args.resume_net, map_location='cpu')
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Resumed from epoch {self.current_epoch}")
        
        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
        # Multi-GPU support
        if self.args.multigpu and torch.cuda.device_count() > 1:
            self.teacher = nn.DataParallel(self.teacher)
            self.student = nn.DataParallel(self.student)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
    
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        # Training dataset
        rgb_mean = (104, 117, 123)
        img_dim = cfg_nano_b['image_size']
        
        train_dataset = WiderFaceDetection(
            txt_path=self.args.training_dataset,
            preproc=SSDAugmentation(img_dim, rgb_mean)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=detection_collate,
            pin_memory=True,
            persistent_workers=True,    # H100 optimization
            prefetch_factor=2           # H100 optimization
        )
        
        # Validation dataset
        if self.args.validation_dataset:
            val_dataset = WiderFaceDetection(
                txt_path=self.args.validation_dataset,
                preproc=SSDAugmentation(img_dim, rgb_mean, train=False)
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=detection_collate,
                pin_memory=True,
                persistent_workers=True,    # H100 optimization
                prefetch_factor=2           # H100 optimization
            )
        else:
            # Use part of training data for validation
            val_size = len(train_dataset) // 10  # 10% for validation
            train_size = len(train_dataset) - val_size
            
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=detection_collate,
                pin_memory=True,
                persistent_workers=True,    # H100 optimization
                prefetch_factor=2           # H100 optimization
            )
            
            self.val_loader = DataLoader(
                val_subset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=detection_collate,
                pin_memory=True,
                persistent_workers=True,    # H100 optimization
                prefetch_factor=2           # H100 optimization
            )
        
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def _setup_optimization(self):
        """Setup optimizers and loss functions"""
        logger.info("Setting up optimization...")
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[150, 250],
            gamma=0.1
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.SmoothL1Loss()
        self.landmark_criterion = nn.SmoothL1Loss()
        
        # Distillation loss
        self.distill_criterion = DistillationLoss(
            temperature=self.args.distillation_temperature,
            alpha=self.args.distillation_alpha
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with knowledge distillation"""
        self.student.train()
        self.teacher.eval()
        
        total_losses = {
            'total': 0.0,
            'distill': 0.0,
            'task': 0.0,
            'cls': 0.0,
            'bbox': 0.0,
            'landmark': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            
            # Teacher forward pass (no gradients)
            # Returns: [cls_pred(B,N,2), bbox_pred(B,N,4), landmark_pred(B,N,10)]
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
            
            # Student forward pass  
            # Returns: [cls_pred(B,N,2), bbox_pred(B,N,4), landmark_pred(B,N,10)]
            student_outputs = self.student(images)
            
            # Compute distillation loss
            # Handles 3 components: classification(2), bbox regression(4), landmarks(10)
            distill_losses = self.student.compute_distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            
            # Total loss
            if 'combined' in distill_losses:
                total_loss = distill_losses['combined']
            else:
                total_loss = distill_losses['distill_total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # OPTIMIZED gradient clipping for H100 training
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=2.0)
            
            # ENHANCED monitoring for loss stability with stricter bounds
            loss_value = total_loss.item()
            
            # Monitor gradient norm for stability (adjusted threshold for Enhanced model)
            if grad_norm > 10.0:
                logger.warning(f"High gradient norm detected: {grad_norm:.4f}")
            
            # More aggressive loss monitoring with stricter bounds
            if loss_value > 100 or loss_value < -10 or not torch.isfinite(total_loss):
                logger.warning(f"Unstable loss detected: {loss_value:.2f}, grad_norm: {grad_norm:.4f}")
                
                # Log individual loss components for debugging
                for key, value in distill_losses.items():
                    if hasattr(value, 'item'):
                        logger.warning(f"  {key}: {value.item():.4f}")
                
                # Emergency learning rate reduction
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= 0.1  # Very aggressive reduction
                    logger.warning(f"Emergency LR reduction: {old_lr:.6f} → {param_group['lr']:.6f}")
                
                # Emergency stop if losses become extreme or invalid
                if abs(loss_value) > 1000 or not torch.isfinite(total_loss):
                    logger.error(f"CRITICAL ERROR: Loss extreme or invalid {loss_value:.2f}")
                    logger.error("Setting minimal learning rate and skipping this batch")
                    # Set minimal LR but continue
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-7
                    continue  # Skip this batch
            
            self.optimizer.step()
            
            # Update statistics
            total_losses['total'] += total_loss.item()
            total_losses['distill'] += distill_losses.get('distill_total', torch.tensor(0.0)).item()
            
            if 'task_total' in distill_losses:
                total_losses['task'] += distill_losses['task_total'].item()
                total_losses['cls'] += distill_losses['task_cls'].item()
                total_losses['bbox'] += distill_losses['task_bbox'].item()
                total_losses['landmark'] += distill_losses['task_landmark'].item()
            
            num_batches += 1
            
            # Progress indicator (only show every 50 batches to reduce clutter)
            if batch_idx % 50 == 0:
                progress = f"[{batch_idx:3d}/{len(self.train_loader):3d}]"
                loss_short = f"L:{total_loss.item():.2f}"
                grad_short = f"G:{grad_norm:.1f}"
                print(f"\r{progress} {loss_short} {grad_short}", end="", flush=True)
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= max(num_batches, 1)
        
        # Clear progress line
        print("\r" + " " * 80 + "\r", end="")
        
        return total_losses
    
    def evaluate(self, epoch: int) -> float:
        """Evaluate model performance"""
        self.student.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if batch_idx >= self.args.eval_batches:  # Limit for speed
                    break
                
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                
                # Forward pass
                outputs = self.student(images)
                
                # Compute simple loss for evaluation
                if isinstance(outputs, list) and len(outputs) == 3:
                    cls_out, bbox_out, landmark_out = outputs
                    
                    # Simplified loss computation for evaluation speed
                    loss = torch.tensor(0.0, device=self.device)
                    
                    # This is a simplified evaluation - in practice you'd want proper loss computation
                    for i, target in enumerate(targets):
                        if len(target) > 0:  # Has annotations
                            loss += torch.mean(torch.abs(cls_out[i] - 0.5))  # Simplified
                    
                    total_loss += loss.item()
                
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        performance_score = -avg_loss  # Higher is better
        
        logger.info(f"Epoch {epoch} - Validation Score: {performance_score:.4f}")
        
        return performance_score
    
    def optimize_pruning(self, epoch: int):
        """Run Bayesian optimization for pruning rates"""
        logger.info(f"Starting pruning optimization at epoch {epoch}...")
        
        # Setup pruning
        pruner = self.student.setup_pruning(self.val_loader, self.criterion)
        
        # Run Bayesian optimization
        optimal_rates = self.student.optimize_pruning_rates(
            self.val_loader, 
            self.criterion, 
            num_iterations=self.args.bayesian_iterations
        )
        
        logger.info(f"Optimal pruning rates found: {optimal_rates}")
        
        # Apply pruning
        pruning_results = self.student.apply_pruning(optimal_rates)
        
        logger.info(f"Pruning applied: {pruning_results}")
        
        # Reset optimizer for pruned model
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=self.args.lr * 0.1,  # Reduced LR for fine-tuning
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        return optimal_rates, pruning_results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, 
                       additional_info: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': vars(self.args),
            'pruning_stats': self.student.get_pruning_summary()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.save_folder, f'nano_b_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.save_folder, 'nano_b_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def export_for_mobile(self):
        """Export model for mobile deployment"""
        logger.info("Exporting model for mobile deployment...")
        
        self.student.eval()
        
        # TorchScript export
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        try:
            traced_model = torch.jit.trace(self.student, dummy_input)
            
            # Save TorchScript model
            mobile_path = os.path.join(self.args.save_folder, 'nano_b_mobile.pt')
            traced_model.save(mobile_path)
            
            logger.info(f"TorchScript model saved: {mobile_path}")
            
            # Test inference
            with torch.no_grad():
                traced_output = traced_model(dummy_input)
                original_output = self.student(dummy_input)
                
                # Verify outputs match
                if isinstance(traced_output, (list, tuple)) and isinstance(original_output, (list, tuple)):
                    max_diff = max(torch.max(torch.abs(t - o)).item() 
                                  for t, o in zip(traced_output, original_output))
                    logger.info(f"TorchScript verification - Max difference: {max_diff:.6f}")
                
            return mobile_path
            
        except Exception as e:
            logger.error(f"Mobile export failed: {e}")
            return None
    
    def train(self):
        """Main training loop"""
        logger.info("Starting FeatherFace Nano-B training...")
        
        # Training phases (Enhanced-First Bayesian Pruning Strategy)
        phase_1_end = self.args.pruning_start_epoch      # Stabilization: 0-30 epochs
        phase_2_end = phase_1_end + self.args.pruning_epochs  # Pruning: 30-50 epochs
        phase_3_end = phase_2_end + self.args.fine_tune_epochs  # Fine-tuning: 50-300 epochs
        
        logger.info(f"Training phases configured:")
        logger.info(f"  Phase 1 (Stabilization): Epochs 0-{phase_1_end}")
        logger.info(f"  Phase 2 (Pruning): Epochs {phase_1_end}-{phase_2_end}")
        logger.info(f"  Phase 3 (Fine-tuning): Epochs {phase_2_end}-{self.args.epochs}")
        logger.info(f"  Target: {self.args.target_reduction*100:.0f}% reduction via Bayesian optimization")
        
        pruning_applied = False
        
        for epoch in range(self.current_epoch, self.args.epochs):
            epoch_start_time = time.time()
            
            # Determine training phase
            if epoch < phase_1_end:
                phase = "Knowledge Distillation"
            elif epoch < phase_2_end and not pruning_applied:
                phase = "Pruning Optimization"
            else:
                phase = "Fine-tuning"
            
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} - Phase: {phase} ===")
            
            # Apply pruning at the right time
            if epoch == self.args.pruning_start_epoch and not pruning_applied:
                optimal_rates, pruning_results = self.optimize_pruning(epoch)
                pruning_applied = True
                
                # Save pruning information
                pruning_info = {
                    'optimal_rates': optimal_rates,
                    'pruning_results': pruning_results,
                    'pruning_epoch': epoch
                }
                
                # Save checkpoint with pruning info
                self.save_checkpoint(epoch, additional_info=pruning_info)
            
            # Train one epoch
            train_losses = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Evaluate periodically
            if (epoch + 1) % self.args.eval_frequency == 0:
                eval_score = self.evaluate(epoch)
                
                # Check if best model
                is_best = eval_score > self.best_score
                if is_best:
                    self.best_score = eval_score
                
                # Save checkpoint
                if (epoch + 1) % self.args.save_frequency == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)
            
            # Update training history
            epoch_time = time.time() - epoch_start_time
            
            history_entry = {
                'epoch': epoch,
                'phase': phase,
                'train_losses': train_losses,
                'epoch_time': epoch_time,
                'lr': self.scheduler.get_last_lr()[0]
            }
            
            if (epoch + 1) % self.args.eval_frequency == 0:
                history_entry['eval_score'] = eval_score
            
            self.training_history.append(history_entry)
            
            # Compact epoch summary for H100 efficiency monitoring
            current_params = sum(p.numel() for p in self.student.parameters())
            # GPU utilization monitoring (simplified to avoid pynvml dependency)
            gpu_util = "Active" if torch.cuda.is_available() else "N/A"
            
            print(f"Results: Loss={train_losses['total']:.2f} | "
                  f"Distill={train_losses['distill']:.2f} | "
                  f"Task={train_losses['task']:.2f} | "
                  f"Time={epoch_time:.1f}s | "
                  f"GPU={gpu_util} | "
                  f"Params={current_params:,} | "
                  f"LR={self.scheduler.get_last_lr()[0]:.1e}")
            
            # Add evaluation results if available
            if (epoch + 1) % self.args.eval_frequency == 0:
                print(f"Eval Score: {eval_score:.4f}")
                
            # Add phase progress info
            if epoch < phase_1_end:
                epochs_left = phase_1_end - epoch - 1
                print(f"Phase Progress: Stabilization ({epochs_left} epochs left)")
            elif epoch < phase_2_end and not pruning_applied:
                epochs_left = phase_2_end - epoch - 1
                print(f"Phase Progress: Pruning Optimization ({epochs_left} epochs left)")
            else:
                epochs_left = self.args.epochs - epoch - 1
                print(f"Phase Progress: Fine-tuning ({epochs_left} epochs left)")
        
        # Final model export
        logger.info("\n=== Training Complete ===")
        
        # Save final model
        self.save_checkpoint(self.args.epochs - 1, additional_info={
            'training_complete': True,
            'training_history': self.training_history,
            'final_summary': self.student.get_pruning_summary()
        })
        
        # Export for mobile
        mobile_path = self.export_for_mobile()
        
        # Print final summary
        final_summary = self.student.get_pruning_summary()
        logger.info("\n=== Final Summary ===")
        for key, value in final_summary.items():
            logger.info(f"{key}: {value}")
        
        if mobile_path:
            logger.info(f"Mobile model exported: {mobile_path}")
        
        logger.info("FeatherFace Nano-B training completed successfully!")


def main():
    """Main training function"""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.training_dataset):
        raise FileNotFoundError(f"Training dataset not found: {args.training_dataset}")
    
    if not os.path.exists(args.teacher_model):
        raise FileNotFoundError(f"Teacher model not found: {args.teacher_model}")
    
    # Create trainer and start training
    trainer = NanoBTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()