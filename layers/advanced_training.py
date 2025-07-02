"""
Advanced Training Techniques for FeatherFace V2
Implements gradient clipping, dynamic α, early stopping, and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


class GradientClipper:
    """
    Gradient clipping with monitoring for training stability
    """
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.gradient_history = []
        
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return the total norm before clipping
        """
        # Calculate total norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)
        
        # Store for monitoring
        self.gradient_history.append(total_norm)
        if len(self.gradient_history) > 1000:  # Keep last 1000 values
            self.gradient_history.pop(0)
        
        return total_norm
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics for monitoring"""
        if not self.gradient_history:
            return {}
            
        history = np.array(self.gradient_history[-100:])  # Last 100 values
        return {
            'grad_norm_mean': float(np.mean(history)),
            'grad_norm_std': float(np.std(history)),
            'grad_norm_max': float(np.max(history)),
            'grad_norm_current': float(history[-1]) if len(history) > 0 else 0.0
        }


class DynamicDistillationLoss:
    """
    Dynamic α for knowledge distillation with adaptive strategies
    """
    def __init__(self, 
                 initial_alpha: float = 0.8, 
                 final_alpha: float = 0.5,
                 total_epochs: int = 400,
                 strategy: str = 'linear'):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.plateau_counter = 0
        self.best_val_loss = float('inf')
        
    def get_alpha(self, epoch: int, val_loss: Optional[float] = None) -> float:
        """
        Calculate dynamic α based on epoch and optional validation metrics
        """
        # Base alpha calculation
        if self.strategy == 'linear':
            progress = epoch / self.total_epochs
            base_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        
        elif self.strategy == 'cosine':
            progress = epoch / self.total_epochs
            base_alpha = self.final_alpha + (self.initial_alpha - self.final_alpha) * \
                        (1 + np.cos(np.pi * progress)) / 2
        
        elif self.strategy == 'exponential':
            decay_rate = np.log(self.final_alpha / self.initial_alpha) / self.total_epochs
            base_alpha = self.initial_alpha * np.exp(decay_rate * epoch)
        
        else:  # default to linear
            progress = epoch / self.total_epochs
            base_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        
        # Adaptive adjustments based on validation performance
        if val_loss is not None:
            if val_loss >= self.best_val_loss:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
                self.best_val_loss = val_loss
            
            # Reduce distillation if stuck in plateau
            if self.plateau_counter > 10:
                base_alpha *= 0.9  # Reduce distillation, focus more on task loss
        
        # End-of-training focus on task performance
        if epoch > 0.8 * self.total_epochs:
            base_alpha = min(base_alpha, 0.4)  # Cap at 40% distillation
        
        # Ensure bounds
        return max(0.1, min(0.9, base_alpha))
    
    def get_loss_weights(self, epoch: int, val_loss: Optional[float] = None) -> Tuple[float, float]:
        """
        Get (task_weight, distill_weight) for loss computation
        """
        alpha = self.get_alpha(epoch, val_loss)
        return (1 - alpha), alpha


class SmartEarlyStopping:
    """
    Intelligent early stopping with epoch-specific strategies
    """
    def __init__(self, 
                 patience: int = 20,
                 min_epoch: int = 100,
                 optimal_window: Tuple[int, int] = (100, 120),
                 min_delta: float = 1e-4):
        self.patience = patience
        self.min_epoch = min_epoch
        self.optimal_start, self.optimal_end = optimal_window
        self.min_delta = min_delta
        
        self.best_val_loss = float('inf')
        self.best_val_map = 0.0
        self.wait = 0
        self.best_epoch = 0
        
    def should_stop(self, epoch: int, val_loss: float, val_map: float = 0.0) -> bool:
        """
        Determine if training should stop based on multiple criteria
        """
        # Never stop before minimum epoch
        if epoch < self.min_epoch:
            return False
        
        # Track best performance
        improved = False
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_map = val_map
            self.best_epoch = epoch
            self.wait = 0
            improved = True
        else:
            self.wait += 1
        
        # Different strategies for different training phases
        if self.optimal_start <= epoch <= self.optimal_end:
            # In optimal window: more aggressive early stopping
            if self.wait >= self.patience // 2:  # Reduced patience
                logging.info(f"Early stopping at epoch {epoch} in optimal window")
                return True
                
        elif epoch > self.optimal_end:
            # After optimal window: standard early stopping
            if self.wait >= self.patience:
                logging.info(f"Early stopping at epoch {epoch} after optimal window")
                return True
            
            # Additional check: stop if performance degrades significantly
            if val_loss > self.best_val_loss * 1.1:  # 10% degradation
                logging.info(f"Early stopping due to performance degradation at epoch {epoch}")
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """Get early stopping statistics"""
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_map': self.best_val_map,
            'wait': self.wait,
            'patience': self.patience
        }


class TrainingMonitor:
    """
    Comprehensive training monitoring and diagnostics
    """
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.metrics_history = defaultdict(list)
        self.epoch_times = []
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        
    def log_epoch_metrics(self, 
                         epoch: int,
                         train_loss: float,
                         val_loss: float,
                         val_map: float,
                         learning_rate: float,
                         grad_stats: Dict[str, float],
                         alpha: float,
                         epoch_time: float):
        """
        Log comprehensive metrics for an epoch
        """
        # Store metrics
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_map'].append(val_map)
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['alpha'].append(alpha)
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # Add gradient stats
        for key, value in grad_stats.items():
            self.metrics_history[key].append(value)
        
        # Log to console/file
        if epoch % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            eta = (elapsed_time / (epoch + 1)) * (400 - epoch - 1)  # Assume 400 total epochs
            
            logging.info(f"Epoch {epoch:3d}/400 | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val mAP: {val_map:.3f} | "
                        f"LR: {learning_rate:.2e} | "
                        f"α: {alpha:.3f} | "
                        f"Grad: {grad_stats.get('grad_norm_current', 0):.3f} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"ETA: {eta/3600:.1f}h")
    
    def check_training_health(self, grad_stats: Dict[str, float], loss_components: Tuple[float, float, float]):
        """
        Check for training anomalies and log warnings
        """
        task_loss, distill_loss, feature_loss = loss_components
        grad_norm = grad_stats.get('grad_norm_current', 0)
        
        # Check for problematic gradients
        if grad_norm > 10.0:
            logging.warning(f"High gradient norm detected: {grad_norm:.2f}")
        elif grad_norm < 1e-6:
            logging.warning(f"Very low gradient norm detected: {grad_norm:.2e} - possible vanishing gradients")
        
        # Check for loss anomalies
        if task_loss > 100 or distill_loss > 100:
            logging.warning(f"High loss values: task={task_loss:.2f}, distill={distill_loss:.2f}")
        
        # Check for NaN/inf
        if not (np.isfinite(task_loss) and np.isfinite(distill_loss)):
            logging.error(f"NaN/Inf detected in losses: task={task_loss}, distill={distill_loss}")
    
    def save_metrics(self, filepath: str = 'training_metrics.json'):
        """Save metrics history to file"""
        import json
        with open(filepath, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_clean = {}
            for key, values in self.metrics_history.items():
                metrics_clean[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
            json.dump(metrics_clean, f, indent=2)


class AdvancedAugmentation:
    """
    Advanced augmentation strategies for face detection
    """
    def __init__(self, 
                 mixup_alpha: float = 0.2,
                 cutmix_prob: float = 0.5,
                 dropblock_prob: float = 0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.dropblock_prob = dropblock_prob
        
    def apply_mixup(self, images: torch.Tensor, targets: List) -> Tuple[torch.Tensor, List]:
        """Apply MixUp augmentation optimized for face detection"""
        if np.random.random() > 0.5:  # 50% chance
            from layers.modules_distill import mixup_data
            return mixup_data(images, targets, self.mixup_alpha)
        return images, targets
    
    def apply_cutmix(self, images: torch.Tensor, targets: List) -> Tuple[torch.Tensor, List]:
        """Apply CutMix augmentation with face preservation"""
        if np.random.random() < self.cutmix_prob:
            from layers.modules_distill import cutmix_data
            return cutmix_data(images, targets)
        return images, targets


def create_advanced_training_components(config: Dict) -> Tuple:
    """
    Factory function to create all advanced training components
    """
    # Gradient management
    gradient_clipper = GradientClipper(
        max_norm=config.get('grad_clip_norm', 1.0)
    )
    
    # Dynamic distillation
    dynamic_distill = DynamicDistillationLoss(
        initial_alpha=config.get('initial_alpha', 0.8),
        final_alpha=config.get('final_alpha', 0.5),
        total_epochs=config.get('epochs', 400),
        strategy=config.get('alpha_strategy', 'linear')
    )
    
    # Early stopping
    early_stopper = SmartEarlyStopping(
        patience=config.get('patience', 20),
        min_epoch=config.get('min_epoch', 100),
        optimal_window=config.get('optimal_window', (100, 120))
    )
    
    # Training monitor
    monitor = TrainingMonitor(
        log_interval=config.get('log_interval', 10)
    )
    
    # Advanced augmentation
    augmentation = AdvancedAugmentation(
        mixup_alpha=config.get('mixup_alpha', 0.2),
        cutmix_prob=config.get('cutmix_prob', 0.5),
        dropblock_prob=config.get('dropblock_prob', 0.1)
    )
    
    return gradient_clipper, dynamic_distill, early_stopper, monitor, augmentation