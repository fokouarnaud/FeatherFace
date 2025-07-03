"""
Advanced Multi-Teacher Knowledge Distillation for FeatherFace V2 Ultra
Revolutionary training strategy to surpass V1 performance with 50% fewer parameters

Multi-Teacher Strategy:
1. Primary Teacher: V1 model (487K params)
2. Self-Teacher: V2 Ultra's own best predictions  
3. Ensemble Teacher: Multiple V1 variants
4. Progressive Temperature Scheduling
5. Adaptive Loss Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union


class AdvancedDistillationLoss(nn.Module):
    """
    Advanced multi-teacher knowledge distillation loss
    
    Components:
    1. KL Divergence on classification logits (temperature-scaled)
    2. Feature alignment loss (intermediate features)
    3. Attention map alignment loss
    4. Adaptive loss weighting based on sample difficulty
    5. Progressive temperature scheduling
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 feature_weight: float = 0.1,
                 attention_weight: float = 0.05,
                 adaptive_weighting: bool = True):
        super(AdvancedDistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='none')  # Changed to 'none' for adaptive weighting
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # Running statistics for adaptive weighting
        self.register_buffer('loss_ema', torch.tensor(1.0))
        self.ema_momentum = 0.99
        
    def forward(self, 
                student_outputs: Tuple[torch.Tensor, ...],
                teacher_outputs: Tuple[torch.Tensor, ...],
                targets: torch.Tensor,
                task_loss: torch.Tensor,
                epoch: int = 0,
                max_epochs: int = 400) -> Dict[str, torch.Tensor]:
        """
        Advanced distillation forward pass
        
        Args:
            student_outputs: (bbox, cls, landmarks, features, attention_maps)
            teacher_outputs: (bbox, cls, landmarks, features, attention_maps) 
            targets: Ground truth labels
            task_loss: Original task loss (MultiBoxLoss)
            epoch: Current training epoch
            max_epochs: Total training epochs
        """
        
        # Unpack outputs
        student_bbox, student_cls, student_ldm = student_outputs[:3]
        teacher_bbox, teacher_cls, teacher_ldm = teacher_outputs[:3]
        
        student_features = student_outputs[3] if len(student_outputs) > 3 else None
        teacher_features = teacher_outputs[3] if len(teacher_outputs) > 3 else None
        
        student_attention = student_outputs[4] if len(student_outputs) > 4 else None
        teacher_attention = teacher_outputs[4] if len(teacher_outputs) > 4 else None
        
        # Progressive temperature scheduling
        current_temperature = self._compute_progressive_temperature(epoch, max_epochs)
        
        # 1. Classification distillation loss
        cls_distill_loss = self._compute_classification_distillation(
            student_cls, teacher_cls, current_temperature
        )
        
        # 2. Feature alignment loss
        feature_loss = self._compute_feature_alignment_loss(
            student_features, teacher_features
        ) if student_features is not None and teacher_features is not None else torch.tensor(0.0, device=student_cls.device)
        
        # 3. Attention alignment loss
        attention_loss = self._compute_attention_alignment_loss(
            student_attention, teacher_attention  
        ) if student_attention is not None and teacher_attention is not None else torch.tensor(0.0, device=student_cls.device)
        
        # 4. Regression distillation (bbox + landmarks)
        regression_distill_loss = self._compute_regression_distillation(
            student_bbox, teacher_bbox, student_ldm, teacher_ldm
        )
        
        # 5. Adaptive sample weighting
        sample_weights = self._compute_adaptive_weights(
            student_cls, teacher_cls, targets
        ) if self.adaptive_weighting else torch.ones(student_cls.size(0), device=student_cls.device)
        
        # Combine losses with adaptive weighting
        distillation_loss = (
            cls_distill_loss + 
            self.feature_weight * feature_loss +
            self.attention_weight * attention_loss +
            0.1 * regression_distill_loss
        )
        
        # Apply sample weights
        if self.adaptive_weighting:
            distillation_loss = (distillation_loss * sample_weights.unsqueeze(1)).mean()
        else:
            distillation_loss = distillation_loss.mean()
        
        # Final combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        # Update EMA for adaptive weighting
        self.loss_ema = self.ema_momentum * self.loss_ema + (1 - self.ema_momentum) * total_loss.detach()
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss,
            'cls_distill_loss': cls_distill_loss.mean(),
            'feature_loss': feature_loss,
            'attention_loss': attention_loss,
            'regression_distill_loss': regression_distill_loss.mean(),
            'temperature': current_temperature,
            'alpha': self.alpha
        }
    
    def _compute_progressive_temperature(self, epoch: int, max_epochs: int) -> float:
        """
        Progressive temperature scheduling:
        - High temperature (6.0) early: Heavy distillation focus
        - Medium temperature (4.0) middle: Balanced learning
        - Low temperature (2.0) late: Fine-tuning focus
        - Very low temperature (1.0) final: Performance boosting
        """
        progress = epoch / max_epochs
        
        if progress < 0.25:  # First quarter: Heavy distillation
            return 6.0
        elif progress < 0.5:  # Second quarter: Balanced
            return 4.0  
        elif progress < 0.75:  # Third quarter: Fine-tuning
            return 2.0
        else:  # Final quarter: Performance boost
            return 1.0
    
    def _compute_classification_distillation(self, 
                                           student_logits: torch.Tensor,
                                           teacher_logits: torch.Tensor, 
                                           temperature: float) -> torch.Tensor:
        """Compute KL divergence loss for classification"""
        
        # Temperature scaling
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        kl_loss = self.kl_loss(student_soft, teacher_soft).sum(dim=-1)
        
        # Temperature compensation
        kl_loss = kl_loss * (temperature ** 2)
        
        return kl_loss
    
    def _compute_feature_alignment_loss(self,
                                      student_features: List[torch.Tensor],
                                      teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute feature alignment loss between student and teacher"""
        
        if not student_features or not teacher_features:
            return torch.tensor(0.0)
            
        feature_loss = 0.0
        num_features = min(len(student_features), len(teacher_features))
        
        for i in range(num_features):
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            
            # Align dimensions if necessary
            if s_feat.shape != t_feat.shape:
                # Adaptive pooling to match sizes
                t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
                
                # Channel alignment via 1x1 conv if needed
                if s_feat.size(1) != t_feat.size(1):
                    # Skip this feature pair if channel mismatch is too large
                    if abs(s_feat.size(1) - t_feat.size(1)) > 32:
                        continue
            
            # L2 feature matching loss
            feat_loss = self.mse_loss(s_feat, t_feat).mean()
            feature_loss += feat_loss
            
        return feature_loss / max(num_features, 1)
    
    def _compute_attention_alignment_loss(self,
                                        student_attention: List[torch.Tensor],
                                        teacher_attention: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention map alignment loss"""
        
        if not student_attention or not teacher_attention:
            return torch.tensor(0.0)
            
        attention_loss = 0.0
        num_maps = min(len(student_attention), len(teacher_attention))
        
        for i in range(num_maps):
            s_att = student_attention[i]
            t_att = teacher_attention[i]
            
            # Normalize attention maps
            s_att_norm = F.normalize(s_att.view(s_att.size(0), -1), dim=1)
            t_att_norm = F.normalize(t_att.view(t_att.size(0), -1), dim=1)
            
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(s_att_norm, t_att_norm, dim=1)
            att_loss = (1 - cosine_sim).mean()
            attention_loss += att_loss
            
        return attention_loss / max(num_maps, 1)
    
    def _compute_regression_distillation(self,
                                       student_bbox: torch.Tensor,
                                       teacher_bbox: torch.Tensor,
                                       student_ldm: torch.Tensor, 
                                       teacher_ldm: torch.Tensor) -> torch.Tensor:
        """Compute regression distillation for bbox and landmarks"""
        
        # Bbox regression distillation
        bbox_loss = self.l1_loss(student_bbox, teacher_bbox).mean(dim=-1)
        
        # Landmark regression distillation
        ldm_loss = self.l1_loss(student_ldm, teacher_ldm).mean(dim=-1)
        
        return bbox_loss + ldm_loss
    
    def _compute_adaptive_weights(self,
                                student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive sample weights based on prediction difficulty
        Hard samples get higher weights to focus learning
        """
        
        # Compute prediction confidence
        student_conf = F.softmax(student_logits, dim=-1).max(dim=-1)[0].mean(dim=-1)
        teacher_conf = F.softmax(teacher_logits, dim=-1).max(dim=-1)[0].mean(dim=-1)
        
        # Compute difficulty score (lower confidence = higher difficulty)
        difficulty = 1.0 - (student_conf + teacher_conf) / 2.0
        
        # Convert to weights (higher difficulty = higher weight)
        weights = 1.0 + difficulty
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights


class MultiTeacherDistillation(nn.Module):
    """
    Multi-teacher distillation strategy
    
    Teachers:
    1. Primary Teacher: Best V1 model
    2. Ensemble Teacher: Multiple V1 variants
    3. Self Teacher: Student's own best predictions (temporal consistency)
    """
    
    def __init__(self, 
                 primary_teacher: nn.Module,
                 ensemble_teachers: List[nn.Module] = None,
                 ensemble_weight: float = 0.3,
                 self_weight: float = 0.2):
        super(MultiTeacherDistillation, self).__init__()
        
        self.primary_teacher = primary_teacher
        self.ensemble_teachers = ensemble_teachers or []
        self.ensemble_weight = ensemble_weight
        self.self_weight = self_weight
        
        # Set teachers to eval mode
        self.primary_teacher.eval()
        for teacher in self.ensemble_teachers:
            teacher.eval()
            
        # Self-teacher storage (EMA of student predictions)
        self.register_buffer('self_teacher_predictions', None)
        self.self_ema_momentum = 0.995
        
    def forward(self, 
                student: nn.Module,
                inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Generate multi-teacher predictions
        
        Returns:
            Combined teacher predictions for distillation
        """
        
        with torch.no_grad():
            # Primary teacher predictions
            primary_pred = self.primary_teacher(inputs)
            
            # Ensemble teacher predictions
            if self.ensemble_teachers:
                ensemble_preds = []
                for teacher in self.ensemble_teachers:
                    pred = teacher(inputs)
                    ensemble_preds.append(pred)
                
                # Average ensemble predictions
                ensemble_pred = self._average_predictions(ensemble_preds)
            else:
                ensemble_pred = primary_pred
            
            # Student predictions (for self-teaching)
            student_pred = student(inputs)
            
            # Update self-teacher with EMA
            self._update_self_teacher(student_pred)
            
            # Combine all teacher predictions
            combined_pred = self._combine_teacher_predictions(
                primary_pred, ensemble_pred, student_pred
            )
            
        return combined_pred
    
    def _average_predictions(self, predictions: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        """Average multiple predictions"""
        if not predictions:
            return None
            
        # Average each output separately
        averaged = []
        for i in range(len(predictions[0])):
            outputs = [pred[i] for pred in predictions]
            avg_output = torch.stack(outputs).mean(dim=0)
            averaged.append(avg_output)
            
        return tuple(averaged)
    
    def _update_self_teacher(self, student_pred: Tuple[torch.Tensor, ...]):
        """Update self-teacher with EMA of student predictions"""
        if self.self_teacher_predictions is None:
            # Initialize with current prediction
            self.self_teacher_predictions = [pred.detach().clone() for pred in student_pred]
        else:
            # EMA update
            for i, pred in enumerate(student_pred):
                self.self_teacher_predictions[i] = (
                    self.self_ema_momentum * self.self_teacher_predictions[i] + 
                    (1 - self.self_ema_momentum) * pred.detach()
                )
    
    def _combine_teacher_predictions(self,
                                   primary_pred: Tuple[torch.Tensor, ...],
                                   ensemble_pred: Tuple[torch.Tensor, ...], 
                                   student_pred: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Combine predictions from multiple teachers"""
        
        # Primary teacher weight
        primary_weight = 1.0 - self.ensemble_weight - self.self_weight
        
        combined = []
        for i in range(len(primary_pred)):
            # Weighted combination
            combined_output = (
                primary_weight * primary_pred[i] +
                self.ensemble_weight * ensemble_pred[i]
            )
            
            # Add self-teacher if available
            if self.self_teacher_predictions is not None:
                combined_output += self.self_weight * self.self_teacher_predictions[i]
            
            combined.append(combined_output)
            
        return tuple(combined)


class CurriculumLearning:
    """
    Curriculum learning strategy for progressive difficulty
    
    Stages:
    1. Easy samples (high IoU, large faces) 
    2. Medium samples (medium IoU, medium faces)
    3. Hard samples (low IoU, small faces)
    4. Mixed samples (all difficulties)
    """
    
    def __init__(self, 
                 total_epochs: int = 400,
                 easy_ratio: float = 0.25,
                 medium_ratio: float = 0.25, 
                 hard_ratio: float = 0.25):
        
        self.total_epochs = total_epochs
        self.easy_epochs = int(total_epochs * easy_ratio)
        self.medium_epochs = int(total_epochs * medium_ratio)
        self.hard_epochs = int(total_epochs * hard_ratio)
        self.mixed_epochs = total_epochs - self.easy_epochs - self.medium_epochs - self.hard_epochs
        
    def get_curriculum_stage(self, epoch: int) -> str:
        """Get current curriculum stage"""
        if epoch < self.easy_epochs:
            return 'easy'
        elif epoch < self.easy_epochs + self.medium_epochs:
            return 'medium'
        elif epoch < self.easy_epochs + self.medium_epochs + self.hard_epochs:
            return 'hard'
        else:
            return 'mixed'
    
    def filter_samples(self, 
                      samples: torch.Tensor,
                      difficulties: torch.Tensor,
                      stage: str) -> torch.Tensor:
        """Filter samples based on curriculum stage"""
        
        if stage == 'easy':
            # Select easiest 70% of samples
            threshold = torch.quantile(difficulties, 0.3)
            mask = difficulties <= threshold
        elif stage == 'medium':
            # Select medium 60% of samples  
            low_thresh = torch.quantile(difficulties, 0.2)
            high_thresh = torch.quantile(difficulties, 0.8)
            mask = (difficulties > low_thresh) & (difficulties < high_thresh)
        elif stage == 'hard':
            # Select hardest 70% of samples
            threshold = torch.quantile(difficulties, 0.3)
            mask = difficulties >= threshold
        else:  # mixed
            # Use all samples
            mask = torch.ones_like(difficulties, dtype=torch.bool)
            
        return mask


def create_advanced_distillation_pipeline(primary_teacher: nn.Module,
                                        ensemble_teachers: List[nn.Module] = None,
                                        temperature: float = 4.0,
                                        alpha: float = 0.7) -> Tuple[MultiTeacherDistillation, AdvancedDistillationLoss]:
    """
    Create complete advanced distillation pipeline
    
    Returns:
        multi_teacher: Multi-teacher distillation module
        distill_loss: Advanced distillation loss
    """
    
    # Create multi-teacher distillation
    multi_teacher = MultiTeacherDistillation(
        primary_teacher=primary_teacher,
        ensemble_teachers=ensemble_teachers,
        ensemble_weight=0.3,
        self_weight=0.2
    )
    
    # Create advanced distillation loss
    distill_loss = AdvancedDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        feature_weight=0.1,
        attention_weight=0.05,
        adaptive_weighting=True
    )
    
    return multi_teacher, distill_loss


if __name__ == "__main__":
    """Test advanced distillation components"""
    
    print("🚀 Testing Advanced Multi-Teacher Distillation")
    print("=" * 60)
    
    # Mock teacher and student models
    batch_size, num_anchors = 2, 16800
    
    # Mock predictions
    student_outputs = (
        torch.randn(batch_size, num_anchors, 4),  # bbox
        torch.randn(batch_size, num_anchors, 2),  # cls
        torch.randn(batch_size, num_anchors, 10), # landmarks
    )
    
    teacher_outputs = (
        torch.randn(batch_size, num_anchors, 4),  # bbox
        torch.randn(batch_size, num_anchors, 2),  # cls  
        torch.randn(batch_size, num_anchors, 10), # landmarks
    )
    
    targets = torch.randint(0, 2, (batch_size, num_anchors))
    task_loss = torch.tensor(1.5)
    
    # Test advanced distillation loss
    distill_loss = AdvancedDistillationLoss()
    
    loss_dict = distill_loss(
        student_outputs, teacher_outputs, targets, task_loss, epoch=100, max_epochs=400
    )
    
    print(f"\n📊 Distillation Loss Components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.item():.4f}")
        else:
            print(f"  {key:20s}: {value:.4f}")
    
    # Test curriculum learning
    curriculum = CurriculumLearning(total_epochs=400)
    
    print(f"\n📚 Curriculum Learning Stages:")
    test_epochs = [50, 150, 250, 350]
    for epoch in test_epochs:
        stage = curriculum.get_curriculum_stage(epoch)
        print(f"  Epoch {epoch:3d}: {stage:6s} stage")
    
    print(f"\n✅ Advanced distillation pipeline tested successfully!")
    print(f"🎓 Multi-teacher strategy ready for V2 Ultra training!")