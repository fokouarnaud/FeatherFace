#!/usr/bin/env python3
"""
FeatherFace Nano-B: Ultra-Lightweight Face Detection with Bayesian-Optimized Pruning

This module implements FeatherFace Nano-B, combining:
1. FeatherFace Nano efficient architecture (344K -> ~120-180K parameters)
2. B-FPGM Bayesian-optimized structured pruning
3. Weighted Knowledge Distillation for edge deployment
4. Small face detection enhancements (2024 research)

Scientific Foundation:
1. FeatherFace Nano: Research-backed efficiency (Li et al. CVPR 2023, Woo et al. ECCV 2018, etc.)
2. B-FPGM: Kaparinos & Mezaris, WACVW 2025
3. Weighted Knowledge Distillation: Various 2025 research
4. Small Face Detection Enhancements (2024):
   - ASSN: "Attention-based scale sequence network for small object detection" (PMC/ScienceDirect)
   - MSE-FPN: "Multi-scale semantic enhancement network for object detection" (Scientific Reports)
   - Scale Decoupling: SNLA approach for P3 optimization

Enhanced Architecture:
- P3 Level: ScaleDecoupling + ASSN (optimized for small faces)
- P4/P5 Levels: Standard CBAM (Woo et al. ECCV 2018)
- Feature Fusion: SemanticEnhancement modules (MSE-FPN 2024)
- Pipeline: Scale-aware processing with research-backed optimizations

Target: 120-180K parameters (+5-8K for small face optimizations) with +15-20% small face detection improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import available modules and create necessary components
try:
    # Use scientifically validated modules from net.py
    from .net import MobileNetV1, CBAM, BiFPN, SSH
    # Import V2 modules for additional functionality
    from .modules_v2 import (SSH_Grouped, ChannelShuffle_Light, DepthwiseSeparableConv)
    # Import pruning if available
    try:
        from .pruning_b_fpgm import FeatherFaceNanoBPruner, create_nano_b_config
        PRUNING_AVAILABLE = True
    except ImportError:
        PRUNING_AVAILABLE = False
        logger.warning("Pruning modules not available - using simplified mode")
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    # Use scientifically validated modules from net.py
    from net import MobileNetV1, CBAM, BiFPN, SSH
    # Import V2 modules for additional functionality
    from modules_v2 import (SSH_Grouped, ChannelShuffle_Light, DepthwiseSeparableConv)
    try:
        from pruning_b_fpgm import FeatherFaceNanoBPruner, create_nano_b_config
        PRUNING_AVAILABLE = True
    except ImportError:
        PRUNING_AVAILABLE = False

# Use scientifically validated modules with proper research references
# CBAM: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018
# BiFPN: Tan et al. "EfficientDet: Scalable and Efficient Object Detection" CVPR 2020
# SSH: Najibi et al. "SSH: Single Stage Headless Face Detector" ICCV 2017
StandardCBAM = CBAM
StandardBiFPN = BiFPN
StandardSSH = SSH  # Use scientifically validated SSH + optimization techniques
ChannelShuffle = ChannelShuffle_Light


def create_nano_b_config_simple(target_reduction=0.4):
    """Simple config creation when full pruning not available"""
    return {
        'target_reduction': target_reduction,
        'layer_groups': ['backbone', 'attention', 'detection'],
        'search_space': [(0.0, 0.6) for _ in range(3)]
    }


class WeightedKnowledgeDistillation(nn.Module):
    """
    Weighted Knowledge Distillation for edge deployment
    
    Based on 2025 research: "Crowd counting at the edge using weighted knowledge distillation"
    Adapts distillation weights based on task importance and computational constraints.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, 
                 adaptive_weights: bool = True):
        """
        Initialize weighted knowledge distillation
        
        Args:
            temperature: Distillation temperature
            alpha: Balance between distillation and task loss
            adaptive_weights: Enable adaptive weight adjustment
        """
        super(WeightedKnowledgeDistillation, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.adaptive_weights = adaptive_weights
        
        # Learnable weights for different output types
        if adaptive_weights:
            self.cls_weight = nn.Parameter(torch.tensor(1.0))
            self.bbox_weight = nn.Parameter(torch.tensor(1.0))
            self.landmark_weight = nn.Parameter(torch.tensor(0.8))  # Less critical for pruning
        else:
            self.cls_weight = 1.0
            self.bbox_weight = 1.0
            self.landmark_weight = 0.8
    
    def forward(self, student_outputs: List[torch.Tensor], 
                teacher_outputs: List[torch.Tensor],
                targets: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute weighted distillation loss
        
        Args:
            student_outputs: [cls_pred, bbox_pred, landmark_pred] from student
            teacher_outputs: [cls_pred, bbox_pred, landmark_pred] from teacher
            targets: Ground truth targets (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Classification distillation
        cls_student, bbox_student, landmark_student = student_outputs
        cls_teacher, bbox_teacher, landmark_teacher = teacher_outputs
        
        # Soft targets from teacher
        cls_teacher_soft = F.softmax(cls_teacher / self.temperature, dim=1)
        cls_student_soft = F.log_softmax(cls_student / self.temperature, dim=1)
        
        # Weighted distillation losses
        cls_distill_loss = F.kl_div(cls_student_soft, cls_teacher_soft, 
                                   reduction='batchmean') * (self.temperature ** 2)
        
        bbox_distill_loss = F.mse_loss(bbox_student, bbox_teacher)
        landmark_distill_loss = F.mse_loss(landmark_student, landmark_teacher)
        
        # Apply weights
        if self.adaptive_weights:
            weighted_cls_loss = self.cls_weight * cls_distill_loss
            weighted_bbox_loss = self.bbox_weight * bbox_distill_loss
            weighted_landmark_loss = self.landmark_weight * landmark_distill_loss
        else:
            weighted_cls_loss = self.cls_weight * cls_distill_loss
            weighted_bbox_loss = self.bbox_weight * bbox_distill_loss
            weighted_landmark_loss = self.landmark_weight * landmark_distill_loss
        
        # Total distillation loss
        total_distill_loss = weighted_cls_loss + weighted_bbox_loss + weighted_landmark_loss
        
        losses.update({
            'distill_cls': weighted_cls_loss,
            'distill_bbox': weighted_bbox_loss,
            'distill_landmark': weighted_landmark_loss,
            'distill_total': total_distill_loss
        })
        
        # Task-specific losses if targets provided
        if targets is not None:
            cls_targets, bbox_targets, landmark_targets = targets
            
            task_cls_loss = F.cross_entropy(cls_student, cls_targets)
            task_bbox_loss = F.smooth_l1_loss(bbox_student, bbox_targets)
            task_landmark_loss = F.smooth_l1_loss(landmark_student, landmark_targets)
            
            total_task_loss = task_cls_loss + task_bbox_loss + task_landmark_loss
            
            # Combined loss
            combined_loss = self.alpha * total_distill_loss + (1 - self.alpha) * total_task_loss
            
            losses.update({
                'task_cls': task_cls_loss,
                'task_bbox': task_bbox_loss,
                'task_landmark': task_landmark_loss,
                'task_total': total_task_loss,
                'combined': combined_loss
            })
        
        return losses


class ScaleSequenceAttention(nn.Module):
    """
    Scale Sequence Attention for small object detection
    
    Based on "Attention-based scale sequence network for small object detection" (2024)
    PMC/ScienceDirect research - optimized for P3 level small face detection
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize Scale Sequence Attention
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for efficiency
        """
        super(ScaleSequenceAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Scale sequence generation optimized for small objects
        reduced_channels = max(in_channels // reduction, 4)
        
        self.scale_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context for scale awareness
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial enhancement for small object details
        self.spatial_enhance = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scale sequence attention
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Enhanced features for small object detection
        """
        # Scale sequence attention (channel-wise)
        scale_attention = self.scale_sequence(x)
        x_scale = x * scale_attention
        
        # Spatial attention for small object details
        avg_out = torch.mean(x_scale, dim=1, keepdim=True)
        max_out, _ = torch.max(x_scale, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_enhance(spatial_input)
        
        # Combined enhancement
        x_enhanced = x_scale * spatial_attention
        
        return x_enhanced


class SemanticEnhancementModule(nn.Module):
    """
    Semantic Enhancement Module for multi-scale feature fusion
    
    Based on "Multi-scale semantic enhancement network for object detection" (2024)
    Scientific Reports - resolves semantic gap between features of various sizes
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize Semantic Enhancement Module
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super(SemanticEnhancementModule, self).__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 16)
        
        # Semantic injection module
        self.semantic_injection = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Gated channel guidance module
        self.gated_channel_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with semantic enhancement
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Semantically enhanced features
        """
        # Semantic injection for feature quality
        semantic_features = self.semantic_injection(x)
        
        # Gated channel guidance for importance weighting
        channel_gate = self.gated_channel_guidance(x)
        gated_features = semantic_features * channel_gate
        
        # Residual connection with refinement
        refined_features = self.feature_refine(gated_features + x)
        
        return refined_features


class ScaleDecouplingModule(nn.Module):
    """
    Scale Decoupling Module for P3 level small face enhancement
    
    Based on SNLA approach - eliminates large object features in shallow layers
    Specifically designed for small face detection optimization
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Initialize Scale Decoupling Module
        
        Args:
            channels: Number of input channels
            kernel_size: Convolution kernel size
        """
        super(ScaleDecouplingModule, self).__init__()
        self.channels = channels
        
        # Small object feature enhancer
        self.small_object_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Large object feature suppressor
        self.large_object_suppressor = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Feature balance
        self.feature_balance = nn.Parameter(torch.tensor(0.8))  # Learnable balance
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scale decoupling for small faces
        
        Args:
            x: Input features [B, C, H, W] - P3 level features
            
        Returns:
            Decoupled features optimized for small face detection
        """
        # Enhance small object features
        enhanced_small = self.small_object_enhancer(x)
        
        # Generate large object suppression mask
        suppress_mask = self.large_object_suppressor(x)
        
        # Apply scale decoupling: enhance small, suppress large
        # Balance factor learned during training
        decoupled_features = enhanced_small * (1 - suppress_mask * self.feature_balance)
        
        return decoupled_features


class PrunedConv2d(nn.Module):
    """
    Convolutional layer with built-in pruning support
    
    Supports both soft and hard pruning with importance tracking
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 pruning_enabled: bool = True):
        """
        Initialize pruned conv layer
        
        Args:
            Standard conv2d arguments plus:
            pruning_enabled: Enable pruning support
        """
        super(PrunedConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=bias)
        self.pruning_enabled = pruning_enabled
        
        if pruning_enabled:
            # Pruning mask (1 = keep, 0 = prune)
            self.register_buffer('pruning_mask', torch.ones(out_channels))
            
            # Importance scores (updated during training)
            self.register_buffer('importance_scores', torch.ones(out_channels))
            
            # Soft pruning temperature
            self.soft_temperature = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional pruning"""
        if self.pruning_enabled and hasattr(self, 'pruning_mask'):
            # Apply pruning mask
            if self.pruning_mask.sum() < self.conv.out_channels:
                # Soft pruning during training
                if self.training:
                    mask = torch.sigmoid(self.pruning_mask / self.soft_temperature)
                    mask = mask.view(-1, 1, 1, 1)
                    pruned_weight = self.conv.weight * mask
                    
                    # Use functional conv with pruned weights
                    return F.conv2d(x, pruned_weight, self.conv.bias,
                                   self.conv.stride, self.conv.padding)
                else:
                    # Hard pruning during inference
                    active_filters = self.pruning_mask.bool()
                    if active_filters.sum() > 0:
                        pruned_weight = self.conv.weight[active_filters]
                        pruned_bias = self.conv.bias[active_filters] if self.conv.bias is not None else None
                        
                        return F.conv2d(x, pruned_weight, pruned_bias,
                                       self.conv.stride, self.conv.padding)
                    else:
                        # All filters pruned - return zeros
                        batch_size = x.size(0)
                        h_out = (x.size(2) + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) // self.conv.stride[0] + 1
                        w_out = (x.size(3) + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
                        return torch.zeros(batch_size, 1, h_out, w_out, device=x.device)
        
        return self.conv(x)
    
    def update_importance(self, gradient_info: torch.Tensor):
        """Update filter importance based on gradients"""
        if self.pruning_enabled:
            # Simple gradient-based importance (can be enhanced)
            self.importance_scores = gradient_info.abs().sum(dim=[1, 2, 3])
    
    def apply_soft_pruning(self, sparsity: float, temperature: float = 1.0):
        """Apply soft pruning with given sparsity"""
        if self.pruning_enabled:
            num_filters = self.conv.out_channels
            num_to_prune = int(num_filters * sparsity)
            
            if num_to_prune > 0:
                # Get least important filters
                _, indices = self.importance_scores.sort()
                filters_to_prune = indices[:num_to_prune]
                
                # Update mask for soft pruning
                self.pruning_mask[filters_to_prune] = 0.1  # Soft pruning value
                self.soft_temperature = temperature


class FeatherFaceNanoB(nn.Module):
    """
    FeatherFace Nano-B: Ultra-lightweight face detection with Bayesian-optimized pruning
    
    Architecture progression:
    1. Start with FeatherFace Nano (344K parameters)
    2. Apply Bayesian-optimized structured pruning
    3. Use weighted knowledge distillation
    4. Target: 120-180K parameters with maintained accuracy
    """
    
    def __init__(self, cfg: Dict = None, phase: str = 'train', 
                 pruning_config: Dict = None, use_pruned_conv: bool = True):
        """
        Initialize FeatherFace Nano-B
        
        Args:
            cfg: Model configuration
            phase: 'train' or 'test'
            pruning_config: Pruning configuration
            use_pruned_conv: Use pruning-aware conv layers
        """
        super(FeatherFaceNanoB, self).__init__()
        
        self.phase = phase
        self.use_pruned_conv = use_pruned_conv
        
        # Default configurations
        if cfg is None:
            # Use simple default config if data.config not available
            cfg = {
                'name': 'Nano-B',
                'out_channel': 32,
                'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
                'in_channel': 32,
                'bifpn_channels': 72,
                'cbam_reduction': 8,
                'assn_reduction': 16,
                'ssh_groups': 2,
                'num_classes': 2,
                'distillation_temperature': 4.0,
                'distillation_alpha': 0.7
            }
        
        if pruning_config is None:
            if PRUNING_AVAILABLE:
                pruning_config = create_nano_b_config(target_reduction=0.4)
            else:
                pruning_config = create_nano_b_config_simple(target_reduction=0.4)
        
        self.cfg = cfg
        self.pruning_config = pruning_config
        
        # Initialize base Nano architecture with pruning support
        self._build_nano_b_architecture()
        
        # Initialize pruning components
        self.pruner = None  # Will be set during pruning phase
        
        # Weighted knowledge distillation
        self.distillation = WeightedKnowledgeDistillation(
            temperature=cfg.get('distillation_temperature', 4.0),
            alpha=cfg.get('distillation_alpha', 0.7),
            adaptive_weights=True
        )
        
        # Performance tracking
        self.pruning_stats = {
            'original_params': 0,
            'current_params': 0,
            'reduction_percent': 0.0,
            'pruning_history': []
        }
        
        self._count_parameters()
    
    def _build_nano_b_architecture(self):
        """Build FeatherFace Nano-B architecture with pruning support"""
        
        # MobileNet-0.25 backbone
        backbone = MobileNetV1()
        return_layers = self.cfg['return_layers']
        
        # Create standard backbone (disable pruning-aware for testing)
        from torchvision.models._utils import IntermediateLayerGetter
        self.body = IntermediateLayerGetter(backbone, return_layers)
        
        # Efficient modules with pruning support
        # Actual MobileNet-0.25 output channels
        in_channels_list = [64, 128, 256]  # P3, P4, P5 real channels
        
        # Small face detection enhancement (2024 research-based)
        # P3 level optimization for small face detection
        self.scale_decoupling_p3 = ScaleDecouplingModule(
            channels=in_channels_list[0],  # P3: 32 channels
            kernel_size=3
        )
        
        # First CBAM attention - P3 gets special treatment
        cbam_reduction = self.cfg.get('cbam_reduction', 8)
        self.cbam1 = nn.ModuleList([
            # P3: Use standard CBAM (will be enhanced with ASSN later)
            StandardCBAM(in_channels_list[0], reduction_ratio=cbam_reduction),
            # P4, P5: Standard CBAM (Woo et al. ECCV 2018)
            StandardCBAM(in_channels_list[1], reduction_ratio=cbam_reduction),
            StandardCBAM(in_channels_list[2], reduction_ratio=cbam_reduction)
        ])
        
        # BiFPN with semantic enhancement (Tan et al. CVPR 2020)
        # Ensure bifpn_channels is divisible by 4 for SSH standard
        bifpn_channels = self.cfg.get('bifpn_channels', 72)  # Changed from 74 to 72
        self.bifpn = StandardBiFPN(
            num_channels=bifpn_channels,
            conv_channels=in_channels_list,
            first_time=True,
            onnx_export=False,
            attention=True
        )
        
        # Semantic enhancement modules for better feature fusion (MSE-FPN 2024)
        self.semantic_enhancement = nn.ModuleList([
            SemanticEnhancementModule(bifpn_channels, reduction=4) for _ in range(3)
        ])
        
        # Second attention - P3 gets Scale Sequence Attention (ASSN 2024)
        self.cbam2_p4p5 = nn.ModuleList([
            # P4, P5: Standard CBAM (Woo et al. ECCV 2018)
            StandardCBAM(bifpn_channels, reduction_ratio=cbam_reduction),
            StandardCBAM(bifpn_channels, reduction_ratio=cbam_reduction)
        ])
        
        # P3: Scale Sequence Attention for small face detection (ASSN 2024)
        self.assn_p3 = ScaleSequenceAttention(
            in_channels=bifpn_channels,
            reduction=self.cfg.get('assn_reduction', 16)
        )
        
        # SSH detection heads (Najibi et al. ICCV 2017) with optimization techniques
        self.ssh_heads = nn.ModuleList([
            StandardSSH(
                in_channel=bifpn_channels,
                out_channel=bifpn_channels
            )
            for _ in range(3)
        ])
        
        # Channel shuffle for parameter-free mixing
        self.channel_shuffle = ChannelShuffle(channels=bifpn_channels, groups=2)
        
        # Final detection heads
        self._make_detection_heads(bifpn_channels)
    
    def _make_pruning_aware_backbone(self, backbone, return_layers):
        """Create pruning-aware version of backbone"""
        # Replace standard conv with pruning-aware conv
        def replace_conv_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    # Replace with PrunedConv2d
                    pruned_conv = PrunedConv2d(
                        child.in_channels, child.out_channels, child.kernel_size[0],
                        child.stride[0], child.padding[0], child.bias is not None,
                        pruning_enabled=True
                    )
                    # Copy weights
                    pruned_conv.conv.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        pruned_conv.conv.bias.data = child.bias.data.clone()
                    
                    setattr(module, name, pruned_conv)
                else:
                    replace_conv_recursive(child)
        
        replace_conv_recursive(backbone)
        
        # Create intermediate layer getter
        from torchvision.models._utils import IntermediateLayerGetter
        return IntermediateLayerGetter(backbone, return_layers)
    
    def _make_detection_heads(self, in_channels: int):
        """Create detection heads with pruning support"""
        
        # Classification head
        if self.use_pruned_conv:
            self.ClassHead = nn.Sequential(
                PrunedConv2d(in_channels, in_channels//2, 3, padding=1, pruning_enabled=True),
                nn.ReLU(inplace=True),
                PrunedConv2d(in_channels//2, self.cfg['num_classes'], 1, pruning_enabled=True)
            )
        else:
            self.ClassHead = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, self.cfg['num_classes'], 1)
            )
        
        # Bounding box regression head
        if self.use_pruned_conv:
            self.BboxHead = nn.Sequential(
                PrunedConv2d(in_channels, in_channels//2, 3, padding=1, pruning_enabled=True),
                nn.ReLU(inplace=True),
                PrunedConv2d(in_channels//2, 4, 1, pruning_enabled=True)
            )
        else:
            self.BboxHead = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, 4, 1)
            )
        
        # Landmark head
        if self.use_pruned_conv:
            self.LandmarkHead = nn.Sequential(
                PrunedConv2d(in_channels, in_channels//2, 3, padding=1, pruning_enabled=True),
                nn.ReLU(inplace=True),
                PrunedConv2d(in_channels//2, 10, 1, pruning_enabled=True)
            )
        else:
            self.LandmarkHead = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, 10, 1)
            )
    
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through FeatherFace Nano-B
        
        Args:
            inputs: Input images [B, 3, H, W]
            
        Returns:
            [classifications, bbox_regressions, landmarks] for each scale
        """
        # Extract multi-scale features via backbone
        features = self.body(inputs)
        feature_list = list(features.values())
        
        # P3 level enhancement for small face detection (2024 research)
        # Apply scale decoupling to P3 before any other processing
        p3_features = self.scale_decoupling_p3(feature_list[0])
        enhanced_feature_list = [p3_features, feature_list[1], feature_list[2]]
        
        # First CBAM attention with enhanced P3
        attended_features = []
        for i, (feat, cbam) in enumerate(zip(enhanced_feature_list, self.cbam1)):
            attended_feat = cbam(feat)
            attended_features.append(attended_feat)
        
        # BiFPN feature fusion
        fused_features = self.bifpn(attended_features)
        
        # Semantic enhancement for improved feature fusion (MSE-FPN 2024)
        semantically_enhanced = []
        for feat, sem_enhance in zip(fused_features, self.semantic_enhancement):
            enhanced_feat = sem_enhance(feat)
            semantically_enhanced.append(enhanced_feat)
        
        # Second attention with specialized processing for P3
        refined_features = []
        
        # P3: Use Scale Sequence Attention for small face detection (ASSN 2024)
        p3_refined = self.assn_p3(semantically_enhanced[0])
        refined_features.append(p3_refined)
        
        # P4, P5: Use standard efficient CBAM
        for i, (feat, cbam) in enumerate(zip(semantically_enhanced[1:], self.cbam2_p4p5)):
            refined_feat = cbam(feat)
            refined_features.append(refined_feat)
        
        # SSH processing
        ssh_features = []
        for feat, ssh in zip(refined_features, self.ssh_heads):
            ssh_feat = ssh(feat)
            ssh_features.append(ssh_feat)
        
        # Channel shuffle for parameter-free mixing
        shuffled_features = [self.channel_shuffle(feat) for feat in ssh_features]
        
        # Detection heads
        classifications = []
        bbox_regressions = []
        landmarks = []
        
        for feat in shuffled_features:
            cls_out = self.ClassHead(feat)
            bbox_out = self.BboxHead(feat)
            landmark_out = self.LandmarkHead(feat)
            
            # Reshape for output
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            bbox_out = bbox_out.permute(0, 2, 3, 1).contiguous()
            landmark_out = landmark_out.permute(0, 2, 3, 1).contiguous()
            
            classifications.append(cls_out.view(cls_out.shape[0], -1, self.cfg['num_classes']))
            bbox_regressions.append(bbox_out.view(bbox_out.shape[0], -1, 4))
            landmarks.append(landmark_out.view(landmark_out.shape[0], -1, 10))
        
        # Concatenate all scales
        classifications = torch.cat(classifications, dim=1)
        bbox_regressions = torch.cat(bbox_regressions, dim=1)
        landmarks = torch.cat(landmarks, dim=1)
        
        if self.phase == 'train':
            return [classifications, bbox_regressions, landmarks]
        else:
            return [F.softmax(classifications, dim=-1), bbox_regressions, landmarks]
    
    def setup_pruning(self, validation_loader, criterion):
        """Setup Bayesian optimization for pruning"""
        if not PRUNING_AVAILABLE:
            logger.warning("Pruning not available - using simplified mode")
            return None
            
        if self.pruner is None:
            self.pruner = FeatherFaceNanoBPruner(self, self.pruning_config)
            self.pruner.compute_layer_importances()
        
        return self.pruner
    
    def optimize_pruning_rates(self, validation_loader, criterion, num_iterations: int = 20):
        """Find optimal pruning rates using Bayesian optimization"""
        if self.pruner is None:
            self.setup_pruning(validation_loader, criterion)
        
        optimal_rates = self.pruner.optimize_pruning_rates(
            validation_loader, criterion, num_iterations
        )
        
        return optimal_rates
    
    def apply_pruning(self, optimal_rates: Dict[str, float]):
        """Apply final structured pruning"""
        if self.pruner is None:
            raise ValueError("Must setup pruning first")
        
        pruning_results = self.pruner.apply_final_pruning(optimal_rates)
        
        # Update stats
        self.pruning_stats.update(pruning_results)
        self.pruning_stats['pruning_history'].append({
            'rates': optimal_rates,
            'results': pruning_results
        })
        
        self._count_parameters()
        
        return pruning_results
    
    def compute_distillation_loss(self, student_outputs, teacher_outputs, targets=None):
        """Compute weighted knowledge distillation loss"""
        return self.distillation(student_outputs, teacher_outputs, targets)
    
    def _count_parameters(self):
        """Count and update parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        if self.pruning_stats['original_params'] == 0:
            self.pruning_stats['original_params'] = total_params
        
        self.pruning_stats['current_params'] = total_params
        
        if self.pruning_stats['original_params'] > 0:
            reduction = (self.pruning_stats['original_params'] - total_params) / self.pruning_stats['original_params']
            self.pruning_stats['reduction_percent'] = reduction * 100
    
    def get_pruning_summary(self) -> Dict:
        """Get summary of pruning statistics"""
        return {
            'original_parameters': self.pruning_stats['original_params'],
            'current_parameters': self.pruning_stats['current_params'],
            'reduction_percent': self.pruning_stats['reduction_percent'],
            'target_range': '120K-180K parameters',
            'pruning_method': 'Bayesian-Optimized B-FPGM',
            'scientific_foundation': 'WACVW 2025 + Knowledge Distillation'
        }


def create_featherface_nano_b(cfg=None, phase='train', pruning_config=None, use_pruned_conv=False):
    """
    Factory function to create FeatherFace Nano-B model
    
    Args:
        cfg: Model configuration
        phase: 'train' or 'test'  
        pruning_config: Pruning configuration
        use_pruned_conv: Enable pruning-aware convolutions
        
    Returns:
        FeatherFace Nano-B model
    """
    return FeatherFaceNanoB(cfg=cfg, phase=phase, pruning_config=pruning_config, use_pruned_conv=use_pruned_conv)


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    print("Creating FeatherFace Nano-B model...")
    
    # Create configuration
    from data.config import cfg_nano
    pruning_config = create_nano_b_config(target_reduction=0.4)
    
    # Create model
    model = create_featherface_nano_b(
        cfg=cfg_nano,
        phase='train',
        pruning_config=pruning_config
    )
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shapes: {[out.shape for out in outputs]}")
    print(f"Parameters: {model.pruning_stats['current_params']:,}")
    print("\nPruning Summary:")
    summary = model.get_pruning_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")