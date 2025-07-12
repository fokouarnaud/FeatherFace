"""
TOOD: Task-Aligned One-stage Object Detection for Face Detection
================================================================

Implementation of TOOD (Task-aligned One-stage Object Detection) specifically 
adapted for face detection with 3 tasks: classification, bounding box regression, 
and facial landmark localization.

Scientific Foundation: 
- Original Paper: "TOOD: Task-aligned One-stage Object Detection" (ICCV 2021)
- Authors: Chenchen Feng, Yabiao Wang, et al.
- arXiv: 2108.07755

Adaptation for Face Detection:
- 3 tasks instead of 2: classification + bbox + landmarks
- Task Alignment Learning (TAL) for better sample assignment
- Optimized for mobile face detection deployment

Key Innovations:
1. Task-aligned Head (T-Head): Better balance between task-interactive and task-specific features
2. Task Alignment Learning (TAL): Explicit alignment of classification and localization tasks
3. Face-specific optimization: 3-task alignment for face detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TaskAlignedHead(nn.Module):
    """
    Task-Aligned Head for Face Detection
    
    Implements the T-Head from TOOD paper, adapted for 3-task face detection:
    1. Classification (face/background)
    2. Bounding box regression (x, y, w, h)
    3. Facial landmark regression (5 landmarks = 10 coordinates)
    
    Key Features:
    - Separate branches for each task with shared feature interaction
    - Task-aligned predictors for better feature learning
    - Learnable task alignment through attention mechanisms
    
    Args:
        in_channels (int): Number of input feature channels
        num_classes (int): Number of classes (2 for face detection: face/background)
        num_anchors (int): Number of anchors per location (default: 2)
        num_landmarks (int): Number of landmark coordinates (default: 10 for 5 landmarks)
        shared_conv_layers (int): Number of shared convolution layers (default: 4)
    """
    
    def __init__(self, in_channels=256, num_classes=2, num_anchors=2, num_landmarks=10, shared_conv_layers=4):
        super(TaskAlignedHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_landmarks = num_landmarks
        self.shared_conv_layers = shared_conv_layers
        
        # Shared feature extractor (task-interactive features)
        self.shared_convs = nn.ModuleList()
        for i in range(shared_conv_layers):
            self.shared_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Task-specific feature extractors
        # Classification branch
        self.cls_convs = nn.ModuleList()
        for i in range(2):  # 2 layers for classification
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Regression branch (bbox)
        self.reg_convs = nn.ModuleList()
        for i in range(2):  # 2 layers for regression
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Landmark branch
        self.ldm_convs = nn.ModuleList()
        for i in range(2):  # 2 layers for landmarks
            self.ldm_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Task-aligned predictors
        self.cls_predictor = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_predictor = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.ldm_predictor = nn.Conv2d(in_channels, num_anchors * num_landmarks, 3, padding=1)
        
        # Task alignment modules (key TOOD innovation)
        self.task_alignment_cls = TaskAlignmentModule(in_channels)
        self.task_alignment_reg = TaskAlignmentModule(in_channels)
        self.task_alignment_ldm = TaskAlignmentModule(in_channels)
        
        # Task interaction attention (for feature sharing between tasks)
        self.task_interaction = TaskInteractionModule(in_channels, num_tasks=3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for optimal convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for classification predictor (focal loss compatibility)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_predictor.bias, bias_value)
    
    def forward(self, features):
        """
        Forward pass of Task-Aligned Head
        
        Args:
            features: List of feature maps from different FPN levels
                     Each feature map: [B, C, H, W]
        
        Returns:
            tuple: (classifications, bbox_regressions, landmark_regressions)
                   Each as list of predictions for different FPN levels
        """
        classifications = []
        bbox_regressions = []
        landmark_regressions = []
        
        for feature in features:
            # 1. Shared feature extraction (task-interactive features)
            shared_feat = feature
            for shared_conv in self.shared_convs:
                shared_feat = shared_conv(shared_feat)
            
            # 2. Task interaction (TOOD innovation: cross-task feature sharing)
            task_features = self.task_interaction(shared_feat)
            cls_feat, reg_feat, ldm_feat = task_features
            
            # 3. Task-specific feature extraction
            # Classification branch
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            
            # Regression branch
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            
            # Landmark branch
            for ldm_conv in self.ldm_convs:
                ldm_feat = ldm_conv(ldm_feat)
            
            # 4. Task alignment (TOOD core innovation)
            cls_aligned = self.task_alignment_cls(cls_feat, reg_feat, ldm_feat)
            reg_aligned = self.task_alignment_reg(reg_feat, cls_feat, ldm_feat)
            ldm_aligned = self.task_alignment_ldm(ldm_feat, cls_feat, reg_feat)
            
            # 5. Task predictions
            cls_pred = self.cls_predictor(cls_aligned)
            reg_pred = self.reg_predictor(reg_aligned)
            ldm_pred = self.ldm_predictor(ldm_aligned)
            
            # 6. Reshape predictions for loss computation
            batch_size, _, height, width = cls_pred.shape
            
            # Classification: [B, num_anchors*num_classes, H, W] -> [B, H*W*num_anchors, num_classes]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(batch_size, -1, self.num_classes)
            
            # Regression: [B, num_anchors*4, H, W] -> [B, H*W*num_anchors, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
            reg_pred = reg_pred.view(batch_size, -1, 4)
            
            # Landmarks: [B, num_anchors*num_landmarks, H, W] -> [B, H*W*num_anchors, num_landmarks]
            ldm_pred = ldm_pred.permute(0, 2, 3, 1).contiguous()
            ldm_pred = ldm_pred.view(batch_size, -1, self.num_landmarks)
            
            classifications.append(cls_pred)
            bbox_regressions.append(reg_pred)
            landmark_regressions.append(ldm_pred)
        
        return classifications, bbox_regressions, landmark_regressions


class TaskAlignmentModule(nn.Module):
    """
    Task Alignment Module - Core TOOD Innovation
    
    Aligns features for a specific task by considering features from other tasks.
    This enables better task-specific feature learning while maintaining task interaction.
    
    Args:
        in_channels (int): Number of input channels
    """
    
    def __init__(self, in_channels):
        super(TaskAlignmentModule, self).__init__()
        
        self.in_channels = in_channels
        
        # Task alignment attention
        self.alignment_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)
        self.alignment_norm = nn.BatchNorm2d(in_channels)
        self.alignment_act = nn.ReLU(inplace=True)
        
        # Channel attention for task-specific feature enhancement
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for spatial feature enhancement
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, target_feat, aux_feat1, aux_feat2):
        """
        Forward pass of task alignment
        
        Args:
            target_feat: Features for the target task
            aux_feat1: Features from auxiliary task 1
            aux_feat2: Features from auxiliary task 2
        
        Returns:
            torch.Tensor: Task-aligned features
        """
        # 1. Concatenate all task features for alignment learning
        combined_feat = torch.cat([target_feat, aux_feat1, aux_feat2], dim=1)
        
        # 2. Learn task alignment weights
        aligned_feat = self.alignment_conv(combined_feat)
        aligned_feat = self.alignment_norm(aligned_feat)
        aligned_feat = self.alignment_act(aligned_feat)
        
        # 3. Apply channel attention
        channel_att = self.channel_attention(aligned_feat)
        aligned_feat = aligned_feat * channel_att
        
        # 4. Apply spatial attention
        # Generate spatial attention from mean and max pooling
        avg_out = torch.mean(aligned_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(aligned_feat, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        aligned_feat = aligned_feat * spatial_att
        
        # 5. Residual connection with target features
        output = aligned_feat + target_feat
        
        return output


class TaskInteractionModule(nn.Module):
    """
    Task Interaction Module for Cross-Task Feature Sharing (Simplified for Efficiency)
    
    Enables information flow between different tasks to improve overall performance.
    Each task benefits from features learned by other tasks.
    
    Args:
        in_channels (int): Number of input channels
        num_tasks (int): Number of tasks (3 for face detection)
    """
    
    def __init__(self, in_channels, num_tasks=3):
        super(TaskInteractionModule, self).__init__()
        
        self.in_channels = in_channels
        self.num_tasks = num_tasks
        
        # Task-specific projection layers
        self.task_projections = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            for _ in range(num_tasks)
        ])
        
        # Simplified cross-task interaction via channel attention
        self.task_interaction_conv = nn.Conv2d(in_channels * num_tasks, in_channels, 1, bias=False)
        self.task_interaction_norm = nn.BatchNorm2d(in_channels)
        self.task_interaction_act = nn.ReLU(inplace=True)
        
        # Task attention weights
        self.task_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, num_tasks, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, shared_features):
        """
        Forward pass of task interaction
        
        Args:
            shared_features: Shared features from backbone [B, C, H, W]
        
        Returns:
            list: Task-specific features [cls_feat, reg_feat, ldm_feat]
        """
        # 1. Generate initial task-specific features
        task_features = []
        for i, proj in enumerate(self.task_projections):
            task_feat = proj(shared_features)
            task_features.append(task_feat)
        
        # 2. Concatenate all task features for interaction learning
        all_task_features = torch.cat(task_features, dim=1)  # [B, 3*C, H, W]
        
        # 3. Learn shared task representation
        shared_task_feat = self.task_interaction_conv(all_task_features)
        shared_task_feat = self.task_interaction_norm(shared_task_feat)
        shared_task_feat = self.task_interaction_act(shared_task_feat)
        
        # 4. Generate task attention weights
        task_weights = self.task_attention(shared_task_feat)  # [B, num_tasks, 1, 1]
        
        # 5. Apply task-specific attention to enhance each task
        enhanced_features = []
        for i, task_feat in enumerate(task_features):
            # Apply task-specific attention weight
            weight = task_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            enhanced_feat = task_feat * weight + shared_task_feat
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


def create_tood_head(in_channels=256, num_classes=2, num_anchors=2):
    """
    Factory function to create TOOD Task-Aligned Head for face detection
    
    Args:
        in_channels (int): Number of input feature channels
        num_classes (int): Number of classes (2 for face detection)
        num_anchors (int): Number of anchors per location
    
    Returns:
        TaskAlignedHead: TOOD head for face detection
    """
    return TaskAlignedHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        num_landmarks=10,  # 5 facial landmarks = 10 coordinates
        shared_conv_layers=4
    )


def test_tood_head():
    """Test TOOD head implementation"""
    print("🧪 Testing TOOD Task-Aligned Head")
    print("=" * 50)
    
    # Create TOOD head
    tood_head = create_tood_head(in_channels=256, num_classes=2, num_anchors=2)
    
    # Test with multi-scale features (typical FPN outputs)
    feature_sizes = [(80, 80), (40, 40), (20, 20)]  # P3, P4, P5
    features = []
    
    for h, w in feature_sizes:
        feat = torch.randn(2, 256, h, w)  # Batch=2, Channels=256
        features.append(feat)
    
    # Forward pass
    with torch.no_grad():
        classifications, bbox_regressions, landmark_regressions = tood_head(features)
    
    # Validate outputs
    print("Output validation:")
    for i, (cls, reg, ldm) in enumerate(zip(classifications, bbox_regressions, landmark_regressions)):
        h, w = feature_sizes[i]
        expected_anchors = h * w * 2  # 2 anchors per location
        
        print(f"  Level {i+1} ({h}x{w}):")
        print(f"    Classification: {cls.shape} (expected: [2, {expected_anchors}, 2])")
        print(f"    BBox regression: {reg.shape} (expected: [2, {expected_anchors}, 4])")
        print(f"    Landmarks: {ldm.shape} (expected: [2, {expected_anchors}, 10])")
        
        assert cls.shape == (2, expected_anchors, 2), f"Classification shape mismatch at level {i}"
        assert reg.shape == (2, expected_anchors, 4), f"Regression shape mismatch at level {i}"
        assert ldm.shape == (2, expected_anchors, 10), f"Landmark shape mismatch at level {i}"
    
    # Parameter count
    total_params = sum(p.numel() for p in tood_head.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ All tests passed! TOOD head ready for FeatherFace V4")


if __name__ == "__main__":
    test_tood_head()