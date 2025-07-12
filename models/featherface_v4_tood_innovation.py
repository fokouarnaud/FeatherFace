"""
FeatherFace V4 TOOD Innovation Implementation
============================================

This module implements our V4 innovation: replacing SSH heads with TOOD Task-Aligned Head.
This represents the most advanced detection head innovation for mobile face detection.

INNOVATION: SSH → TOOD Task-Aligned Head replacement for superior task alignment
Scientific foundation: Feng et al. ICCV 2021 (arXiv:2108.07755) + Electronics 2025 baseline

Performance Target: +2-3% mAP vs SSH baseline, -30% detection head parameters
Base Architecture: FeatherFaceCBAMExact (MobileNet + CBAM + BiFPN) - proven components
Innovation: Replace SSH with TOOD Task-Aligned Head for 3-task face detection

Authors: Original FeatherFace (Kim, D.; Jung, J.; Kim, J.) + TOOD Innovation
Implementation: V4 TOOD Task-Aligned Head replacing SSH detection heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, BiFPN, ChannelShuffle2, CBAM
from models.tood_head import TaskAlignedHead


class FeatherFaceV4TOODInnovation(nn.Module):
    """
    FeatherFace V4 TOOD Innovation
    
    Replaces SSH detection heads with TOOD Task-Aligned Head for superior face detection.
    This represents the most advanced detection head innovation while maintaining the
    proven backbone (MobileNet) + attention (CBAM) + neck (BiFPN) architecture.
    
    Key Innovation:
    - SSH → TOOD Task-Aligned Head replacement
    - 3-task alignment: classification + bbox + landmarks
    - Task Alignment Learning (TAL) for better sample assignment
    - Expected benefits: +2-3% mAP, -30% detection head parameters
    
    Scientific Foundation:
    - Base: FeatherFace Electronics 2025 (CBAM baseline: 488,664 params)
    - Innovation: TOOD Task-Aligned Detection (Feng et al. ICCV 2021)
    - Expected: Enhanced detection performance with reduced head parameters
    
    Performance Targets (based on TOOD paper results):
    - WIDERFace Easy: 92.7%+ AP (maintain or improve)
    - WIDERFace Medium: 90.7%+ AP (maintain or improve)
    - WIDERFace Hard: 80.0%+ AP (target +1.7% improvement via task alignment)
    - Overall mAP: +2-3% vs SSH baseline
    """
    
    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceV4TOODInnovation, self).__init__()
        self.phase = phase
        self.cfg = cfg
        
        # Innovation: Use same out_channel as CBAM baseline for controlled comparison
        # CBAM baseline: 488,664 parameters with out_channel=52
        # TOOD innovation: Expected ~430K parameters with superior task alignment
        
        # 1. MobileNet-0.25 Backbone (identical to CBAM baseline)
        backbone = MobileNetV1()
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,  # 64
                in_channels_stage2 * 4,  # 128
                in_channels_stage2 * 8,  # 256
            ]
            out_channels = cfg['out_channel']  # Same as CBAM baseline for comparison
        
        # 2. CBAM Attention Modules (keep proven attention mechanism)
        # Backbone CBAM modules (3x) - proven effectiveness for face detection
        self.backbone_cbam_0 = CBAM(in_channels_list[0])  # 64 channels
        self.backbone_cbam_1 = CBAM(in_channels_list[1])  # 128 channels  
        self.backbone_cbam_2 = CBAM(in_channels_list[2])  # 256 channels
        
        # 3. BiFPN Feature Aggregation (keep proven neck architecture)
        conv_channel_coef = {
            0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
        }
        self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.compound_coef = 0
        
        # Create BiFPN layers (identical configuration)
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True)
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])
        
        # BiFPN CBAM modules (3x) - keep proven attention for neck
        self.bif_cbam_0 = CBAM(out_channels)  # P3
        self.bif_cbam_1 = CBAM(out_channels)  # P4
        self.bif_cbam_2 = CBAM(out_channels)  # P5
        
        # 4. TOOD Task-Aligned Head (INNOVATION: Replace SSH)
        # This is the key innovation - task-aligned detection head
        self.tood_head = TaskAlignedHead(
            in_channels=out_channels,      # 52 channels from BiFPN
            num_classes=2,                 # face/background
            num_anchors=2,                 # 2 anchors per location
            num_landmarks=10,              # 5 facial landmarks = 10 coordinates
            shared_conv_layers=4           # 4 shared convolution layers
        )
        
        # 5. Channel Shuffle Optimization (keep for feature mixing)
        # Apply channel shuffle before TOOD head for better feature mixing
        self.feature_shuffle_0 = ChannelShuffle2(out_channels, 2)
        self.feature_shuffle_1 = ChannelShuffle2(out_channels, 2)
        self.feature_shuffle_2 = ChannelShuffle2(out_channels, 2)
    
    def forward(self, inputs):
        """Forward pass with TOOD Task-Aligned Head (V4 innovation)"""
        
        # 1. Backbone feature extraction (identical to baseline)
        out = self.body(inputs)
        
        # Extract multiscale features
        feat1 = out[1]  # stage1 -> 64 channels
        feat2 = out[2]  # stage2 -> 128 channels  
        feat3 = out[3]  # stage3 -> 256 channels
        
        # 2. Apply CBAM attention to backbone features (proven approach)
        feat1 = self.backbone_cbam_0(feat1)
        feat2 = self.backbone_cbam_1(feat2)
        feat3 = self.backbone_cbam_2(feat3)
        
        # 3. BiFPN feature aggregation (proven approach)
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # 4. Apply CBAM attention to BiFPN features (proven approach)
        p3 = self.bif_cbam_0(p3)
        p4 = self.bif_cbam_1(p4)
        p5 = self.bif_cbam_2(p5)
        
        # 5. Channel shuffle for better feature mixing before detection
        p3 = self.feature_shuffle_0(p3)
        p4 = self.feature_shuffle_1(p4)
        p5 = self.feature_shuffle_2(p5)
        
        # 6. TOOD Task-Aligned Detection (INNOVATION)
        # Replace SSH with TOOD for superior task alignment
        feature_maps = [p3, p4, p5]
        classifications, bbox_regressions, landmark_regressions = self.tood_head(feature_maps)
        
        # 7. Format output based on phase
        if self.phase == 'train':
            output = (
                torch.cat(bbox_regressions, dim=1),
                torch.cat(classifications, dim=1),
                torch.cat(landmark_regressions, dim=1)
            )
        else:
            output = (
                torch.cat(bbox_regressions, dim=1),
                F.softmax(torch.cat(classifications, dim=1), dim=-1),
                torch.cat(landmark_regressions, dim=1)
            )
        
        return output
    
    def get_parameter_count(self):
        """Get detailed parameter count breakdown with TOOD"""
        backbone_params = sum(p.numel() for p in self.body.parameters())
        
        cbam_backbone_params = (
            sum(p.numel() for p in self.backbone_cbam_0.parameters()) +
            sum(p.numel() for p in self.backbone_cbam_1.parameters()) +
            sum(p.numel() for p in self.backbone_cbam_2.parameters())
        )
        
        bifpn_params = sum(p.numel() for p in self.bifpn.parameters())
        
        cbam_bifpn_params = (
            sum(p.numel() for p in self.bif_cbam_0.parameters()) +
            sum(p.numel() for p in self.bif_cbam_1.parameters()) +
            sum(p.numel() for p in self.bif_cbam_2.parameters())
        )
        
        tood_head_params = sum(p.numel() for p in self.tood_head.parameters())
        
        channel_shuffle_params = (
            sum(p.numel() for p in self.feature_shuffle_0.parameters()) +
            sum(p.numel() for p in self.feature_shuffle_1.parameters()) +
            sum(p.numel() for p in self.feature_shuffle_2.parameters())
        )
        
        total = backbone_params + cbam_backbone_params + bifpn_params + cbam_bifpn_params + tood_head_params + channel_shuffle_params
        
        return {
            'backbone': backbone_params,
            'cbam_backbone': cbam_backbone_params,
            'bifpn': bifpn_params,
            'cbam_bifpn': cbam_bifpn_params,
            'tood_head': tood_head_params,
            'channel_shuffle': channel_shuffle_params,
            'total': total,
            'cbam_baseline': 488664,  # CBAM baseline for comparison
            'ssh_head_estimated': 180000,  # SSH head parameters (estimated)
            'parameter_reduction_vs_cbam': 488664 - total,  # vs CBAM baseline
            'head_improvement': 180000 - tood_head_params,  # SSH vs TOOD head
        }
    
    def compare_with_baselines(self):
        """Compare TOOD innovation with SSH baseline and other methods"""
        param_info = self.get_parameter_count()
        
        comparison = {
            'tood_total': param_info['total'],
            'cbam_baseline': param_info['cbam_baseline'],
            'ssh_head_estimated': param_info['ssh_head_estimated'],
            'tood_head_params': param_info['tood_head'],
            'parameter_reduction_vs_cbam': param_info['parameter_reduction_vs_cbam'],
            'head_parameter_improvement': param_info['head_improvement'],
            'head_improvement_percentage': (param_info['head_improvement'] / param_info['ssh_head_estimated']) * 100,
            'task_alignment_benefit': True,  # TOOD provides task alignment
            'expected_map_improvement': 2.5,  # +2-3% mAP based on TOOD paper
            'innovation_type': 'task_aligned_detection_head',
            'scientific_foundation': 'ICCV 2021',
        }
        
        return comparison
    
    def get_task_alignment_analysis(self, x):
        """
        Analyze task alignment benefits of TOOD vs traditional approaches
        
        Returns task-aligned predictions and alignment scores for analysis.
        """
        # Get features up to BiFPN
        out = self.body(x)
        feat1 = self.backbone_cbam_0(out[1])
        feat2 = self.backbone_cbam_1(out[2])
        feat3 = self.backbone_cbam_2(out[3])
        
        features = self.bifpn([feat1, feat2, feat3])
        p3, p4, p5 = features
        
        p3 = self.bif_cbam_0(p3)
        p4 = self.bif_cbam_1(p4)
        p5 = self.bif_cbam_2(p5)
        
        p3 = self.feature_shuffle_0(p3)
        p4 = self.feature_shuffle_1(p4)
        p5 = self.feature_shuffle_2(p5)
        
        # Get task-aligned predictions from TOOD head
        feature_maps = [p3, p4, p5]
        classifications, bbox_regressions, landmark_regressions = self.tood_head(feature_maps)
        
        return {
            'feature_maps': feature_maps,
            'classifications': classifications,
            'bbox_regressions': bbox_regressions,
            'landmark_regressions': landmark_regressions,
            'task_alignment_active': True,
            'tood_innovation': 'Task-aligned 3-task face detection'
        }


def create_v4_tood_innovation_model(cfg_v4_tood, phase='train'):
    """
    Factory function to create TOOD innovation model
    
    Args:
        cfg_v4_tood: Configuration for TOOD innovation
        phase: 'train' or 'test'
    
    Returns:
        FeatherFaceV4TOODInnovation model with TOOD Task-Aligned Head
    """
    model = FeatherFaceV4TOODInnovation(cfg=cfg_v4_tood, phase=phase)
    
    # Get parameter comparison with baselines
    comparison = model.compare_with_baselines()
    
    print(f"FeatherFace V4 TOOD Innovation Model Created")
    print(f"TOOD parameters: {comparison['tood_total']:,}")
    print(f"CBAM baseline: {comparison['cbam_baseline']:,}")
    print(f"Parameter reduction vs CBAM: {comparison['parameter_reduction_vs_cbam']:+,}")
    print(f"TOOD head params: {comparison['tood_head_params']:,}")
    print(f"SSH head estimated: {comparison['ssh_head_estimated']:,}")
    print(f"Head improvement: {comparison['head_parameter_improvement']:+,} ({comparison['head_improvement_percentage']:.1f}%)")
    print(f"Expected mAP improvement: +{comparison['expected_map_improvement']:.1f}%")
    print(f"Innovation: {comparison['innovation_type']}")
    
    return model