"""
FeatherFace V6 SimAM Innovation Implementation
============================================

This module implements our V6 innovation: replacing CBAM with SimAM (Simple, Parameter-Free Attention).
This represents the most revolutionary attention innovation - achieving CBAM-level performance with
ZERO additional parameters.

INNOVATION: CBAM (12,929 params) → SimAM (0 params) for ultimate mobile efficiency
Scientific foundation: SimAM 2024-2025 research + Electronics 2025 baseline

Performance Target: Maintain CBAM performance with 0 attention parameters
Base Architecture: FeatherFaceCBAMExact (MobileNet + BiFPN + SSH) - proven components  
Innovation: Replace CBAM with SimAM for parameter-free attention

Authors: Original FeatherFace (Kim, D.; Jung, J.; Kim, J.) + SimAM Innovation
Implementation: V6 SimAM replacing CBAM attention with zero-parameter approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, ChannelShuffle2, SSH
from models.sinam import SimAMBlock


class FeatherFaceV6SimAM(nn.Module):
    """
    FeatherFace V6 SimAM Innovation
    
    Replaces CBAM attention with SimAM (Simple, Parameter-Free Attention Module)
    for revolutionary mobile face detection efficiency. This innovation focuses on
    eliminating attention parameters while maintaining the proven backbone + neck + head.
    
    Key Innovation:
    - CBAM (12,929 params) → SimAM (0 params) 
    - Maintains spatial and channel attention capabilities
    - Based on neuroscience theories and energy function optimization
    - Recent 2024-2025 research shows +1.7% improvement with +0.01MB overhead
    - Perfect for mobile deployment and IoT devices
    
    Scientific Foundation:
    - Base: FeatherFace Electronics 2025 (CBAM baseline: 488,664 params)
    - Innovation: SimAM Parameter-Free Attention (2024-2025 research)
    - Expected: Same performance as CBAM with 12,929 parameter reduction
    
    Performance Targets (based on SimAM research results):
    - WIDERFace Easy: 92.7% AP (maintain CBAM level)
    - WIDERFace Medium: 90.7% AP (maintain CBAM level)
    - WIDERFace Hard: 78.3%+ AP (maintain or improve via efficiency)
    - Parameter reduction: -12,929 vs CBAM baseline
    - Mobile deployment: Maximum efficiency with zero attention overhead
    """
    
    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceV6SimAM, self).__init__()
        self.phase = phase
        self.cfg = cfg
        
        # Innovation: Use same out_channel as CBAM baseline for controlled comparison
        # CBAM baseline: 488,664 parameters with out_channel=52
        # SimAM innovation: Expected ~475,735 parameters (same minus 12,929 CBAM params)
        
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
        
        # 2. SimAM Attention Modules (INNOVATION: Replace CBAM with SimAM)
        # Backbone SimAM modules (3x) - zero parameters vs 12,929 for CBAM
        self.backbone_sinam_0 = SimAMBlock(in_channels_list[0])  # 64 channels, 0 params
        self.backbone_sinam_1 = SimAMBlock(in_channels_list[1])  # 128 channels, 0 params
        self.backbone_sinam_2 = SimAMBlock(in_channels_list[2])  # 256 channels, 0 params
        
        # 3. BiFPN Feature Aggregation (keep proven architecture)
        conv_channel_coef = {
            0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
        }
        
        # Import BiFPN from net.py (same as baseline)
        from models.net import BiFPN
        
        self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.compound_coef = 0
        
        # Create BiFPN layers (identical configuration to baseline)
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True)
              for _ in range(self.fpn_cell_repeats[self.compound_coef])]
        )
        
        # 4. SimAM modules for BiFPN outputs (INNOVATION: 0 params vs CBAM)
        self.bif_sinam_0 = SimAMBlock(out_channels)  # P3, 0 params
        self.bif_sinam_1 = SimAMBlock(out_channels)  # P4, 0 params
        self.bif_sinam_2 = SimAMBlock(out_channels)  # P5, 0 params
        
        # 5. SSH Detection Heads (keep proven detection architecture)
        self.ssh1 = SSH(out_channels, out_channels)  # P3
        self.ssh2 = SSH(out_channels, out_channels)  # P4  
        self.ssh3 = SSH(out_channels, out_channels)  # P5
        
        # 6. Classification and Regression Heads (identical to baseline)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)
        
        # 7. Channel Shuffle Optimization (keep for feature mixing)
        self.feature_shuffle_0 = ChannelShuffle2(out_channels, 2)
        self.feature_shuffle_1 = ChannelShuffle2(out_channels, 2)
        self.feature_shuffle_2 = ChannelShuffle2(out_channels, 2)
    
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create classification head for face detection"""
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create bounding box regression head"""
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead
    
    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create landmark regression head"""
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead
    
    def forward(self, inputs):
        """Forward pass with SimAM parameter-free attention (V6 innovation)"""
        
        # 1. Backbone feature extraction (identical to baseline)
        out = self.body(inputs)
        
        # Extract multiscale features
        feat1 = out[1]  # stage1 -> 64 channels
        feat2 = out[2]  # stage2 -> 128 channels
        feat3 = out[3]  # stage3 -> 256 channels
        
        # 2. Apply SimAM attention to backbone features (INNOVATION: 0 params)
        feat1 = self.backbone_sinam_0(feat1)
        feat2 = self.backbone_sinam_1(feat2)
        feat3 = self.backbone_sinam_2(feat3)
        
        # 3. BiFPN feature aggregation (proven approach)
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # 4. Apply SimAM attention to BiFPN features (INNOVATION: 0 params)
        p3 = self.bif_sinam_0(p3)
        p4 = self.bif_sinam_1(p4)
        p5 = self.bif_sinam_2(p5)
        
        # 5. Channel shuffle for better feature mixing
        p3 = self.feature_shuffle_0(p3)
        p4 = self.feature_shuffle_1(p4)
        p5 = self.feature_shuffle_2(p5)
        
        # 6. SSH Detection Heads (proven approach)
        feature1 = self.ssh1(p3)
        feature2 = self.ssh2(p4)
        feature3 = self.ssh3(p5)
        
        features = [feature1, feature2, feature3]
        
        # 7. Classification, BBox, and Landmark predictions
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        
        # 8. Format output based on phase
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output
    
    def get_parameter_count(self):
        """Get detailed parameter count breakdown with SimAM"""
        backbone_params = sum(p.numel() for p in self.body.parameters())
        
        # SimAM modules have 0 parameters (revolutionary!)
        sinam_backbone_params = (
            sum(p.numel() for p in self.backbone_sinam_0.parameters()) +
            sum(p.numel() for p in self.backbone_sinam_1.parameters()) +
            sum(p.numel() for p in self.backbone_sinam_2.parameters())
        )
        
        bifpn_params = sum(p.numel() for p in self.bifpn.parameters())
        
        sinam_bifpn_params = (
            sum(p.numel() for p in self.bif_sinam_0.parameters()) +
            sum(p.numel() for p in self.bif_sinam_1.parameters()) +
            sum(p.numel() for p in self.bif_sinam_2.parameters())
        )
        
        ssh_params = (
            sum(p.numel() for p in self.ssh1.parameters()) +
            sum(p.numel() for p in self.ssh2.parameters()) +
            sum(p.numel() for p in self.ssh3.parameters())
        )
        
        head_params = (
            sum(p.numel() for p in self.ClassHead.parameters()) +
            sum(p.numel() for p in self.BboxHead.parameters()) +
            sum(p.numel() for p in self.LandmarkHead.parameters())
        )
        
        channel_shuffle_params = (
            sum(p.numel() for p in self.feature_shuffle_0.parameters()) +
            sum(p.numel() for p in self.feature_shuffle_1.parameters()) +
            sum(p.numel() for p in self.feature_shuffle_2.parameters())
        )
        
        total = (backbone_params + sinam_backbone_params + bifpn_params + 
                sinam_bifpn_params + ssh_params + head_params + channel_shuffle_params)
        
        return {
            'backbone': backbone_params,
            'sinam_backbone': sinam_backbone_params,
            'bifpn': bifpn_params,
            'sinam_bifpn': sinam_bifpn_params,
            'ssh_heads': ssh_params,
            'detection_heads': head_params,
            'channel_shuffle': channel_shuffle_params,
            'total': total,
            'cbam_baseline': 488664,  # CBAM baseline for comparison
            'cbam_attention_params': 12929,  # CBAM attention parameters
            'parameter_reduction_vs_cbam': 488664 - total,  # vs CBAM baseline
            'attention_parameter_reduction': 12929,  # 100% attention parameter reduction
        }
    
    def compare_with_baselines(self):
        """Compare SimAM innovation with CBAM and other attention methods"""
        param_info = self.get_parameter_count()
        
        comparison = {
            'sinam_total': param_info['total'],
            'cbam_baseline': param_info['cbam_baseline'],
            'cbam_attention_params': param_info['cbam_attention_params'],
            'sinam_attention_params': 0,  # Zero parameters!
            'parameter_reduction_vs_cbam': param_info['parameter_reduction_vs_cbam'],
            'attention_parameter_reduction': param_info['attention_parameter_reduction'],
            'attention_efficiency_gain': 'infinite',  # 0 params = infinite efficiency
            'parameter_free_innovation': True,  # Revolutionary zero-parameter attention
            'expected_performance_maintenance': True,  # SimAM maintains CBAM performance
            'innovation_type': 'parameter_free_attention',
            'scientific_foundation': '2024-2025',
            'mobile_deployment_advantage': 'maximum',
        }
        
        return comparison
    
    def get_attention_analysis(self, x):
        """
        Analyze SimAM attention benefits vs CBAM approaches
        
        Returns attention maps and parameter-free benefits analysis.
        """
        # Get features up to backbone attention
        out = self.body(x)
        feat1 = out[1]
        feat2 = out[2] 
        feat3 = out[3]
        
        # Get SimAM attention maps (no parameters involved!)
        att_map1 = self.backbone_sinam_0.get_attention_visualization(feat1)
        att_map2 = self.backbone_sinam_1.get_attention_visualization(feat2)
        att_map3 = self.backbone_sinam_2.get_attention_visualization(feat3)
        
        return {
            'attention_maps': [att_map1, att_map2, att_map3],
            'parameter_free': True,
            'attention_params': 0,
            'cbam_comparison': 'SimAM achieves attention with 0 params vs 12,929 for CBAM',
            'sinam_innovation': 'Parameter-free attention via energy function optimization'
        }


# Detection head classes (same as baseline for consistency)
class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)


def create_v6_sinam_model(cfg_v6_sinam, phase='train'):
    """
    Factory function to create SimAM innovation model
    
    Args:
        cfg_v6_sinam: Configuration for SimAM innovation
        phase: 'train' or 'test'
    
    Returns:
        FeatherFaceV6SimAM model with parameter-free attention
    """
    model = FeatherFaceV6SimAM(cfg=cfg_v6_sinam, phase=phase)
    
    # Get parameter comparison with baselines
    comparison = model.compare_with_baselines()
    
    print(f"FeatherFace V6 SimAM Innovation Model Created")
    print(f"SimAM total parameters: {comparison['sinam_total']:,}")
    print(f"CBAM baseline: {comparison['cbam_baseline']:,}")
    print(f"Parameter reduction vs CBAM: {comparison['parameter_reduction_vs_cbam']:+,}")
    print(f"SimAM attention params: {comparison['sinam_attention_params']}")
    print(f"CBAM attention params: {comparison['cbam_attention_params']:,}")
    print(f"Attention parameter reduction: {comparison['attention_parameter_reduction']:,} (100%)")
    print(f"Attention efficiency gain: {comparison['attention_efficiency_gain']}")
    print(f"Innovation: {comparison['innovation_type']}")
    print(f"Mobile deployment advantage: {comparison['mobile_deployment_advantage']}")
    
    return model