"""
FeatherFace V2 ECA-Net Innovation Implementation
===============================================

This module implements our V2 innovation: replacing CBAM attention with ECA-Net attention.
This creates a controlled scientific comparison with the exact CBAM baseline.

INNOVATION: CBAM → ECA-Net replacement for mobile-optimized face detection
Scientific foundation: Wang et al. CVPR 2020 + Electronics 2025 baseline

Base Architecture: FeatherFaceCBAMExact (488,664 parameters)
Innovation: Replace CBAM with ECA-Net (expect ~475K parameters, +efficiency)

Authors: Original FeatherFace (Kim, D.; Jung, J.; Kim, J.) + ECA-Net Innovation
Implementation: V2 ECA-Net attention mechanism replacing CBAM baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, SSH, BiFPN, ChannelShuffle2
from models.eca_net import EfficientChannelAttention


class ClassHead(nn.Module):
    """Classification head for face detection"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    """Bounding box regression head"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    """Facial landmark detection head"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)


class FeatherFaceV2ECAInnovation(nn.Module):
    """
    FeatherFace V2 ECA-Net Innovation
    
    Replaces CBAM attention with ECA-Net attention for mobile-optimized face detection.
    This creates a controlled scientific comparison with the CBAM baseline.
    
    Key Innovation:
    - CBAM → ECA-Net attention replacement
    - Identical architecture otherwise for controlled comparison
    - Expected benefits: Reduced parameters, faster inference, maintained accuracy
    
    Scientific Foundation:
    - Base: FeatherFace Electronics 2025 (CBAM baseline: 488,664 params)
    - Innovation: ECA-Net attention (Wang et al. CVPR 2020)
    - Expected: ~475K parameters (-13K vs CBAM, +efficiency)
    
    Performance Targets:
    - Maintain: WIDERFace Easy/Medium performance
    - Improve: Mobile inference speed (2x faster attention)
    - Reduce: Parameter count and memory usage
    """
    
    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceV2ECAInnovation, self).__init__()
        self.phase = phase
        self.cfg = cfg
        
        # Innovation: Use same out_channel as CBAM baseline for controlled comparison
        # CBAM baseline: 488,664 parameters with out_channel=52
        # ECA innovation: Expected ~475K parameters with same out_channel=52
        
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
        
        # 2. ECA-Net Attention Modules (INNOVATION: Replace CBAM)
        # Backbone ECA modules (3x) - ultra-efficient attention
        self.backbone_eca_0 = EfficientChannelAttention(in_channels_list[0])  # 64 channels - ~3 params
        self.backbone_eca_1 = EfficientChannelAttention(in_channels_list[1])  # 128 channels - ~5 params  
        self.backbone_eca_2 = EfficientChannelAttention(in_channels_list[2])  # 256 channels - ~5 params
        
        # 3. BiFPN Feature Aggregation (identical to CBAM baseline)
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
        
        # BiFPN ECA modules (3x) - INNOVATION: Replace CBAM with ECA
        self.bif_eca_0 = EfficientChannelAttention(out_channels)  # P3 - ~3 params
        self.bif_eca_1 = EfficientChannelAttention(out_channels)  # P4 - ~3 params
        self.bif_eca_2 = EfficientChannelAttention(out_channels)  # P5 - ~3 params
        
        # 4. SSH Detection Heads with DCN (identical to baseline)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        
        # 5. Channel Shuffle Optimization (identical to baseline)
        self.ssh1_cs = ChannelShuffle2(out_channels, 2)
        self.ssh2_cs = ChannelShuffle2(out_channels, 2)
        self.ssh3_cs = ChannelShuffle2(out_channels, 2)
        
        # 6. Detection Heads (identical to baseline)
        self.ClassHead = nn.ModuleList([
            ClassHead(out_channels, 2) for _ in range(3)
        ])
        self.BboxHead = nn.ModuleList([
            BboxHead(out_channels, 2) for _ in range(3)
        ])
        self.LandmarkHead = nn.ModuleList([
            LandmarkHead(out_channels, 2) for _ in range(3)
        ])
    
    def forward(self, inputs):
        """Forward pass with ECA-Net attention (V2 innovation)"""
        
        # 1. Backbone feature extraction (identical to baseline)
        out = self.body(inputs)
        
        # Extract multiscale features
        feat1 = out[1]  # stage1 -> 64 channels
        feat2 = out[2]  # stage2 -> 128 channels  
        feat3 = out[3]  # stage3 -> 256 channels
        
        # 2. Apply ECA-Net attention to backbone features (INNOVATION)
        feat1 = self.backbone_eca_0(feat1)
        feat2 = self.backbone_eca_1(feat2)
        feat3 = self.backbone_eca_2(feat3)
        
        # 3. BiFPN feature aggregation (identical to baseline)
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # 4. Apply ECA-Net attention to BiFPN features (INNOVATION)
        p3 = self.bif_eca_0(p3)
        p4 = self.bif_eca_1(p4)
        p5 = self.bif_eca_2(p5)
        
        # 5. SSH detection with DCN (identical to baseline)
        f1 = self.ssh1(p3)
        f2 = self.ssh2(p4) 
        f3 = self.ssh3(p5)
        
        # 6. Channel shuffle optimization (identical to baseline)
        f1 = self.ssh1_cs(f1)
        f2 = self.ssh2_cs(f2)
        f3 = self.ssh3_cs(f3)
        
        # 7. Detection heads (identical to baseline)
        features = [f1, f2, f3]
        
        bbox_regressions = []
        classifications = []
        ldm_regressions = []
        
        for i, feature in enumerate(features):
            bbox_regressions.append(self.BboxHead[i](feature))
            classifications.append(self.ClassHead[i](feature))
            ldm_regressions.append(self.LandmarkHead[i](feature))
        
        if self.phase == 'train':
            output = (
                torch.cat(bbox_regressions, dim=1),
                torch.cat(classifications, dim=1),
                torch.cat(ldm_regressions, dim=1)
            )
        else:
            output = (
                torch.cat(bbox_regressions, dim=1),
                F.softmax(torch.cat(classifications, dim=1), dim=-1),
                torch.cat(ldm_regressions, dim=1)
            )
        
        return output
    
    def get_parameter_count(self):
        """Get detailed parameter count breakdown with ECA-Net"""
        backbone_params = sum(p.numel() for p in self.body.parameters())
        
        eca_backbone_params = (
            sum(p.numel() for p in self.backbone_eca_0.parameters()) +
            sum(p.numel() for p in self.backbone_eca_1.parameters()) +
            sum(p.numel() for p in self.backbone_eca_2.parameters())
        )
        
        bifpn_params = sum(p.numel() for p in self.bifpn.parameters())
        
        eca_bifpn_params = (
            sum(p.numel() for p in self.bif_eca_0.parameters()) +
            sum(p.numel() for p in self.bif_eca_1.parameters()) +
            sum(p.numel() for p in self.bif_eca_2.parameters())
        )
        
        ssh_params = (
            sum(p.numel() for p in self.ssh1.parameters()) +
            sum(p.numel() for p in self.ssh2.parameters()) +
            sum(p.numel() for p in self.ssh3.parameters())
        )
        
        cs_params = (
            sum(p.numel() for p in self.ssh1_cs.parameters()) +
            sum(p.numel() for p in self.ssh2_cs.parameters()) +
            sum(p.numel() for p in self.ssh3_cs.parameters())
        )
        
        head_params = (
            sum(p.numel() for p in self.ClassHead.parameters()) +
            sum(p.numel() for p in self.BboxHead.parameters()) +
            sum(p.numel() for p in self.LandmarkHead.parameters())
        )
        
        total = backbone_params + eca_backbone_params + bifpn_params + eca_bifpn_params + ssh_params + cs_params + head_params
        
        return {
            'backbone': backbone_params,
            'eca_backbone': eca_backbone_params,
            'bifpn': bifpn_params,
            'eca_bifpn': eca_bifpn_params,
            'ssh': ssh_params,
            'channel_shuffle': cs_params,
            'detection_heads': head_params,
            'total': total,
            'cbam_baseline': 488664,  # CBAM baseline for comparison
            'parameter_reduction': 488664 - total,  # Innovation benefit
            'efficiency_gain': (488664 - total) / 488664 * 100  # Percentage reduction
        }
    
    def compare_with_cbam_baseline(self):
        """Compare ECA-Net innovation with CBAM baseline"""
        param_info = self.get_parameter_count()
        
        comparison = {
            'eca_total': param_info['total'],
            'cbam_baseline': param_info['cbam_baseline'],
            'parameter_reduction': param_info['parameter_reduction'],
            'efficiency_gain_percent': param_info['efficiency_gain'],
            'attention_params_eca': param_info['eca_backbone'] + param_info['eca_bifpn'],
            'attention_params_cbam_estimated': 12929,  # From CBAM baseline
            'attention_efficiency': 12929 - (param_info['eca_backbone'] + param_info['eca_bifpn']),
            'innovation_validated': param_info['parameter_reduction'] > 10000,  # Significant reduction
        }
        
        return comparison


def create_v2_eca_innovation_model(cfg_v2_eca, phase='train'):
    """
    Factory function to create ECA-Net innovation model
    
    Args:
        cfg_v2_eca: Configuration for ECA-Net innovation
        phase: 'train' or 'test'
    
    Returns:
        FeatherFaceV2ECAInnovation model with ECA-Net attention
    """
    model = FeatherFaceV2ECAInnovation(cfg=cfg_v2_eca, phase=phase)
    
    # Get parameter comparison with CBAM baseline
    comparison = model.compare_with_cbam_baseline()
    
    print(f"FeatherFace V2 ECA-Net Innovation Model Created")
    print(f"ECA-Net parameters: {comparison['eca_total']:,}")
    print(f"CBAM baseline: {comparison['cbam_baseline']:,}")
    print(f"Parameter reduction: {comparison['parameter_reduction']:+,}")
    print(f"Efficiency gain: {comparison['efficiency_gain_percent']:.1f}%")
    print(f"Innovation validated: {comparison['innovation_validated']}")
    
    return model