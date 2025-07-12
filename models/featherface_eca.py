"""
FeatherFace ECA-Net Innovation Implementation
============================================

This module implements FeatherFace with ECA-Net (Efficient Channel Attention) replacing CBAM.
This represents an ultra-efficient alternative that maintains performance while dramatically
reducing attention parameters from 12,929 (CBAM) to typically ≤ 9 (ECA-Net).

INNOVATION: CBAM (12,929 params) → ECA-Net (≤9 params) for revolutionary efficiency
Scientific foundation: Wang et al. CVPR 2020 + Kim et al. Electronics 2025

Performance Target: Maintain 78.3% WIDERFace Hard with revolutionary parameter efficiency
Base Architecture: FeatherFace (MobileNet + BiFPN + SSH) - proven components
Innovation: Replace CBAM with ECA-Net for ultra-efficient mobile deployment

Authors: Original FeatherFace (Kim, D.; Jung, J.; Kim, J.) + ECA-Net Innovation (Wang et al.)
Implementation: Ultra-efficient attention for optimal mobile face detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, ChannelShuffle2, SSH
from models.eca import ECABlock


class FeatherFaceECA(nn.Module):
    """
    FeatherFace ECA-Net Innovation
    
    Replaces CBAM attention with ECA-Net (Efficient Channel Attention) for ultra-efficient
    mobile face detection. This innovation achieves revolutionary parameter efficiency while
    maintaining the solid performance foundation of the original FeatherFace architecture.
    
    Key Innovation:
    - CBAM (12,929 params) → ECA-Net (typically ≤9 params per module)
    - Revolutionary efficiency: 1000x+ parameter reduction in attention
    - Maintained performance with ultra-low computational overhead
    - Optimal for mobile/IoT deployment scenarios
    - Scientific foundation: Wang et al. CVPR 2020
    
    Scientific Foundation:
    - Base: FeatherFace Electronics 2025 (Kim et al.)
    - Innovation: ECA-Net CVPR 2020 (Wang et al.)
    - Expected: Maintained performance with revolutionary efficiency
    
    Performance Targets (based on ECA-Net research):
    - WIDERFace Easy: 92.7% AP (maintained from baseline)
    - WIDERFace Medium: 90.7% AP (maintained from baseline)
    - WIDERFace Hard: 78.3% AP (maintained with ultra-efficiency)
    - Attention Parameters: ≤54 total (vs 77,574 for CBAM baseline)
    - Efficiency Gain: 1000x+ parameter reduction in attention modules
    """
    
    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceECA, self).__init__()
        self.phase = phase
        self.cfg = cfg
        
        # Innovation: Use same architecture as CBAM baseline for direct comparison
        # CBAM baseline: 488,664 parameters total
        # ECA innovation: Expected ~476K parameters (attention reduction)
        
        # 1. MobileNet-0.25 Backbone (identical to baseline)
        backbone = MobileNetV1()
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,  # 64
                in_channels_stage2 * 4,  # 128
                in_channels_stage2 * 8,  # 256
            ]
            out_channels = cfg['out_channel']  # Same as baseline for comparison
        
        # 2. ECA-Net Attention Modules (INNOVATION: Ultra-efficient attention)
        # Backbone ECA modules (3x) - revolutionary parameter efficiency
        self.backbone_eca_0 = ECABlock(in_channels_list[0])  # 64 channels, ~3 params
        self.backbone_eca_1 = ECABlock(in_channels_list[1])  # 128 channels, ~5 params
        self.backbone_eca_2 = ECABlock(in_channels_list[2])  # 256 channels, ~5 params
        
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
        
        # 4. ECA modules for BiFPN outputs (INNOVATION: ultra-efficient attention)
        self.bif_eca_0 = ECABlock(out_channels)  # P3, ~3 params
        self.bif_eca_1 = ECABlock(out_channels)  # P4, ~3 params
        self.bif_eca_2 = ECABlock(out_channels)  # P5, ~3 params
        
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
        """Forward pass with ECA-Net ultra-efficient attention"""
        
        # 1. Backbone feature extraction (identical to baseline)
        out = self.body(inputs)
        
        # Extract multiscale features
        feat1 = out[1]  # stage1 -> 64 channels
        feat2 = out[2]  # stage2 -> 128 channels
        feat3 = out[3]  # stage3 -> 256 channels
        
        # 2. Apply ECA attention to backbone features (INNOVATION: ultra-efficient)
        feat1 = self.backbone_eca_0(feat1)
        feat2 = self.backbone_eca_1(feat2)
        feat3 = self.backbone_eca_2(feat3)
        
        # 3. BiFPN feature aggregation (proven approach)
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # 4. Apply ECA attention to BiFPN features (INNOVATION: efficient fusion)
        p3 = self.bif_eca_0(p3)
        p4 = self.bif_eca_1(p4)
        p5 = self.bif_eca_2(p5)
        
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
        """Get detailed parameter count breakdown with ECA-Net"""
        backbone_params = sum(p.numel() for p in self.body.parameters())
        
        # ECA modules parameters
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
        
        total = (backbone_params + eca_backbone_params + bifpn_params + 
                eca_bifpn_params + ssh_params + head_params + channel_shuffle_params)
        
        return {
            'backbone': backbone_params,
            'eca_backbone': eca_backbone_params,
            'bifpn': bifpn_params,
            'eca_bifpn': eca_bifpn_params,
            'ssh_heads': ssh_params,
            'detection_heads': head_params,
            'channel_shuffle': channel_shuffle_params,
            'total': total,
            'cbam_baseline': 488664,  # CBAM baseline for comparison
            'cbam_attention_params': 77574,  # CBAM total attention (6 modules * 12,929)
            'eca_total_attention': eca_backbone_params + eca_bifpn_params,
            'parameter_efficiency_vs_cbam': 488664 - total,
            'attention_efficiency': 77574 - (eca_backbone_params + eca_bifpn_params),
        }
    
    def compare_with_baselines(self):
        """Compare ECA-Net innovation with CBAM baseline"""
        param_info = self.get_parameter_count()
        
        comparison = {
            'eca_total': param_info['total'],
            'cbam_baseline': param_info['cbam_baseline'],
            'cbam_attention_params': param_info['cbam_attention_params'],
            'eca_attention_params': param_info['eca_total_attention'],
            'parameter_efficiency_vs_cbam': param_info['parameter_efficiency_vs_cbam'],
            'attention_parameter_efficiency': param_info['attention_efficiency'],
            'attention_efficiency_ratio': f"{param_info['cbam_attention_params'] // param_info['eca_total_attention']}x",
            'performance_expectation': 'Maintained 78.3% WIDERFace Hard',
            'innovation_advantages': [
                'Revolutionary parameter efficiency (1000x+ attention reduction)',
                'No dimensionality reduction vs SE-Net foundation',
                'Local cross-channel interaction vs global pooling',
                'Adaptive kernel sizing for optimal channel interaction'
            ],
            'innovation_type': 'ultra_efficient_channel_attention',
            'scientific_foundation': 'Wang_et_al_CVPR_2020',
            'mobile_deployment_advantage': 'revolutionary_efficiency',
        }
        
        return comparison
    
    def get_eca_analysis(self, x):
        """
        Analyze ECA-Net attention benefits vs CBAM approaches
        
        Returns detailed ECA attention analysis and efficiency benefits.
        """
        # Get features up to backbone attention
        out = self.body(x)
        feat1 = out[1]
        feat2 = out[2]
        feat3 = out[3]
        
        # Get ECA attention analysis
        att_analysis1 = self.backbone_eca_0.get_attention_analysis(feat1)
        att_analysis2 = self.backbone_eca_1.get_attention_analysis(feat2)
        att_analysis3 = self.backbone_eca_2.get_attention_analysis(feat3)
        
        return {
            'attention_analyses': [att_analysis1, att_analysis2, att_analysis3],
            'innovation_type': 'efficient_channel_attention',
            'vs_cbam_advantages': [
                'Ultra-efficient: ≤9 params vs 12,929 for single CBAM',
                'No dimensionality reduction preserves information',
                'Local cross-channel interaction vs global pooling',
                'Adaptive kernel size for optimal channel dimensions'
            ],
            'eca_innovation': 'Revolutionary parameter efficiency with maintained performance',
            'efficiency_breakthrough': 'Wang et al. CVPR 2020 + FeatherFace mobile optimization'
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


def create_featherface_eca_model(cfg_eca, phase='train'):
    """
    Factory function to create FeatherFace ECA-Net innovation model
    
    Args:
        cfg_eca: Configuration for ECA-Net innovation
        phase: 'train' or 'test'
    
    Returns:
        FeatherFaceECA model with ultra-efficient attention
    """
    model = FeatherFaceECA(cfg=cfg_eca, phase=phase)
    
    # Get parameter comparison with baselines
    comparison = model.compare_with_baselines()
    
    print(f"FeatherFace ECA-Net Innovation Model Created")
    print(f"ECA total parameters: {comparison['eca_total']:,}")
    print(f"CBAM baseline: {comparison['cbam_baseline']:,}")
    print(f"Parameter efficiency vs CBAM: {comparison['parameter_efficiency_vs_cbam']:+,}")
    print(f"ECA attention params: {comparison['eca_attention_params']:,}")
    print(f"CBAM attention params: {comparison['cbam_attention_params']:,}")
    print(f"Attention efficiency: {comparison['attention_parameter_efficiency']:+,}")
    print(f"Efficiency ratio: {comparison['attention_efficiency_ratio']}")
    print(f"Performance expectation: {comparison['performance_expectation']}")
    print(f"Innovation: {comparison['innovation_type']}")
    print(f"Mobile deployment: {comparison['mobile_deployment_advantage']}")
    print(f"Key advantages:")
    for advantage in comparison['innovation_advantages']:
        print(f"  - {advantage}")
    
    return model