"""
FeatherFace V3 ELA-S Innovation Implementation
=============================================

This module implements our V3 innovation: replacing CBAM baseline with ELA-S attention.
This creates the most advanced spatial attention mechanism for mobile face detection.

INNOVATION: CBAM → ELA-S replacement for superior spatial awareness + efficiency
Scientific foundation: Xuwei et al. 2024 (arXiv:2403.01123) + Electronics 2025 baseline

Performance Target: +0.97% mAP vs ECA-Net, +0.56% vs CBAM (based on YOLOX-Nano results)
Base Architecture: FeatherFaceCBAMExact (488,664 parameters)
Innovation: Replace CBAM with ELA-S (expect significant spatial awareness improvement)

Authors: Original FeatherFace (Kim, D.; Jung, J.; Kim, J.) + ELA-S Innovation
Implementation: V3 ELA-S spatial attention mechanism replacing CBAM baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, SSH, BiFPN, ChannelShuffle2
from models.ela_s import EfficientLocalAttentionSpatial


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


class FeatherFaceV3ELAInnovation(nn.Module):
    """
    FeatherFace V3 ELA-S Innovation
    
    Replaces CBAM attention with ELA-S spatial attention for superior face detection.
    This represents the most advanced spatial attention mechanism for mobile deployment.
    
    Key Innovation:
    - CBAM → ELA-S spatial attention replacement
    - Superior spatial awareness via strip pooling + 1D convolutions
    - Expected benefits: +0.97% mAP vs ECA-Net, enhanced spatial feature capture
    
    Scientific Foundation:
    - Base: FeatherFace Electronics 2025 (CBAM baseline: 488,664 params)
    - Innovation: ELA-S spatial attention (Xuwei et al. 2024)
    - Expected: Enhanced spatial performance, maintained efficiency
    
    Performance Targets (based on YOLOX-Nano results):
    - WIDERFace Easy: 92.7%+ AP (maintain or improve)
    - WIDERFace Medium: 90.7%+ AP (maintain or improve)
    - WIDERFace Hard: 78.3%+ AP (target improvement)
    - Overall mAP: +0.97% vs ECA-Net, +0.56% vs CBAM
    """
    
    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceV3ELAInnovation, self).__init__()
        self.phase = phase
        self.cfg = cfg
        
        # Innovation: Use same out_channel as CBAM baseline for controlled comparison
        # CBAM baseline: 488,664 parameters with out_channel=52
        # ELA-S innovation: Expected similar parameters with superior spatial attention
        
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
        
        # 2. ELA-S Attention Modules (INNOVATION: Replace CBAM with spatial attention)
        # Backbone ELA-S modules (3x) - superior spatial attention
        self.backbone_ela_0 = EfficientLocalAttentionSpatial(in_channels_list[0], reduction_ratio=8)  # 64 channels
        self.backbone_ela_1 = EfficientLocalAttentionSpatial(in_channels_list[1], reduction_ratio=8)  # 128 channels  
        self.backbone_ela_2 = EfficientLocalAttentionSpatial(in_channels_list[2], reduction_ratio=8)  # 256 channels
        
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
        
        # BiFPN ELA-S modules (3x) - INNOVATION: Replace CBAM with ELA-S
        self.bif_ela_0 = EfficientLocalAttentionSpatial(out_channels, reduction_ratio=8)  # P3
        self.bif_ela_1 = EfficientLocalAttentionSpatial(out_channels, reduction_ratio=8)  # P4
        self.bif_ela_2 = EfficientLocalAttentionSpatial(out_channels, reduction_ratio=8)  # P5
        
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
        """Forward pass with ELA-S spatial attention (V3 innovation)"""
        
        # 1. Backbone feature extraction (identical to baseline)
        out = self.body(inputs)
        
        # Extract multiscale features
        feat1 = out[1]  # stage1 -> 64 channels
        feat2 = out[2]  # stage2 -> 128 channels  
        feat3 = out[3]  # stage3 -> 256 channels
        
        # 2. Apply ELA-S spatial attention to backbone features (INNOVATION)
        feat1 = self.backbone_ela_0(feat1)
        feat2 = self.backbone_ela_1(feat2)
        feat3 = self.backbone_ela_2(feat3)
        
        # 3. BiFPN feature aggregation (identical to baseline)
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # 4. Apply ELA-S spatial attention to BiFPN features (INNOVATION)
        p3 = self.bif_ela_0(p3)
        p4 = self.bif_ela_1(p4)
        p5 = self.bif_ela_2(p5)
        
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
        """Get detailed parameter count breakdown with ELA-S"""
        backbone_params = sum(p.numel() for p in self.body.parameters())
        
        ela_backbone_params = (
            sum(p.numel() for p in self.backbone_ela_0.parameters()) +
            sum(p.numel() for p in self.backbone_ela_1.parameters()) +
            sum(p.numel() for p in self.backbone_ela_2.parameters())
        )
        
        bifpn_params = sum(p.numel() for p in self.bifpn.parameters())
        
        ela_bifpn_params = (
            sum(p.numel() for p in self.bif_ela_0.parameters()) +
            sum(p.numel() for p in self.bif_ela_1.parameters()) +
            sum(p.numel() for p in self.bif_ela_2.parameters())
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
        
        total = backbone_params + ela_backbone_params + bifpn_params + ela_bifpn_params + ssh_params + cs_params + head_params
        
        return {
            'backbone': backbone_params,
            'ela_backbone': ela_backbone_params,
            'bifpn': bifpn_params,
            'ela_bifpn': ela_bifpn_params,
            'ssh': ssh_params,
            'channel_shuffle': cs_params,
            'detection_heads': head_params,
            'total': total,
            'cbam_baseline': 488664,  # CBAM baseline for comparison
            'eca_innovation': 475757,  # ECA-Net innovation for comparison
            'parameter_difference_cbam': 488664 - total,  # vs CBAM
            'parameter_difference_eca': total - 475757,   # vs ECA-Net
        }
    
    def compare_with_baselines(self):
        """Compare ELA-S innovation with CBAM baseline and ECA-Net innovation"""
        param_info = self.get_parameter_count()
        
        comparison = {
            'ela_s_total': param_info['total'],
            'cbam_baseline': param_info['cbam_baseline'],
            'eca_innovation': param_info['eca_innovation'],
            'vs_cbam_difference': param_info['parameter_difference_cbam'],
            'vs_eca_difference': param_info['parameter_difference_eca'],
            'attention_params_ela_s': param_info['ela_backbone'] + param_info['ela_bifpn'],
            'attention_params_cbam_estimated': 12929,  # From CBAM baseline
            'attention_params_eca_estimated': 22,      # From ECA innovation
            'spatial_attention_advantage': True,       # ELA-S provides spatial awareness
            'expected_map_improvement': 0.97,          # +0.97% vs ECA-Net (YOLOX-Nano)
            'innovation_type': 'spatial_attention',
        }
        
        return comparison
    
    def get_spatial_attention_maps(self, x):
        """
        Extract spatial attention maps for visualization and analysis
        
        Returns attention maps from all ELA-S modules for understanding
        spatial attention patterns in face detection.
        """
        # Get backbone features
        out = self.body(x)
        feat1 = out[1]  # 64 channels
        feat2 = out[2]  # 128 channels
        feat3 = out[3]  # 256 channels
        
        # Get backbone spatial attention maps
        backbone_attention_maps = {
            'stage1_64ch': self.backbone_ela_0.get_attention_map(feat1),
            'stage2_128ch': self.backbone_ela_1.get_attention_map(feat2),
            'stage3_256ch': self.backbone_ela_2.get_attention_map(feat3),
        }
        
        # Process through BiFPN
        features = [self.backbone_ela_0(feat1), self.backbone_ela_1(feat2), self.backbone_ela_2(feat3)]
        features = self.bifpn(features)
        p3, p4, p5 = features
        
        # Get BiFPN spatial attention maps
        bifpn_attention_maps = {
            'p3_52ch': self.bif_ela_0.get_attention_map(p3),
            'p4_52ch': self.bif_ela_1.get_attention_map(p4),
            'p5_52ch': self.bif_ela_2.get_attention_map(p5),
        }
        
        return {
            'backbone_attention': backbone_attention_maps,
            'bifpn_attention': bifpn_attention_maps
        }


def create_v3_ela_innovation_model(cfg_v3_ela, phase='train'):
    """
    Factory function to create ELA-S innovation model
    
    Args:
        cfg_v3_ela: Configuration for ELA-S innovation
        phase: 'train' or 'test'
    
    Returns:
        FeatherFaceV3ELAInnovation model with ELA-S spatial attention
    """
    model = FeatherFaceV3ELAInnovation(cfg=cfg_v3_ela, phase=phase)
    
    # Get parameter comparison with baselines
    comparison = model.compare_with_baselines()
    
    print(f"FeatherFace V3 ELA-S Innovation Model Created")
    print(f"ELA-S parameters: {comparison['ela_s_total']:,}")
    print(f"CBAM baseline: {comparison['cbam_baseline']:,}")
    print(f"ECA innovation: {comparison['eca_innovation']:,}")
    print(f"vs CBAM: {comparison['vs_cbam_difference']:+,}")
    print(f"vs ECA: {comparison['vs_eca_difference']:+,}")
    print(f"Expected mAP improvement: +{comparison['expected_map_improvement']:.2f}% vs ECA-Net")
    print(f"Innovation type: {comparison['innovation_type']}")
    
    return model