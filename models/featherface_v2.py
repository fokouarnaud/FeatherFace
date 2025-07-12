#!/usr/bin/env python3
"""
FeatherFace V2 - Innovation avec Coordinate Attention

Architecture V1 with Coordinate Attention replacing CBAM modules.
Maintains V1 stability with mobile-optimized spatial attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import ChannelShuffle2 as ChannelShuffle
from models.attention_v2 import CoordinateAttention


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


class FeatherFaceV2(nn.Module):
    """
    FeatherFace V2 - Innovation avec Coordinate Attention
    
    Cette version améliore FeatherFace V1 avec l'innovation Coordinate Attention:
    1. Base RetinaFace identique (comme V1)
    2. Remplace CBAM par Coordinate Attention (innovation mobile-optimisée)
    3. Training standard avec MultiBoxLoss
    4. Architecture cohérente avec V1 baseline
    
    Performance attendue:
    - WIDERFace Hard: +8-10% vs V1 
    - Paramètres: ~493K (+4K vs V1)
    - Inference: 2x plus rapide que CBAM
    - Stabilité: Équivalente à V1
    """
    
    def __init__(self, cfg=None, phase='train'):
        """
        Initialize FeatherFace V2
        
        Args:
            cfg: Configuration dict (cfg_v2 recommandé)
            phase: 'train' or 'test'
        """
        super(FeatherFaceV2, self).__init__()
        
        self.cfg = cfg
        self.phase = phase
        backbone = None
        
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
            out_channels = cfg['out_channel']
            
            
            # INNOVATION V2: Coordinate Attention remplace CBAM
            # Placement: Post-backbone pour optimiser les features extraites
            self.backbone_ca_0 = CoordinateAttention(in_channels_list[0], reduction_ratio=32)
            self.backbone_ca_1 = CoordinateAttention(in_channels_list[1], reduction_ratio=32)
            self.backbone_ca_2 = CoordinateAttention(in_channels_list[2], reduction_ratio=32)
            
            # ReLU activation après attention (comme V1 avec CBAM)
            self.relu_0 = nn.ReLU()
            self.relu_1 = nn.ReLU()
            self.relu_2 = nn.ReLU()
            
            # BiFPN configuration (identique V1)
            conv_channel_coef = {
                0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
                1: [40, 112, 320],
                2: [48, 120, 352],
                3: [48, 136, 384],
                4: [56, 160, 448],
                5: [64, 176, 512],
                6: [72, 200, 576],
                7: [72, 200, 576],
                8: [80, 224, 640],
            }
            self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]
            self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]  # V1 config
            self.compound_coef = 0
            
            # BiFPN (identique V1)
            self.bifpn = nn.Sequential(
                *[BiFPN(self.fpn_num_filters[self.compound_coef],
                        conv_channel_coef[self.compound_coef],
                        True if _ == 0 else False,
                        attention=True if self.compound_coef < 6 else False
                        )
                  for _ in range(self.fpn_cell_repeats[self.compound_coef])])
            
            # Coordinate Attention post-BiFPN (innovation vs CBAM)
            self.bif_ca_0 = CoordinateAttention(out_channels, reduction_ratio=32)
            self.bif_ca_1 = CoordinateAttention(out_channels, reduction_ratio=32)
            self.bif_ca_2 = CoordinateAttention(out_channels, reduction_ratio=32)
            
            self.bif_relu_0 = nn.ReLU()
            self.bif_relu_1 = nn.ReLU()
            self.bif_relu_2 = nn.ReLU()
            
            # SSH context modules (identique V1)
            self.ssh1 = SSH(out_channels, out_channels)
            self.ssh2 = SSH(out_channels, out_channels)
            self.ssh3 = SSH(out_channels, out_channels)
            
            # Channel Shuffle (identique V1)
            self.ssh1_cs = nn.Sequential(
                ChannelShuffle(channels=out_channels, groups=2),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            
            self.ssh2_cs = nn.Sequential(
                ChannelShuffle(channels=out_channels, groups=2),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            self.ssh3_cs = nn.Sequential(
                ChannelShuffle(channels=out_channels, groups=2),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        # Detection heads (identique V1)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        
    
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        """
        Forward pass FeatherFace V2
        
        Architecture comme V1 avec innovation Coordinate Attention:
        1. MobileNet backbone extraction
        2. Coordinate Attention sur backbone features (vs CBAM)
        3. BiFPN feature pyramid
        4. Coordinate Attention sur BiFPN features (vs CBAM)
        5. SSH context enhancement
        6. Channel Shuffle
        7. Detection heads
        
        Args:
            inputs: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple: (bbox_regressions, classifications, ldm_regressions)
        """
        if self.cfg['name'] == 'mobilenet0.25':
            # 1. MobileNet backbone extraction (identique V1)
            out = self.body(inputs)
            out = list(out.values())  # [P3, P4, P5]
            
            # 2. INNOVATION: Coordinate Attention sur backbone (vs CBAM)
            ca_backbone_0 = self.backbone_ca_0(out[0])
            ca_backbone_1 = self.backbone_ca_1(out[1])
            ca_backbone_2 = self.backbone_ca_2(out[2])
            
            # Residual connection + activation (comme V1)
            ca_backbone_0 = ca_backbone_0 + out[0]
            ca_backbone_1 = ca_backbone_1 + out[1]
            ca_backbone_2 = ca_backbone_2 + out[2]
            
            ca_backbone_0 = self.relu_0(ca_backbone_0)
            ca_backbone_1 = self.relu_1(ca_backbone_1)
            ca_backbone_2 = self.relu_2(ca_backbone_2)
            
            b_ca = [ca_backbone_0, ca_backbone_1, ca_backbone_2]
            
            # 3. BiFPN feature pyramid (identique V1)
            bifpn = self.bifpn(b_ca)
            
            # 4. INNOVATION: Coordinate Attention sur BiFPN (vs CBAM)
            bif_ca_0 = self.bif_ca_0(bifpn[0])
            bif_ca_1 = self.bif_ca_1(bifpn[1])
            bif_ca_2 = self.bif_ca_2(bifpn[2])
            
            # Residual connection + activation (comme V1)
            bif_ca_0 = bif_ca_0 + bifpn[0]
            bif_ca_1 = bif_ca_1 + bifpn[1]
            bif_ca_2 = bif_ca_2 + bifpn[2]

            bif_c_0 = self.bif_relu_0(bif_ca_0)
            bif_c_1 = self.bif_relu_1(bif_ca_1)
            bif_c_2 = self.bif_relu_2(bif_ca_2)

            bif_ca = [bif_c_0, bif_c_1, bif_c_2]
            
            # 5. SSH context enhancement (identique V1)
            feature1 = self.ssh1(bif_ca[0])
            feature2 = self.ssh2(bif_ca[1])
            feature3 = self.ssh3(bif_ca[2])
            
            # 6. Channel Shuffle (identique V1)
            feat1 = self.ssh1_cs(feature1)
            feat2 = self.ssh2_cs(feature2)
            feat3 = self.ssh3_cs(feature3)
           
            features = [feat1, feat2, feat3]
        
        # 7. Detection heads (identique V1)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output