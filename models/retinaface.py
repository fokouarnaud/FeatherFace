import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import CBAM as CBAM
from models.net import ChannelShuffle2 as ChannelShuffle




class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()

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
                # load params
                backbone.load_state_dict(new_state_dict)
       

        if cfg['name'] == 'mobilenet0.25' or cfg['name'] == 'Resnet50':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
            out_channels = cfg['out_channel']
            
            # V1 BASELINE: Simple architecture without backbone CBAM
            # Standard baseline implementation for comparison
            
            conv_channel_coef = {
                # the channels of P3/P4/P5.
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
            self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]  # out_channels for 489K target
            self.fpn_cell_repeats = [2, 4, 5, 6, 7, 7, 8, 8, 8]  # Increased to 2 repeats for exactly 488.7K target
            self.compound_coef=0
            self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True if self.compound_coef < 6 else False
                    )
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])
            
            # V1 BASELINE: Simple architecture without additional CBAM
            # Standard baseline implementation for comparison

            # PAPER COMPLIANT: Context Enhancement using SSH for multiscale contextual information
            # SSH modules capture adaptive context through parallel 3x3, 5x5, 7x7 deformable convolutions
            self.ssh1 = SSH(out_channels, out_channels)  # P3: SSH with 3 branches for small face context
            self.ssh2 = SSH(out_channels, out_channels)  # P4: SSH with 3 branches for medium face context  
            self.ssh3 = SSH(out_channels, out_channels)  # P5: SSH with 3 branches for large face context
            
            # PAPER COMPLIANT: Channel Shuffling for effective inter-channel information exchange
            # Facilitates information mixing to further enrich feature representation  
            self.cs1 = ChannelShuffle(out_channels, groups=2)  # P3: enhance small-face feature mixing
            self.cs2 = ChannelShuffle(out_channels, groups=2)  # P4: enhance medium-face feature mixing
            self.cs3 = ChannelShuffle(out_channels, groups=2)  # P5: enhance large-face feature mixing

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        
        if self.cfg['name'] == 'mobilenet0.25':
            # PAPER COMPLIANT ARCHITECTURE: MobileNet-0.25 backbone + attention + multiscale aggregation + detection heads
            # The integration of these modules jointly enhances feature representation for accurate face detection
            
            # 1. MobileNet-0.25 Backbone: Multi-scale feature extraction
            out = self.body(inputs)
            out = list(out.values())  # [P3:64ch, P4:128ch, P5:256ch] - hierarchical features
            
            # 2. V1 BASELINE: Direct feature processing without CBAM
            # Simple baseline for comparison with enhanced models
            backbone_attention_features = out
            
            # 3. Multiscale Feature Aggregation (BiFPN): Strategic fusion for multi-scale face detection
            # High-resolution features (P3) help detect small faces, semantically rich features (P4,P5) enhance large faces
            bifpn_features = self.bifpn(backbone_attention_features)
            
            # 4. V1 BASELINE: Direct feature processing without additional CBAM
            # Simple baseline for comparison with enhanced models
            bif_features = bifpn_features

            # 5. Detection Heads: Context Enhancement using SSH for multiscale contextual information  
            # SSH modules use parallel deformable conv branches (3x3, 5x5, 7x7) for adaptive context
            ssh_feature1 = self.ssh1(bif_features[0])  # P3: multi-branch context for small faces
            ssh_feature2 = self.ssh2(bif_features[1])  # P4: multi-branch context for medium faces
            ssh_feature3 = self.ssh3(bif_features[2])  # P5: multi-branch context for large faces
            
            # 6. Channel Shuffling: Facilitate effective inter-channel information exchange
            # Further enriches feature representation through organized channel mixing
            feat1 = self.cs1(ssh_feature1)  # P3: enhance inter-channel exchange
            feat2 = self.cs2(ssh_feature2)  # P4: optimize feature mixing  
            feat3 = self.cs3(ssh_feature3)  # P5: enrich representation
           
            features = [feat1, feat2, feat3]
        

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
    
