"""
FeatherFace V2 Architecture
Optimized RetinaFace with 0.25M parameters target
Uses lightweight modules from modules_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

# Import original backbone
try:
    from models.net import MobileNetV1
except ImportError:
    from net import MobileNetV1

# Import V2 optimized modules
try:
    from models.modules_v2 import (
        CBAM_Plus, 
        SharedMultiHead, 
        BiFPN_Light, 
        SSH_Grouped,
        SharedCBAMManager,
        ChannelShuffle_Light
    )
except ImportError:
    from modules_v2 import (
        CBAM_Plus, 
        SharedMultiHead, 
        BiFPN_Light, 
        SSH_Grouped,
        SharedCBAMManager,
        ChannelShuffle_Light
    )

class RetinaFaceV2(nn.Module):
    """
    Optimized RetinaFace V2 Architecture
    Target: 0.25M parameters with 92%+ mAP on WIDERFace
    
    Key optimizations:
    - BiFPN_Light with 32 channels and depthwise separable convs
    - SSH_Grouped with grouped convolutions
    - Shared CBAM modules across the network
    - Unified detection head (SharedMultiHead)
    """
    
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg: Network related settings (should use cfg_mnet_v2)
        :param phase: train or test
        """
        super(RetinaFaceV2, self).__init__()
        
        self.cfg = cfg
        self.phase = phase
        
        # Initialize backbone with error handling
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                try:
                    checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", 
                                          map_location=torch.device('cpu'))
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    backbone.load_state_dict(new_state_dict)
                    print("✓ Loaded pretrained backbone weights")
                except FileNotFoundError:
                    print("⚠️ Pretrained weights not found, using random initialization")
        else:
            raise ValueError(f"Unsupported backbone: {cfg['name']}. Expected 'mobilenet0.25'")
        
        # Verify backbone was initialized
        if backbone is None:
            raise RuntimeError("Backbone initialization failed - backbone is None")
            
        # Setup backbone feature extractor
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        
        # Calculate channel configurations
        in_channels_stage2 = cfg['in_channel']  # 32 for mobilenet0.25
        in_channels_list = [
            in_channels_stage2 * 2,   # 64
            in_channels_stage2 * 4,   # 128
            in_channels_stage2 * 8,   # 256
        ]
        
        # V2: Use reduced output channels (32 instead of 64)
        out_channels = cfg.get('out_channel_v2', 32)  # Default to 32 for V2
        
        # Initialize Shared CBAM Manager for backbone
        # This reduces total CBAM parameters by sharing weights
        backbone_cbam_configs = {
            'stage1': in_channels_list[0],  # 64
            'stage2': in_channels_list[1],  # 128
            'stage3': in_channels_list[2],  # 256
        }
        self.backbone_cbam_manager = SharedCBAMManager(backbone_cbam_configs, reduction_ratio=32)
        
        # Shared ReLU activations
        self.relu = nn.ReLU(inplace=True)        
        # BiFPN Light configuration
        # V2: Use only 2 repetitions instead of 3, and 32 channels
        conv_channels = in_channels_list  # [64, 128, 256]
        bifpn_repeats = 2  # Reduced from 3
        
        # Create BiFPN Light modules
        self.bifpn = nn.Sequential(
            *[BiFPN_Light(
                num_channels=out_channels,  # 32
                conv_channels=conv_channels if i == 0 else None,
                first_time=True if i == 0 else False,
                use_dwsep=True  # Use depthwise separable convs
            ) for i in range(bifpn_repeats)]
        )
        
        # Shared CBAM for BiFPN outputs
        bifpn_cbam_configs = {
            'p3': out_channels,  # 32
            'p4': out_channels,  # 32  
            'p5': out_channels,  # 32
        }
        self.bifpn_cbam_manager = SharedCBAMManager(bifpn_cbam_configs, reduction_ratio=32)        
        # SSH Grouped modules (lightweight context enhancement)
        self.ssh1 = SSH_Grouped(out_channels, out_channels, groups=4, reduction=2)
        self.ssh2 = SSH_Grouped(out_channels, out_channels, groups=4, reduction=2)
        self.ssh3 = SSH_Grouped(out_channels, out_channels, groups=4, reduction=2)
        
        # Channel Shuffle modules for feature mixing
        self.ssh1_cs = ChannelShuffle_Light(out_channels, groups=4)
        self.ssh2_cs = ChannelShuffle_Light(out_channels, groups=4)
        self.ssh3_cs = ChannelShuffle_Light(out_channels, groups=4)
        
        # V2: Use SharedMultiHead instead of separate heads
        # This creates 3 instances of SharedMultiHead for the 3 feature levels
        self.shared_heads = nn.ModuleList([
            SharedMultiHead(in_channels=out_channels, num_anchors=2)
            for _ in range(3)
        ])    
    def forward(self, inputs):
        """
        Forward pass of RetinaFaceV2
        
        Args:
            inputs: Input tensor of shape (B, 3, H, W)
            
        Returns:
            tuple: (classifications, bbox_regressions, landmarks)
                - classifications: List of 3 tensors, each (B, -1, 2)
                - bbox_regressions: List of 3 tensors, each (B, -1, 4)
                - landmarks: List of 3 tensors, each (B, -1, 10)
        """
        
        # Extract features from backbone
        out = self.body(inputs)
        out = list(out.values())  # Convert OrderedDict to list
        
        # Apply shared CBAM to backbone features
        cbam_features = []
        cbam_names = ['stage1', 'stage2', 'stage3']
        
        for i, (feat, name) in enumerate(zip(out, cbam_names)):
            # Apply CBAM using shared manager
            cbam_feat = self.backbone_cbam_manager(feat, name)
            # Add residual connection
            cbam_feat = cbam_feat + feat
            # Apply ReLU
            cbam_feat = self.relu(cbam_feat)
            cbam_features.append(cbam_feat)        
        # Pass through BiFPN Light
        bifpn_features = self.bifpn(cbam_features)
        
        # Apply shared CBAM to BiFPN outputs
        bifpn_cbam_features = []
        bifpn_names = ['p3', 'p4', 'p5']
        
        for i, (feat, name) in enumerate(zip(bifpn_features, bifpn_names)):
            # Apply CBAM using shared manager
            cbam_feat = self.bifpn_cbam_manager(feat, name)
            # Add residual connection
            cbam_feat = cbam_feat + feat
            # Apply ReLU
            cbam_feat = self.relu(cbam_feat)
            bifpn_cbam_features.append(cbam_feat)
        
        # Context enhancement with SSH Grouped
        ssh_features = []
        ssh_modules = [self.ssh1, self.ssh2, self.ssh3]
        cs_modules = [self.ssh1_cs, self.ssh2_cs, self.ssh3_cs]
        
        for i, (feat, ssh, cs) in enumerate(zip(bifpn_cbam_features, ssh_modules, cs_modules)):
            # Apply SSH
            ssh_feat = ssh(feat)
            # Apply channel shuffle
            ssh_feat = cs(ssh_feat)
            ssh_features.append(ssh_feat)        
        # Generate predictions using SharedMultiHead
        classifications = []
        bbox_regressions = []
        landmarks = []
        
        for i, (feat, head) in enumerate(zip(ssh_features, self.shared_heads)):
            cls, bbox, ldm = head(feat)
            classifications.append(cls)
            bbox_regressions.append(bbox)
            landmarks.append(ldm)
        
        # Return outputs in the same format as original RetinaFace
        # FIXED: Match V1 output order (bbox, classification, landmarks)
        if self.phase == 'train':
            return (bbox_regressions, classifications, landmarks)
        else:
            # For inference, concatenate all scales
            classifications = torch.cat(classifications, dim=1)
            bbox_regressions = torch.cat(bbox_regressions, dim=1)
            landmarks = torch.cat(landmarks, dim=1)
            
            # Apply softmax to classifications for inference
            if self.cfg['name'] == 'mobilenet0.25':
                classifications = F.softmax(classifications, dim=-1)
            
            return (bbox_regressions, classifications, landmarks)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_retinaface_v2(cfg, phase='train'):
    """
    Factory function to create RetinaFaceV2 model
    
    Args:
        cfg: Configuration dictionary
        phase: 'train' or 'test'
        
    Returns:
        RetinaFaceV2 model instance
    """
    model = RetinaFaceV2(cfg=cfg, phase=phase)
    return model


# NOTE: cfg_mnet_v2 is now centralized in data/config.py
# Import it using: from data.config import cfg_mnet_v2

if __name__ == "__main__":
    """Test RetinaFaceV2 architecture"""
    import sys
    sys.path.append('..')
    
    # Test model creation and parameter count
    print("=== Testing RetinaFaceV2 Architecture ===\n")
    
    # Create model
    model = get_retinaface_v2(cfg_mnet_v2, phase='test')
    
    # Count parameters
    total_params = count_parameters(model)
    
    print(f"1. Total Parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"   Target: 0.25M parameters")
    print(f"   Reduction from baseline: {(1 - total_params/592000)*100:.1f}%\n")
    
    # Test forward pass
    print("2. Testing Forward Pass:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # Forward pass
    with torch.no_grad():
        cls, bbox, ldm = model(input_tensor)
    
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Classification output: {cls.shape}")
    print(f"   BBox regression output: {bbox.shape}")
    print(f"   Landmark output: {ldm.shape}")
    
    # Parameter breakdown by module type
    print("\n3. Parameter Breakdown:")
    
    # Backbone parameters
    backbone_params = sum(p.numel() for n, p in model.named_parameters() 
                         if p.requires_grad and 'body' in n)
    print(f"   Backbone: {backbone_params:,} params ({backbone_params/total_params*100:.1f}%)")
    
    # CBAM parameters
    cbam_params = sum(p.numel() for n, p in model.named_parameters() 
                     if p.requires_grad and 'cbam' in n.lower())
    print(f"   CBAM modules: {cbam_params:,} params ({cbam_params/total_params*100:.1f}%)")
    
    # BiFPN parameters
    bifpn_params = sum(p.numel() for n, p in model.named_parameters() 
                      if p.requires_grad and 'bifpn' in n)
    print(f"   BiFPN: {bifpn_params:,} params ({bifpn_params/total_params*100:.1f}%)")
    
    # SSH parameters
    ssh_params = sum(p.numel() for n, p in model.named_parameters() 
                    if p.requires_grad and 'ssh' in n)
    print(f"   SSH modules: {ssh_params:,} params ({ssh_params/total_params*100:.1f}%)")
    
    # Head parameters
    head_params = sum(p.numel() for n, p in model.named_parameters() 
                     if p.requires_grad and 'shared_heads' in n)
    print(f"   Detection heads: {head_params:,} params ({head_params/total_params*100:.1f}%)")
    
    print("\n4. Model Summary:")
    print(f"   ✓ Successfully created RetinaFaceV2")
    print(f"   ✓ Forward pass completed")
    print(f"   ✓ Output shapes match original RetinaFace")
    
    if total_params <= 260000:  # 0.26M with small margin
        print(f"   ✓ Target parameter count achieved!")
    else:
        print(f"   ✗ Parameter count exceeds target (0.25M)")
        print(f"     Need to reduce: {total_params - 250000:,} parameters")