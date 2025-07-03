#!/usr/bin/env python3
"""
Quick parameter count test for debugging
"""

def test_model_parameters():
    """Test model parameters and architecture."""
    try:
        import torch
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        
        print("Creating FeatherFace V1 model...")
        print(f"Configuration: out_channel={cfg_mnet['out_channel']}")
        
        # Create model
        model = RetinaFace(cfg=cfg_mnet, phase='test')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Parameter Analysis:")
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f}M)")
        
        # Target analysis
        target = 489000
        diff = total_params - target
        print(f"Target: {target:,}")
        print(f"Difference: {diff:+,}")
        print(f"Status: {'✅ ACHIEVED' if abs(diff) <= 5000 else '❌ MISSED'}")
        
        # Component breakdown
        print(f"\nComponent Breakdown:")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            percentage = (params / total_params) * 100
            print(f"  {name}: {params:,} ({percentage:.1f}%)")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        
        return total_params
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("FeatherFace V1 Parameter Test")
    print("=" * 40)
    test_model_parameters()