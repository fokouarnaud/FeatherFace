#!/usr/bin/env python3
"""
Test SSH constraint and verify model creation
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_ssh_constraint():
    """Test that out_channel satisfies SSH constraints."""
    try:
        from data.config import cfg_mnet
        
        print("SSH Constraint Verification")
        print("=" * 40)
        
        out_channel = cfg_mnet['out_channel']
        print(f"Current out_channel: {out_channel}")
        
        # Check SSH constraint
        divisible_by_4 = (out_channel % 4 == 0)
        print(f"Divisible by 4: {'✅ YES' if divisible_by_4 else '❌ NO'}")
        
        if not divisible_by_4:
            # Suggest nearest valid values
            lower = (out_channel // 4) * 4
            upper = lower + 4
            print(f"SSH requires out_channel % 4 == 0")
            print(f"Nearest valid values: {lower} or {upper}")
            return False
        
        # Try to create model
        print(f"\nTesting model creation...")
        
        import torch
        from models.retinaface import RetinaFace
        
        model = RetinaFace(cfg=cfg_mnet, phase='test')
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        
        # Parameter target analysis
        target = 489000
        diff = total_params - target
        percentage = (diff / target) * 100
        
        print(f"\nParameter Analysis:")
        print(f"Target: {target:,}")
        print(f"Actual: {total_params:,}")
        print(f"Difference: {diff:+,} ({percentage:+.2f}%)")
        
        if abs(diff) <= 5000:
            print(f"Status: ✅ TARGET ACHIEVED")
        elif abs(diff) <= 10000:
            print(f"Status: ⚠️ CLOSE TO TARGET")
        else:
            print(f"Status: ❌ NEEDS ADJUSTMENT")
        
        return True
        
    except AssertionError as e:
        print(f"❌ SSH Constraint Error: {e}")
        print(f"The SSH module requires out_channel to be divisible by 4")
        return False
    except Exception as e:
        print(f"❌ Model Creation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ssh_constraint()
    if success:
        print(f"\n🎉 All tests passed! Model is ready for training.")
    else:
        print(f"\n❌ Tests failed. Please fix configuration.")