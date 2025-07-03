#!/usr/bin/env python3
"""
Quick check of current parameter count after adjustment
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_current_parameters():
    """Check current parameter count."""
    try:
        import torch
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        
        print("FeatherFace V1 Parameter Check")
        print("=" * 40)
        print(f"Current out_channel: {cfg_mnet['out_channel']}")
        
        # Create model
        model = RetinaFace(cfg=cfg_mnet, phase='test')
        total_params = sum(p.numel() for p in model.parameters())
        
        # Target analysis
        target = 489000
        diff = total_params - target
        percentage_diff = (diff / target) * 100
        
        print(f"\nResults:")
        print(f"Total parameters: {total_params:,}")
        print(f"Target: {target:,}")
        print(f"Difference: {diff:+,} ({percentage_diff:+.2f}%)")
        
        # Status
        if abs(diff) <= 2000:
            status = "🎯 PERFECT"
        elif abs(diff) <= 5000:
            status = "✅ ACHIEVED"
        elif abs(diff) <= 10000:
            status = "⚠️ CLOSE"
        else:
            status = "❌ MISSED"
            
        print(f"Status: {status}")
        
        # Further adjustment suggestion if needed
        if abs(diff) > 2000:
            if diff > 0:
                new_channel = cfg_mnet['out_channel'] - max(1, abs(diff) // 9600)
                print(f"\nFine-tuning: Try out_channel = {new_channel}")
            else:
                new_channel = cfg_mnet['out_channel'] + max(1, abs(diff) // 9600)
                print(f"\nFine-tuning: Try out_channel = {new_channel}")
        else:
            print(f"\n🎉 Parameter count is optimal!")
        
        return total_params
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    check_current_parameters()