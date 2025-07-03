#!/usr/bin/env python3
"""
Quick test to verify parameter count with new configuration
"""

def quick_parameter_test():
    """Quick test of parameter count with current configuration."""
    try:
        import sys
        import os
        from pathlib import Path
        
        # Ensure we can import from project root (go up 2 levels from scripts/validation/)
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        print("Testing FeatherFace V1 parameter count...")
        print("=" * 50)
        
        # Import configuration first
        from data.config import cfg_mnet
        print(f"Current out_channel: {cfg_mnet['out_channel']}")
        
        # Try to import torch and model
        try:
            import torch
            from models.retinaface import RetinaFace
            
            # Create model
            model = RetinaFace(cfg=cfg_mnet, phase='test')
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            target = 489000
            diff = total_params - target
            
            print(f"\nResults:")
            print(f"Total parameters: {total_params:,}")
            print(f"Target: {target:,}")
            print(f"Difference: {diff:+,}")
            print(f"Status: {'✅ ACHIEVED' if abs(diff) <= 5000 else '⚠️ CLOSE' if abs(diff) <= 15000 else '❌ MISSED'}")
            
            # Quick component breakdown
            print(f"\nKey Components:")
            ssh_total = 0
            for name, module in model.named_children():
                params = sum(p.numel() for p in module.parameters())
                if 'ssh' in name:
                    ssh_total += params
                    print(f"  {name}: {params:,}")
                elif name in ['body', 'bifpn']:
                    print(f"  {name}: {params:,}")
            
            print(f"  SSH total: {ssh_total:,} ({ssh_total/total_params*100:.1f}%)")
            
            # Suggest adjustment if needed
            if abs(diff) > 5000:
                if diff > 0:
                    new_channel = cfg_mnet['out_channel'] - max(1, abs(diff) // 4000)
                    print(f"\nSuggestion: Try out_channel = {new_channel}")
                else:
                    new_channel = cfg_mnet['out_channel'] + max(1, abs(diff) // 4000)
                    print(f"\nSuggestion: Try out_channel = {new_channel}")
            
            return total_params
            
        except ImportError as e:
            print(f"Could not import torch/model: {e}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    quick_parameter_test()