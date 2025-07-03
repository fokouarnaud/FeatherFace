#!/usr/bin/env python3
"""
Auto-tune out_channel to achieve exactly 489K parameters
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_parameter_count(out_channel):
    """Get parameter count for a given out_channel value."""
    try:
        import torch
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        
        # Temporarily modify config
        cfg_temp = cfg_mnet.copy()
        cfg_temp['out_channel'] = out_channel
        
        # Create model with temporary config
        model = RetinaFace(cfg=cfg_temp, phase='test')
        total_params = sum(p.numel() for p in model.parameters())
        
        return total_params
        
    except Exception as e:
        print(f"Error with out_channel={out_channel}: {e}")
        return None

def auto_tune_parameters():
    """Auto-tune to achieve exactly 489K parameters."""
    target = 489000
    tolerance = 1000  # ±1K tolerance
    
    print("FeatherFace V1 Auto-Parameter Tuning")
    print("=" * 50)
    print(f"Target: {target:,} parameters (±{tolerance:,})")
    
    # Start with current config
    from data.config import cfg_mnet
    current_channel = cfg_mnet['out_channel']
    
    print(f"Starting from out_channel = {current_channel}")
    
    best_channel = current_channel
    best_diff = float('inf')
    best_params = None
    
    # Test range around current value
    test_range = range(max(32, current_channel - 5), current_channel + 10)
    
    print(f"\nTesting out_channel values from {min(test_range)} to {max(test_range)}:")
    
    for channel in test_range:
        params = get_parameter_count(channel)
        if params is None:
            continue
            
        diff = abs(params - target)
        status = "✅" if diff <= tolerance else "⚠️" if diff <= 5000 else "❌"
        
        print(f"  out_channel={channel:2d}: {params:,} parameters (diff: {params-target:+,}) {status}")
        
        if diff < best_diff:
            best_diff = diff
            best_channel = channel
            best_params = params
    
    print(f"\n🎯 OPTIMAL CONFIGURATION:")
    print(f"  out_channel: {best_channel}")
    print(f"  Parameters: {best_params:,}")
    print(f"  Difference: {best_params - target:+,}")
    print(f"  Status: {'🎯 PERFECT' if best_diff <= tolerance else '✅ OPTIMAL'}")
    
    # Update config file if needed
    if best_channel != current_channel:
        print(f"\n🔧 UPDATING CONFIGURATION:")
        print(f"  Changing out_channel from {current_channel} to {best_channel}")
        
        # Read and update config file
        config_path = PROJECT_ROOT / 'data' / 'config.py'
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace the out_channel line
        old_line = f"'out_channel': {current_channel},"
        new_line = f"'out_channel': {best_channel},  # AUTO-TUNED: Calculated for exactly 489K parameters"
        
        content = content.replace(old_line, new_line)
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"  ✅ Configuration updated in {config_path}")
    else:
        print(f"\n✅ Configuration already optimal!")
    
    return best_channel, best_params

if __name__ == "__main__":
    auto_tune_parameters()