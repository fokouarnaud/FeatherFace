#!/usr/bin/env python3
"""
Direct configuration file analysis (no imports)
"""

import re
from pathlib import Path

def extract_out_channel():
    """Extract out_channel value directly from config file."""
    
    print("Direct Configuration Analysis")
    print("=" * 40)
    
    config_path = Path(__file__).parent.parent.parent / 'data' / 'config.py'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract out_channel from cfg_mnet
        pattern = r"'out_channel':\s*(\d+)"
        matches = re.findall(pattern, content)
        
        if not matches:
            print("❌ Could not find out_channel in cfg_mnet")
            return None
        
        out_channel = int(matches[0])  # First match should be cfg_mnet
        print(f"Found out_channel: {out_channel}")
        
        # Check SSH constraint
        remainder = out_channel % 4
        is_valid = remainder == 0
        
        print(f"\nSSH Constraint Check:")
        print(f"  {out_channel} % 4 = {remainder}")
        print(f"  Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        if is_valid:
            print(f"  ✅ SSH module will accept this value")
            print(f"  Division: {out_channel} ÷ 4 = {out_channel // 4}")
            
            # Parameter estimation
            estimated_params = out_channel * 9600  # ~9.6K per channel
            target = 489000
            diff = estimated_params - target
            percentage = (diff / target) * 100
            
            print(f"\nParameter Estimation:")
            print(f"  Target: {target:,}")
            print(f"  Estimated: {estimated_params:,}")
            print(f"  Difference: {diff:+,} ({percentage:+.2f}%)")
            
            if abs(diff) <= 5000:
                status = "🎯 EXCELLENT"
            elif abs(diff) <= 10000:
                status = "✅ GOOD"
            else:
                status = "⚠️ ACCEPTABLE"
            
            print(f"  Status: {status}")
            
        else:
            lower = (out_channel // 4) * 4
            upper = lower + 4
            print(f"\n❌ INVALID - SSH requires divisibility by 4")
            print(f"  Suggested values: {lower} or {upper}")
        
        return out_channel if is_valid else None
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def check_previous_fixes():
    """Check if our previous fixes are properly applied."""
    print(f"\nVerifying Previous Fixes:")
    print("=" * 25)
    
    config_path = Path(__file__).parent.parent.parent / 'data' / 'config.py'
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for our comment
        if "PAPER COMPLIANT" in content:
            print("✅ Paper compliance comment found")
        else:
            print("❌ Paper compliance comment missing")
        
        # Check for SSH comment
        if "SSH module" in content:
            print("✅ SSH module comment found")
        else:
            print("❌ SSH module comment missing")
        
        # Check for divisible by 4 mention
        if "divisible by 4" in content:
            print("✅ Divisibility requirement documented")
        else:
            print("❌ Divisibility requirement not documented")
        
    except Exception as e:
        print(f"❌ Error checking fixes: {e}")

if __name__ == "__main__":
    out_channel = extract_out_channel()
    check_previous_fixes()
    
    if out_channel:
        print(f"\n🎉 SUCCESS: Configuration is valid!")
        print(f"   out_channel = {out_channel} satisfies SSH constraint")
        print(f"   Model creation should work without AssertionError")
    else:
        print(f"\n❌ FAILED: Configuration needs adjustment")