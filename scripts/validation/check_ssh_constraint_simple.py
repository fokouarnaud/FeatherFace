#!/usr/bin/env python3
"""
Simple SSH constraint verification (no PyTorch required)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_ssh_constraint_simple():
    """Check SSH constraint without creating model."""
    print("SSH Constraint Verification (Simple)")
    print("=" * 40)
    
    try:
        # Import config
        from data.config import cfg_mnet
        
        out_channel = cfg_mnet['out_channel']
        print(f"Current out_channel: {out_channel}")
        
        # Check SSH constraint
        remainder = out_channel % 4
        is_valid = remainder == 0
        
        print(f"SSH constraint check: {out_channel} % 4 = {remainder}")
        print(f"Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        if is_valid:
            print(f"✅ SSH module will accept out_channel = {out_channel}")
            print(f"   Division result: {out_channel} ÷ 4 = {out_channel // 4}")
            
            # Estimate parameters (based on previous analysis)
            # ~9.6K parameters per out_channel unit
            estimated_params = out_channel * 9600
            target = 489000
            diff = estimated_params - target
            
            print(f"\nParameter estimation:")
            print(f"Target: {target:,}")
            print(f"Estimated: {estimated_params:,}")
            print(f"Difference: {diff:+,}")
            
            if abs(diff) <= 10000:
                print(f"Status: ✅ CLOSE TO TARGET")
            else:
                print(f"Status: ⚠️ NEEDS VERIFICATION")
                
        else:
            # Suggest nearest valid values
            lower = (out_channel // 4) * 4
            upper = lower + 4
            print(f"\n❌ SSH constraint VIOLATED!")
            print(f"Nearest valid values: {lower} or {upper}")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

if __name__ == "__main__":
    success = check_ssh_constraint_simple()
    if success:
        print(f"\n🎉 SSH constraint is satisfied!")
        print(f"Configuration is ready for model creation.")
    else:
        print(f"\n❌ SSH constraint validation failed.")
        print(f"Please update out_channel to be divisible by 4.")