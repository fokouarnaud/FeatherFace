#!/usr/bin/env python3
"""
Test script to verify teacher model compatibility detection
Validates that the teacher model from notebook 01 is properly recognized
"""

import sys
import torch
import traceback
from pathlib import Path

def test_teacher_compatibility():
    """Test teacher model compatibility detection"""
    print("=== Teacher Model Compatibility Test ===")
    
    teacher_weights = Path('./weights/mobilenet0.25_Final.pth')
    
    if not teacher_weights.exists():
        print(f"❌ Teacher model not found: {teacher_weights}")
        print("   Please train V1 first using notebook 01")
        return False
    
    try:
        print(f"Loading teacher model from: {teacher_weights}")
        
        # Load checkpoint
        checkpoint = torch.load(teacher_weights, map_location='cpu')
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        print(f"✓ Checkpoint loaded, {len(state_dict)} keys found")
        
        # Debug: Show sample keys
        sample_keys = list(state_dict.keys())[:10]
        print(f"Sample keys: {sample_keys}")
        
        # Test architecture detection
        has_bifpn = any('bifpn' in k.lower() for k in state_dict.keys())
        has_old_fpn = any('fpn.' in k for k in state_dict.keys())
        has_ssh = any('ssh' in k.lower() for k in state_dict.keys())
        has_cbam = any('cbam' in k.lower() for k in state_dict.keys())
        
        print(f"\nArchitecture detection results:")
        print(f"  - BiFPN modules: {'✓' if has_bifpn else '✗'}")
        print(f"  - SSH modules: {'✓' if has_ssh else '✗'}")
        print(f"  - CBAM modules: {'✓' if has_cbam else '✗'}")
        print(f"  - Old FPN: {'✓' if has_old_fpn else '✗'}")
        
        # Show some actual keys for each type
        if has_bifpn:
            bifpn_keys = [k for k in state_dict.keys() if 'bifpn' in k.lower()][:3]
            print(f"  BiFPN keys sample: {bifpn_keys}")
            
        if has_ssh:
            ssh_keys = [k for k in state_dict.keys() if 'ssh' in k.lower()][:3]
            print(f"  SSH keys sample: {ssh_keys}")
            
        if has_cbam:
            cbam_keys = [k for k in state_dict.keys() if 'cbam' in k.lower()][:3]
            print(f"  CBAM keys sample: {cbam_keys}")
        
        # Determine compatibility using same logic as notebook
        if has_bifpn and has_ssh and has_cbam and not has_old_fpn:
            compatibility = "COMPATIBLE (FeatherFace V1 architecture)"
            status = True
        elif has_old_fpn and not has_bifpn:
            compatibility = "INCOMPATIBLE (uses old FPN architecture)"
            status = False
        elif has_bifpn:
            compatibility = "COMPATIBLE (has BiFPN)"
            status = True
        else:
            compatibility = "UNKNOWN (assuming compatible)"
            status = True
            
        print(f"\nCompatibility assessment: {compatibility}")
        
        # Check parameters
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        print(f"\nParameter analysis:")
        print(f"  - Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
        
        # Validate parameter range
        if 592000 <= total_params <= 610000:
            print(f"  - ✅ Parameter count in expected range (592K±18K)")
            param_status = True
        else:
            print(f"  - ⚠️  Parameter count outside expected range")
            param_status = False
            
        # Overall assessment
        overall_status = status and param_status
        
        print(f"\n{'='*50}")
        print(f"OVERALL COMPATIBILITY: {'✅ COMPATIBLE' if overall_status else '❌ INCOMPATIBLE'}")
        print(f"{'='*50}")
        
        return overall_status
        
    except Exception as e:
        print(f"❌ Error testing teacher compatibility: {e}")
        traceback.print_exc()
        return False

def main():
    """Run teacher compatibility test"""
    print("FeatherFace Teacher Model Compatibility Test")
    print("=" * 60)
    
    success = test_teacher_compatibility()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 SUCCESS: Teacher model is compatible!")
        print("✅ Knowledge distillation can proceed")
        print("✅ Notebook 03 should work correctly")
        return 0
    else:
        print("❌ FAILURE: Teacher model compatibility issues")
        print("⚠️  May need to re-train V1 model")
        return 1

if __name__ == "__main__":
    sys.exit(main())