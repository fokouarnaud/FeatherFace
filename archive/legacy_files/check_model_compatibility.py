"""
Quick check if a model file is compatible with current architecture
"""

import torch
import sys
sys.path.append('.')

def check_model_compatibility(model_path):
    """Check if a saved model is compatible with current RetinaFace"""
    print(f"Checking: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        # Check architecture markers
        has_bifpn = any('bifpn' in k for k in state_dict.keys())
        has_fpn = any('fpn.' in k for k in state_dict.keys())
        has_ssh = any('ssh' in k for k in state_dict.keys())
        has_cbam = any('cbam' in k for k in state_dict.keys())
        
        print(f"\nArchitecture Analysis:")
        print(f"  ✓ BiFPN: {'Yes' if has_bifpn else 'No'}")
        print(f"  ✓ FPN: {'Yes' if has_fpn else 'No'}")
        print(f"  ✓ SSH: {'Yes' if has_ssh else 'No'}")
        print(f"  ✓ CBAM: {'Yes' if has_cbam else 'No'}")
        
        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.3f}M)")
        
        # Verdict
        if has_bifpn and not has_fpn:
            print("\n✅ COMPATIBLE: This model uses the current BiFPN architecture")
            return True
        elif has_fpn and not has_bifpn:
            print("\n❌ INCOMPATIBLE: This model uses old FPN architecture")
            print("   → Solution: Retrain with notebook 01_train_evaluate_featherface.ipynb")
            return False
        else:
            print("\n⚠️  UNKNOWN: Cannot determine architecture")
            return None
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    import os
    
    # Check the downloaded model
    model_path = "./weights/mobilenet0.25_Final.pth"
    
    if os.path.exists(model_path):
        compatible = check_model_compatibility(model_path)
        
        if not compatible:
            print("\n" + "="*60)
            print("RECOMMENDED ACTION:")
            print("="*60)
            print("1. Open notebook: 01_train_evaluate_featherface.ipynb")
            print("2. Train a fresh V1 model (will create compatible teacher)")
            print("3. Then use notebook: 03_train_evaluate_featherface_v2.ipynb")
            print("4. This ensures perfect compatibility!")
    else:
        print(f"Model not found: {model_path}")
        print("You need to train V1 first anyway!")