#!/usr/bin/env python3
"""
Detailed inspection script for teacher model
Provides comprehensive analysis of the model architecture and state dict
"""

import sys
import torch
import traceback
from pathlib import Path
from collections import defaultdict

def inspect_teacher_model():
    """Detailed inspection of teacher model"""
    print("=== Teacher Model Detailed Inspection ===")
    
    teacher_weights = Path('./weights/mobilenet0.25_Final.pth')
    
    if not teacher_weights.exists():
        print(f"❌ Teacher model not found: {teacher_weights}")
        return False
    
    try:
        print(f"Loading teacher model from: {teacher_weights}")
        
        # Load checkpoint
        checkpoint = torch.load(teacher_weights, map_location='cpu')
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        print(f"State dict type: {type(state_dict)}")
        print(f"Total keys in state dict: {len(state_dict)}")
        
        # Group keys by module type
        key_groups = defaultdict(list)
        
        for key in state_dict.keys():
            if 'bifpn' in key.lower():
                key_groups['BiFPN'].append(key)
            elif 'ssh' in key.lower():
                key_groups['SSH'].append(key)
            elif 'cbam' in key.lower():
                key_groups['CBAM'].append(key)
            elif 'fpn' in key.lower():
                key_groups['FPN'].append(key)
            elif 'body' in key.lower():
                key_groups['Backbone'].append(key)
            elif any(x in key.lower() for x in ['classhead', 'class']):
                key_groups['ClassHead'].append(key)
            elif any(x in key.lower() for x in ['bboxhead', 'bbox']):
                key_groups['BboxHead'].append(key)
            elif any(x in key.lower() for x in ['landmark', 'ldm']):
                key_groups['LandmarkHead'].append(key)
            elif key.lower() in ['total_ops', 'total_params']:
                key_groups['Metadata'].append(key)
            else:
                key_groups['Other'].append(key)
        
        # Display analysis
        print(f"\n{'='*60}")
        print("ARCHITECTURE ANALYSIS")
        print(f"{'='*60}")
        
        for group, keys in key_groups.items():
            print(f"\n{group}: {len(keys)} keys")
            if len(keys) <= 10:
                for key in keys:
                    print(f"  - {key}")
            else:
                print(f"  - {keys[0]}")
                print(f"  - {keys[1]}")
                print(f"  - ... ({len(keys)-4} more)")
                print(f"  - {keys[-2]}")
                print(f"  - {keys[-1]}")
        
        # Compatibility assessment
        print(f"\n{'='*60}")
        print("COMPATIBILITY ASSESSMENT")
        print(f"{'='*60}")
        
        has_bifpn = len(key_groups['BiFPN']) > 0
        has_ssh = len(key_groups['SSH']) > 0
        has_cbam = len(key_groups['CBAM']) > 0
        has_old_fpn = len(key_groups['FPN']) > 0
        has_heads = (len(key_groups['ClassHead']) > 0 and 
                    len(key_groups['BboxHead']) > 0 and 
                    len(key_groups['LandmarkHead']) > 0)
        
        print(f"BiFPN modules: {'✓' if has_bifpn else '✗'} ({len(key_groups['BiFPN'])} keys)")
        print(f"SSH modules: {'✓' if has_ssh else '✗'} ({len(key_groups['SSH'])} keys)")
        print(f"CBAM modules: {'✓' if has_cbam else '✗'} ({len(key_groups['CBAM'])} keys)")
        print(f"Old FPN: {'✓' if has_old_fpn else '✗'} ({len(key_groups['FPN'])} keys)")
        print(f"Detection heads: {'✓' if has_heads else '✗'}")
        print(f"Backbone: {'✓' if len(key_groups['Backbone']) > 0 else '✗'} ({len(key_groups['Backbone'])} keys)")
        
        # Parameter analysis
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        metadata_params = sum(state_dict[k].item() if k in ['total_params'] else 0 
                            for k in key_groups['Metadata'])
        
        print(f"\nParameter analysis:")
        print(f"  - Calculated parameters: {total_params:,}")
        if metadata_params > 0:
            print(f"  - Metadata parameters: {metadata_params:,}")
        print(f"  - Expected range: 592K ± 18K")
        print(f"  - Status: {'✅' if 574000 <= total_params <= 610000 else '⚠️'}")
        
        # Architecture classification
        print(f"\nArchitecture classification:")
        if has_bifpn and has_ssh and has_cbam and not has_old_fpn:
            architecture = "✅ Pure FeatherFace V1 (BiFPN + SSH + CBAM)"
            compatible = True
        elif has_bifpn and has_ssh and has_cbam and has_old_fpn:
            architecture = "⚠️ Hybrid architecture (BiFPN + SSH + CBAM + Old FPN)"
            compatible = True  # Probably still compatible
        elif has_old_fpn and not has_bifpn:
            architecture = "❌ Legacy architecture (Old FPN only)"
            compatible = False
        elif has_bifpn:
            architecture = "✅ Modern architecture (has BiFPN)"
            compatible = True
        else:
            architecture = "❓ Unknown architecture"
            compatible = False
            
        print(f"  - {architecture}")
        print(f"  - Compatibility: {'✅ COMPATIBLE' if compatible else '❌ INCOMPATIBLE'}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if compatible:
            print("✅ PROCEED with knowledge distillation")
            print("✅ Model appears compatible with FeatherFace V2")
            if has_old_fpn and has_bifpn:
                print("⚠️  Monitor training closely (hybrid architecture)")
        else:
            print("❌ DO NOT PROCEED with current model")
            print("🔄 Re-train V1 using notebook 01")
            
        return compatible
        
    except Exception as e:
        print(f"❌ Error inspecting teacher model: {e}")
        traceback.print_exc()
        return False

def main():
    """Run detailed teacher model inspection"""
    print("FeatherFace Teacher Model Detailed Inspection")
    print("=" * 70)
    
    success = inspect_teacher_model()
    
    print("\n" + "=" * 70)
    print("INSPECTION SUMMARY")
    print("=" * 70)
    
    if success:
        print("🎉 SUCCESS: Teacher model analysis complete")
        print("📋 Review recommendations above")
        return 0
    else:
        print("❌ FAILURE: Teacher model analysis failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())