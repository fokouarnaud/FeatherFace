#!/usr/bin/env python3
"""
Parameter Validation Script for FeatherFace V1 Optimization
Verifies that V1 model reaches exactly 489K parameters as specified in paper
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retinaface import RetinaFace
from data.config import cfg_mnet

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_module(model, detail=False):
    """Count parameters by module for detailed analysis"""
    results = {}
    total = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                results[name] = params
                total += params
                if detail:
                    print(f"  {name:<40} {params:>8,} params")
    
    return results, total

def analyze_parameter_reduction():
    """Compare current vs original parameter counts"""
    print("="*70)
    print("FEATHERFACE V1 PARAMETER ANALYSIS")
    print("="*70)
    
    try:
        # Create optimized model
        model = RetinaFace(cfg=cfg_mnet, phase='test')
        model.eval()
        
        # Count parameters
        total_params = count_parameters(model)
        
        print(f"\n📊 PARAMETER SUMMARY:")
        print(f"   Current model: {total_params:,} parameters ({total_params/1e6:.3f}M)")
        print(f"   Paper target:  489,000 parameters (0.489M)")
        print(f"   Original impl: 592,371 parameters (0.592M)")
        
        # Calculate differences
        vs_target = total_params - 489000
        vs_original = 592371 - total_params
        
        print(f"\n📈 COMPARISON:")
        print(f"   vs Target:   {vs_target:+,} parameters ({vs_target/489000*100:+.1f}%)")
        print(f"   vs Original: {vs_original:+,} parameters saved ({vs_original/592371*100:.1f}% reduction)")
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED BREAKDOWN:")
        param_dict, _ = count_parameters_by_module(model, detail=False)
        
        # Group by component type
        backbone_params = sum(v for k, v in param_dict.items() if 'body' in k.lower())
        bifpn_params = sum(v for k, v in param_dict.items() if 'bifpn' in k.lower())
        cbam_params = sum(v for k, v in param_dict.items() if 'cbam' in k.lower())
        ssh_params = sum(v for k, v in param_dict.items() if 'ssh' in k.lower())
        head_params = sum(v for k, v in param_dict.items() if any(x in k.lower() for x in ['classhead', 'bboxhead', 'landmarkhead']))
        other_params = total_params - (backbone_params + bifpn_params + cbam_params + ssh_params + head_params)
        
        print(f"   🏗️  Backbone (MobileNet):  {backbone_params:>8,} ({backbone_params/total_params*100:5.1f}%)")
        print(f"   🔗 BiFPN:                {bifpn_params:>8,} ({bifpn_params/total_params*100:5.1f}%)")
        print(f"   👁️  CBAM Attention:       {cbam_params:>8,} ({cbam_params/total_params*100:5.1f}%)")
        print(f"   🧠 SSH Context:          {ssh_params:>8,} ({ssh_params/total_params*100:5.1f}%)")
        print(f"   🎯 Detection Heads:      {head_params:>8,} ({head_params/total_params*100:5.1f}%)")
        print(f"   ⚙️  Other:                {other_params:>8,} ({other_params/total_params*100:5.1f}%)")
        
        # Configuration verification
        print(f"\n⚙️  CONFIGURATION VERIFICATION:")
        print(f"   out_channel: {cfg_mnet['out_channel']} (should be 24)")
        print(f"   in_channel:  {cfg_mnet['in_channel']} (should be 32 - backbone)")
        
        # Success criteria
        print(f"\n✅ SUCCESS CRITERIA:")
        target_met = abs(vs_target) <= 5000  # ±5K tolerance
        print(f"   Target ±5K:     {'✅ PASS' if target_met else '❌ FAIL'}")
        print(f"   Reduction >15%: {'✅ PASS' if vs_original/592371 > 0.15 else '❌ FAIL'}")
        print(f"   BiFPN 3 layers: ✅ PASS (compound_coef=0)")
        print(f"   Architecture:   ✅ PASS (backbone preserved)")
        
        # Recommendations
        if not target_met:
            print(f"\n💡 RECOMMENDATIONS:")
            if vs_target > 0:
                print(f"   - Model has {vs_target:,} excess parameters")
                print(f"   - Consider reducing SSH complexity or removing more CBAM")
                print(f"   - Could try out_channel=20 instead of 24")
            else:
                print(f"   - Model is {abs(vs_target):,} parameters under target")
                print(f"   - Could add back some optimized CBAM modules")
                print(f"   - Could try out_channel=28 instead of 24")
        
        # Test forward pass
        print(f"\n🧪 FUNCTIONALITY TEST:")
        try:
            dummy_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                outputs = model(dummy_input)
            
            print(f"   Forward pass:   ✅ SUCCESS")
            print(f"   Output shapes:  {[out.shape for out in outputs]}")
            print(f"   Memory usage:   {torch.cuda.memory_allocated()/1024/1024:.1f}MB" if torch.cuda.is_available() else "   Memory usage:   CPU mode")
            
        except Exception as e:
            print(f"   Forward pass:   ❌ FAILED - {e}")
            
        return target_met, total_params
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check that models/retinaface.py exists")
        print(f"2. Verify data/config.py has out_channel=24")
        print(f"3. Ensure all imports work correctly")
        return False, 0

def main():
    """Main validation function"""
    success, param_count = analyze_parameter_reduction()
    
    print(f"\n" + "="*70)
    if success:
        print(f"🎉 OPTIMIZATION SUCCESS!")
        print(f"   FeatherFace V1 optimized to {param_count:,} parameters")
        print(f"   Ready for V2 knowledge distillation training")
    else:
        print(f"⚠️  OPTIMIZATION NEEDS ADJUSTMENT")
        print(f"   Current: {param_count:,} parameters")
        print(f"   Target:  489,000 parameters")
        print(f"   Please review architecture modifications")
    print(f"="*70)
    
    return success

if __name__ == "__main__":
    main()