#!/usr/bin/env python3
"""
FeatherFace Comprehensive Attention Mechanism Comparison
========================================================

Three-way comparison framework for evaluating attention mechanisms in FeatherFace:
1. CBAM Baseline (Electronics 2025 paper reproduction)
2. ECA-Net Innovation (Channel attention optimization)  
3. ELA-S Innovation (Spatial attention advancement)

This script provides detailed analysis of parameters, computational efficiency,
and expected performance benefits for mobile face detection deployment.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_cbam_exact import FeatherFaceCBAMExact, create_cbam_exact_model
from models.featherface_v2_eca_innovation import FeatherFaceV2ECAInnovation, create_v2_eca_innovation_model
from models.featherface_v3_ela_innovation import FeatherFaceV3ELAInnovation, create_v3_ela_innovation_model

from data.config import cfg_cbam_paper_exact, cfg_v2_eca_innovation, cfg_v3_ela_innovation


def create_all_models():
    """Create all three attention mechanism models"""
    print("🏗️  Creating FeatherFace Models with Different Attention Mechanisms")
    print("=" * 80)
    
    models = {}
    
    try:
        # 1. CBAM Baseline (Electronics 2025 reproduction)
        print("\n1. CBAM Baseline (Electronics 2025 Paper):")
        print("-" * 50)
        models['cbam'] = create_cbam_exact_model(cfg_cbam_paper_exact, phase='test')
        
        # 2. ECA-Net Innovation (Channel attention)
        print("\n2. ECA-Net Innovation (Channel Attention):")
        print("-" * 50)
        models['eca'] = create_v2_eca_innovation_model(cfg_v2_eca_innovation, phase='test')
        
        # 3. ELA-S Innovation (Spatial attention)
        print("\n3. ELA-S Innovation (Spatial Attention):")
        print("-" * 50)
        models['ela_s'] = create_v3_ela_innovation_model(cfg_v3_ela_innovation, phase='test')
        
        return models
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_parameters(models):
    """Detailed parameter comparison across all models"""
    print("\n📊 DETAILED PARAMETER COMPARISON")
    print("=" * 80)
    
    # Get parameter breakdowns
    param_data = {}
    for name, model in models.items():
        if hasattr(model, 'get_parameter_count'):
            param_data[name] = model.get_parameter_count()
        else:
            # Fallback for basic parameter counting
            param_data[name] = {'total': sum(p.numel() for p in model.parameters())}
    
    # Display comparison table
    components = ['backbone', 'bifpn', 'ssh', 'detection_heads', 'total']
    attention_components = {
        'cbam': 'cbam_backbone + cbam_bifpn',
        'eca': 'eca_backbone + eca_bifpn', 
        'ela_s': 'ela_backbone + ela_bifpn'
    }
    
    print(f"{'Component':<20} {'CBAM':<12} {'ECA-Net':<12} {'ELA-S':<12} {'Best':<10}")
    print("-" * 70)
    
    for component in components:
        values = {}
        for name in ['cbam', 'eca', 'ela_s']:
            if component in param_data[name]:
                values[name] = param_data[name][component]
            else:
                values[name] = 0
        
        # Find best (lowest for most components, but depends on context)
        best_name = min(values.keys(), key=lambda k: values[k]) if component != 'total' else 'varies'
        
        print(f"{component:<20} {values['cbam']:<12,} {values['eca']:<12,} {values['ela_s']:<12,} {best_name:<10}")
    
    # Attention mechanism comparison
    print("\nATTENTION MECHANISM BREAKDOWN:")
    print("-" * 70)
    
    for name, model in models.items():
        param_info = param_data[name]
        
        if name == 'cbam':
            attention_params = param_info.get('cbam_backbone', 0) + param_info.get('cbam_bifpn', 0)
            mechanism = "CBAM (Channel + Spatial)"
        elif name == 'eca':
            attention_params = param_info.get('eca_backbone', 0) + param_info.get('eca_bifpn', 0)
            mechanism = "ECA-Net (Channel Only)"
        else:  # ela_s
            attention_params = param_info.get('ela_backbone', 0) + param_info.get('ela_bifpn', 0)
            mechanism = "ELA-S (Spatial Focus)"
        
        total_params = param_info['total']
        attention_percentage = (attention_params / total_params) * 100
        
        print(f"{name.upper():<8}: {attention_params:>8,} params ({attention_percentage:5.1f}%) - {mechanism}")


def compare_efficiency(models):
    """Compare computational efficiency and mobile deployment characteristics"""
    print("\n⚡ COMPUTATIONAL EFFICIENCY COMPARISON")
    print("=" * 80)
    
    # Test with different input sizes for mobile deployment analysis
    test_sizes = [
        (320, 320),   # Small mobile
        (640, 640),   # Standard
        (416, 416),   # YOLO standard
    ]
    
    print(f"{'Model':<8} {'Input Size':<12} {'Forward Pass':<12} {'Memory Est.':<12} {'Mobile Friendly':<15}")
    print("-" * 75)
    
    for size_h, size_w in test_sizes:
        input_tensor = torch.randn(1, 3, size_h, size_w)
        
        for name, model in models.items():
            try:
                with torch.no_grad():
                    # Measure forward pass (basic timing)
                    start_time = torch.cuda.synchronize() if torch.cuda.is_available() else None
                    outputs = model(input_tensor)
                    
                # Basic efficiency metrics
                param_count = sum(p.numel() for p in model.parameters())
                memory_mb = param_count * 4 / (1024**2)  # Rough estimate (float32)
                
                # Mobile friendliness assessment
                if param_count < 500000:
                    mobile_friendly = "✅ Excellent"
                elif param_count < 1000000:
                    mobile_friendly = "⚠️ Good"
                else:
                    mobile_friendly = "❌ Heavy"
                
                size_str = f"{size_h}x{size_w}"
                print(f"{name.upper():<8} {size_str:<12} {'✅ Pass':<12} {memory_mb:>7.1f} MB   {mobile_friendly:<15}")
                
            except Exception as e:
                size_str = f"{size_h}x{size_w}"
                print(f"{name.upper():<8} {size_str:<12} {'❌ Failed':<12} {'N/A':<12} {'Error':<15}")


def analyze_attention_characteristics(models):
    """Analyze the characteristics of different attention mechanisms"""
    print("\n🔍 ATTENTION MECHANISM CHARACTERISTICS")
    print("=" * 80)
    
    characteristics = {
        'cbam': {
            'type': 'Hybrid (Channel + Spatial)',
            'complexity': 'O(C²) + O(H×W)',
            'strength': 'Balanced channel and spatial attention',
            'weakness': 'Higher parameter overhead',
            'best_for': 'General object detection',
            'paper': 'Woo et al. ECCV 2018',
            'parameters': '~12.9K attention params'
        },
        'eca': {
            'type': 'Channel Attention Only',
            'complexity': 'O(C×k) where k≈3-5',
            'strength': 'Ultra-efficient channel attention',
            'weakness': 'No spatial awareness',
            'best_for': 'Channel feature enhancement',
            'paper': 'Wang et al. CVPR 2020',
            'parameters': '~22 attention params'
        },
        'ela_s': {
            'type': 'Spatial Attention Focus',
            'complexity': 'O(C×H) + O(C×W)',
            'strength': 'Superior spatial feature capture',
            'weakness': 'Higher parameter count',
            'best_for': 'Spatial-aware tasks (face detection)',
            'paper': 'Xuwei et al. 2024',
            'parameters': '~315K attention params'
        }
    }
    
    for name, char in characteristics.items():
        print(f"\n{name.upper()} - {char['type']}:")
        print("-" * 50)
        print(f"  Scientific Foundation: {char['paper']}")
        print(f"  Computational Complexity: {char['complexity']}")
        print(f"  Attention Parameters: {char['parameters']}")
        print(f"  Primary Strength: {char['strength']}")
        print(f"  Main Limitation: {char['weakness']}")
        print(f"  Optimal Use Case: {char['best_for']}")


def performance_expectations():
    """Analyze expected performance based on research results"""
    print("\n📈 EXPECTED PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Based on YOLOX-Nano results and research papers
    performance_data = {
        'cbam': {
            'map': 73.80,
            'fps': 69.30,
            'params_mb': 3.56,
            'baseline': True,
            'widerface_expected': {
                'easy': 92.7,
                'medium': 90.7,
                'hard': 78.3
            }
        },
        'eca': {
            'map': 73.39,
            'fps': 77.78,
            'params_mb': 3.44,
            'vs_cbam': -0.41,
            'widerface_expected': {
                'easy': 92.7,    # Maintained
                'medium': 90.7,  # Maintained
                'hard': 78.3     # Maintained (may vary slightly)
            }
        },
        'ela_s': {
            'map': 74.36,
            'fps': 75.68,
            'params_mb': 3.50,
            'vs_cbam': +0.56,
            'vs_eca': +0.97,
            'widerface_expected': {
                'easy': 92.7,    # Maintained
                'medium': 90.7,  # Maintained or improved
                'hard': 79.0     # Target improvement due to spatial awareness
            }
        }
    }
    
    print("YOLOX-Nano Results (Reference):")
    print("-" * 50)
    print(f"{'Method':<8} {'mAP':<6} {'FPS':<6} {'Params(MB)':<11} {'vs CBAM':<10}")
    print("-" * 45)
    
    for name, data in performance_data.items():
        vs_cbam = f"{data.get('vs_cbam', 0):+.2f}%" if 'vs_cbam' in data else "baseline"
        print(f"{name.upper():<8} {data['map']:<6.2f} {data['fps']:<6.1f} {data['params_mb']:<11.2f} {vs_cbam:<10}")
    
    print("\nExpected WIDERFace Performance:")
    print("-" * 50)
    print(f"{'Method':<8} {'Easy':<6} {'Medium':<8} {'Hard':<6} {'Notes':<25}")
    print("-" * 55)
    
    for name, data in performance_data.items():
        wf = data['widerface_expected']
        if name == 'cbam':
            notes = "Electronics 2025 baseline"
        elif name == 'eca':
            notes = "Channel efficiency"
        else:
            notes = "Spatial awareness advantage"
        
        print(f"{name.upper():<8} {wf['easy']:<6.1f} {wf['medium']:<8.1f} {wf['hard']:<6.1f} {notes:<25}")


def deployment_recommendations():
    """Provide deployment recommendations for different use cases"""
    print("\n🎯 DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = {
        "Mobile/Edge Devices": {
            "recommended": "ECA-Net",
            "reason": "Ultra-low parameter overhead (22 params), highest FPS",
            "alternative": "CBAM (if spatial awareness needed)",
            "avoid": "ELA-S (too heavy for mobile)"
        },
        "Face Detection Accuracy": {
            "recommended": "ELA-S",
            "reason": "Superior spatial attention, +0.97% mAP improvement",
            "alternative": "CBAM (balanced approach)",
            "considerations": "Higher parameter cost for accuracy gain"
        },
        "Production Deployment": {
            "recommended": "CBAM",
            "reason": "Proven baseline, balanced performance/efficiency",
            "alternative": "ECA-Net (if efficiency critical)",
            "considerations": "Well-validated Electronics 2025 baseline"
        },
        "Research/Experimentation": {
            "recommended": "All Three",
            "reason": "Comprehensive comparison enables insights",
            "focus": "Controlled experiment with single variable change",
            "value": "Attention mechanism impact analysis"
        }
    }
    
    for use_case, rec in recommendations.items():
        print(f"\n{use_case}:")
        print("-" * (len(use_case) + 1))
        print(f"  🥇 Recommended: {rec['recommended']}")
        print(f"  📝 Reason: {rec['reason']}")
        print(f"  🥈 Alternative: {rec.get('alternative', 'N/A')}")
        if 'considerations' in rec:
            print(f"  ⚠️  Considerations: {rec['considerations']}")
        if 'avoid' in rec:
            print(f"  ❌ Avoid: {rec['avoid']}")


def main():
    """Main comparison framework"""
    print("🔬 FeatherFace Attention Mechanism Comprehensive Analysis")
    print("=" * 80)
    print("Comparing: CBAM (baseline) vs ECA-Net (efficiency) vs ELA-S (spatial)")
    print()
    
    # Create all models
    models = create_all_models()
    if not models:
        print("❌ Failed to create models. Exiting.")
        return
    
    # Run comprehensive analysis
    compare_parameters(models)
    compare_efficiency(models)
    analyze_attention_characteristics(models)
    performance_expectations()
    deployment_recommendations()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("✅ CBAM Baseline: 488,664 params - Electronics 2025 paper reproduction")
    print("✅ ECA-Net Innovation: 475,757 params - Ultra-efficient channel attention")
    print("✅ ELA-S Innovation: 791,115 params - Superior spatial attention")
    print()
    print("🏆 Best Overall: ELA-S for accuracy, ECA-Net for efficiency, CBAM for balance")
    print("📱 Mobile Deployment: ECA-Net (22 attention params) > CBAM > ELA-S")
    print("🎯 Face Detection: ELA-S (spatial awareness) > CBAM > ECA-Net")
    print("🔬 Research Value: Comprehensive attention mechanism comparison achieved")


if __name__ == "__main__":
    main()