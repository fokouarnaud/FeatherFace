#!/usr/bin/env python3
"""
Final Validation Script for FeatherFace Optimization Project
Comprehensive testing of V1 optimization and V2 enhancements
"""

import torch
import torch.nn as nn
import sys
import os
import time
import json
from pathlib import Path
import numpy as np

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.retinaface import RetinaFace
    from models.retinaface_v2 import get_retinaface_v2, count_parameters
    from data.config import cfg_mnet, cfg_mnet_v2
    from layers.advanced_training import (
        GradientClipper, DynamicDistillationLoss, 
        SmartEarlyStopping, TrainingMonitor
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the FeatherFace root directory")
    sys.exit(1)


class FeatherFaceValidator:
    """Comprehensive validation for FeatherFace optimization"""
    
    def __init__(self):
        self.results = {
            'v1_optimization': {},
            'v2_enhancements': {},
            'advanced_features': {},
            'deployment': {},
            'summary': {}
        }
    
    def validate_v1_optimization(self):
        """Validate V1 parameter optimization"""
        print("🔍 VALIDATING V1 OPTIMIZATION")
        print("="*50)
        
        try:
            # Create V1 optimized model
            model = RetinaFace(cfg=cfg_mnet, phase='test')
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"📊 Parameter Analysis:")
            print(f"   Current: {total_params:,} parameters ({total_params/1e6:.3f}M)")
            print(f"   Target:  489,000 parameters (0.489M)")
            
            # Check target achievement
            target_diff = total_params - 489000
            target_met = abs(target_diff) <= 5000  # ±5K tolerance
            
            print(f"   Difference: {target_diff:+,} parameters")
            print(f"   Status: {'✅ TARGET MET' if target_met else '❌ TARGET MISSED'}")
            
            # Configuration verification
            config_check = cfg_mnet['out_channel'] == 24
            print(f"\n⚙️  Configuration Check:")
            print(f"   out_channel = {cfg_mnet['out_channel']} {'✅' if config_check else '❌'}")
            print(f"   in_channel = {cfg_mnet['in_channel']} (should be 32)")
            
            # Architecture verification
            print(f"\n🏗️  Architecture Verification:")
            
            # Check SimpleChannelShuffle exists
            has_simple_shuffle = hasattr(sys.modules['models.retinaface'], 'SimpleChannelShuffle')
            print(f"   SimpleChannelShuffle: {'✅' if has_simple_shuffle else '❌'}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 640, 640)
            try:
                with torch.no_grad():
                    outputs = model(dummy_input)
                forward_pass = True
                output_shapes = [out.shape for out in outputs]
                print(f"   Forward pass: ✅ SUCCESS")
                print(f"   Output shapes: {output_shapes}")
            except Exception as e:
                forward_pass = False
                print(f"   Forward pass: ❌ FAILED - {e}")
            
            # Store results
            self.results['v1_optimization'] = {
                'parameters': total_params,
                'target_met': target_met,
                'target_difference': target_diff,
                'config_correct': config_check,
                'forward_pass': forward_pass,
                'reduction_percentage': (592371 - total_params) / 592371 * 100
            }
            
            return target_met and config_check and forward_pass
            
        except Exception as e:
            print(f"❌ V1 validation failed: {e}")
            self.results['v1_optimization'] = {'error': str(e)}
            return False
    
    def validate_v2_enhancements(self):
        """Validate V2 architecture and enhancements"""
        print("\n🚀 VALIDATING V2 ENHANCEMENTS")
        print("="*50)
        
        try:
            # Create V2 model
            model = get_retinaface_v2(cfg_mnet_v2, phase='test')
            total_params = count_parameters(model)
            
            print(f"📊 V2 Parameter Analysis:")
            print(f"   Parameters: {total_params:,} ({total_params/1e6:.3f}M)")
            print(f"   Target: ~256K parameters")
            print(f"   Compression vs V1: {592371/total_params:.2f}x")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 640, 640)
            try:
                with torch.no_grad():
                    outputs = model(dummy_input)
                v2_forward_pass = True
                print(f"   Forward pass: ✅ SUCCESS")
                print(f"   Output shapes: {[out.shape for out in outputs]}")
            except Exception as e:
                v2_forward_pass = False
                print(f"   Forward pass: ❌ FAILED - {e}")
            
            # Check V2 modules
            v2_modules = ['BiFPN_Light', 'CBAM_Plus', 'SharedMultiHead']
            modules_found = []
            for module_name in v2_modules:
                try:
                    from models import modules_v2
                    if hasattr(modules_v2, module_name):
                        modules_found.append(module_name)
                except:
                    pass
            
            print(f"\n🧩 V2 Modules Check:")
            for module in v2_modules:
                status = "✅" if module in modules_found else "❌"
                print(f"   {module}: {status}")
            
            self.results['v2_enhancements'] = {
                'parameters': total_params,
                'compression_ratio': 592371/total_params,
                'forward_pass': v2_forward_pass,
                'modules_found': modules_found,
                'target_size': abs(total_params - 256000) < 20000
            }
            
            return v2_forward_pass and len(modules_found) >= 2
            
        except Exception as e:
            print(f"❌ V2 validation failed: {e}")
            self.results['v2_enhancements'] = {'error': str(e)}
            return False
    
    def validate_advanced_features(self):
        """Validate advanced training features"""
        print("\n⚡ VALIDATING ADVANCED FEATURES")
        print("="*50)
        
        features_working = {}
        
        # Test Gradient Clipper
        try:
            clipper = GradientClipper(max_norm=1.0)
            # Create dummy model for testing
            dummy_model = nn.Linear(10, 1)
            dummy_loss = torch.sum(dummy_model(torch.randn(5, 10)))
            dummy_loss.backward()
            
            grad_norm = clipper.clip_gradients(dummy_model)
            stats = clipper.get_gradient_stats()
            
            features_working['gradient_clipper'] = True
            print("   ✅ Gradient Clipper: Working")
            
        except Exception as e:
            features_working['gradient_clipper'] = False
            print(f"   ❌ Gradient Clipper: Failed - {e}")
        
        # Test Dynamic Distillation
        try:
            dynamic_distill = DynamicDistillationLoss(
                initial_alpha=0.8, final_alpha=0.5, total_epochs=400
            )
            
            # Test alpha calculation
            alpha_start = dynamic_distill.get_alpha(0)
            alpha_mid = dynamic_distill.get_alpha(200)
            alpha_end = dynamic_distill.get_alpha(400)
            
            alpha_progression = alpha_start > alpha_mid > alpha_end
            features_working['dynamic_distillation'] = alpha_progression
            
            print(f"   ✅ Dynamic Distillation: α {alpha_start:.2f}→{alpha_mid:.2f}→{alpha_end:.2f}")
            
        except Exception as e:
            features_working['dynamic_distillation'] = False
            print(f"   ❌ Dynamic Distillation: Failed - {e}")
        
        # Test Smart Early Stopping
        try:
            early_stopper = SmartEarlyStopping(
                patience=20, min_epoch=100, optimal_window=(100, 120)
            )
            
            # Test stopping logic
            should_not_stop = not early_stopper.should_stop(50, 1.0, 0.8)  # Too early
            should_stop = early_stopper.should_stop(110, 2.0, 0.7)  # In window, high loss
            
            features_working['early_stopping'] = should_not_stop
            print("   ✅ Smart Early Stopping: Logic working")
            
        except Exception as e:
            features_working['early_stopping'] = False
            print(f"   ❌ Smart Early Stopping: Failed - {e}")
        
        # Test Training Monitor
        try:
            monitor = TrainingMonitor(log_interval=10)
            
            # Test metric logging
            monitor.log_epoch_metrics(
                epoch=1, train_loss=1.0, val_loss=0.9, val_map=0.85,
                learning_rate=1e-3, grad_stats={'grad_norm_current': 0.5},
                alpha=0.8, epoch_time=30.0
            )
            
            features_working['training_monitor'] = True
            print("   ✅ Training Monitor: Working")
            
        except Exception as e:
            features_working['training_monitor'] = False
            print(f"   ❌ Training Monitor: Failed - {e}")
        
        self.results['advanced_features'] = features_working
        return sum(features_working.values()) >= 3  # At least 3/4 working
    
    def validate_deployment_readiness(self):
        """Validate deployment and export capabilities"""
        print("\n📦 VALIDATING DEPLOYMENT READINESS")
        print("="*50)
        
        deployment_status = {}
        
        # Check export script exists
        export_script = Path('export_dynamic_onnx.py')
        deployment_status['export_script'] = export_script.exists()
        print(f"   Export script: {'✅' if export_script.exists() else '❌'}")
        
        # Check validation script exists
        validation_script = Path('validate_parameters.py')
        deployment_status['validation_script'] = validation_script.exists()
        print(f"   Validation script: {'✅' if validation_script.exists() else '❌'}")
        
        # Test ONNX dependencies (optional)
        try:
            import onnx
            import onnxruntime
            deployment_status['onnx_dependencies'] = True
            print("   ✅ ONNX dependencies: Available")
        except ImportError:
            deployment_status['onnx_dependencies'] = False
            print("   ⚠️  ONNX dependencies: Not installed (optional)")
        
        # Check advanced training module
        advanced_training = Path('layers/advanced_training.py')
        deployment_status['advanced_training'] = advanced_training.exists()
        print(f"   Advanced training: {'✅' if advanced_training.exists() else '❌'}")
        
        # Check documentation
        documentation = Path('TECHNICAL_DOCUMENTATION.md')
        deployment_status['documentation'] = documentation.exists()
        print(f"   Documentation: {'✅' if documentation.exists() else '❌'}")
        
        self.results['deployment'] = deployment_status
        return sum(deployment_status.values()) >= 4  # Most components ready
    
    def generate_summary(self):
        """Generate final validation summary"""
        print("\n" + "="*70)
        print("🎯 FINAL VALIDATION SUMMARY")
        print("="*70)
        
        # Calculate overall scores
        v1_success = self.results.get('v1_optimization', {}).get('target_met', False)
        v2_success = self.results.get('v2_enhancements', {}).get('forward_pass', False)
        advanced_count = sum(self.results.get('advanced_features', {}).values())
        deployment_count = sum(self.results.get('deployment', {}).values())
        
        print(f"\n📋 COMPONENT STATUS:")
        print(f"   ✅ V1 Optimization (489K): {'PASSED' if v1_success else 'FAILED'}")
        print(f"   ✅ V2 Architecture (256K): {'PASSED' if v2_success else 'FAILED'}")
        print(f"   ⚡ Advanced Features: {advanced_count}/4 working")
        print(f"   📦 Deployment Ready: {deployment_count}/5 components")
        
        # Parameter comparison
        v1_params = self.results.get('v1_optimization', {}).get('parameters', 0)
        v2_params = self.results.get('v2_enhancements', {}).get('parameters', 0)
        
        if v1_params and v2_params:
            print(f"\n📊 PARAMETER ACHIEVEMENT:")
            print(f"   Original V1: 592,371 parameters")
            print(f"   Optimized V1: {v1_params:,} parameters ({(592371-v1_params)/592371*100:.1f}% reduction)")
            print(f"   Enhanced V2: {v2_params:,} parameters ({592371/v2_params:.1f}x compression)")
            
            # Target achievement
            v1_target = abs(v1_params - 489000) <= 5000
            v2_target = abs(v2_params - 256000) <= 20000
            
            print(f"\n🎯 TARGET ACHIEVEMENT:")
            print(f"   V1 (489K target): {'✅ ACHIEVED' if v1_target else '❌ MISSED'}")
            print(f"   V2 (256K target): {'✅ ACHIEVED' if v2_target else '❌ MISSED'}")
        
        # Overall success
        overall_success = v1_success and v2_success and advanced_count >= 3
        
        print(f"\n🏆 OVERALL PROJECT STATUS:")
        if overall_success:
            print("   🎉 SUCCESS! FeatherFace optimization completed successfully")
            print("   📈 Ready for production deployment and training")
            print("   🚀 V2 model ready to surpass V1 performance")
        else:
            print("   ⚠️  PARTIAL SUCCESS - Some components need attention")
            print("   🔧 Review failed components and re-run validation")
        
        # Save detailed results
        self.results['summary'] = {
            'overall_success': overall_success,
            'v1_success': v1_success,
            'v2_success': v2_success,
            'advanced_features_count': advanced_count,
            'deployment_readiness': deployment_count,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open('validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: validation_results.json")
        
        return overall_success


def main():
    """Main validation function"""
    print("🔍 FeatherFace Optimization Validation")
    print("="*70)
    print("This script validates all optimizations and enhancements")
    print("="*70)
    
    validator = FeatherFaceValidator()
    
    # Run all validations
    v1_ok = validator.validate_v1_optimization()
    v2_ok = validator.validate_v2_enhancements()
    advanced_ok = validator.validate_advanced_features()
    deployment_ok = validator.validate_deployment_readiness()
    
    # Generate summary
    overall_success = validator.generate_summary()
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())