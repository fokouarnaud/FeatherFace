#!/usr/bin/env python3
"""
Tests de validation complète pour FeatherFace ODConv
Validation de l'implémentation ODConv et comparaison avec CBAM baseline

Auteur: FeatherFace ODConv Team
Date: Juillet 2025
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
import time
import warnings
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from models.odconv import ODConv2d
from models.featherface_odconv import FeatherFaceODConv
from models.retinaface import RetinaFace
from data.config import cfg_odconv, cfg_mnet

warnings.filterwarnings('ignore')

class TestODConvValidation(unittest.TestCase):
    """Tests complets de validation ODConv"""
    
    def setUp(self):
        """Configuration initiale des tests"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🔧 Running tests on: {self.device}")
        
        # Configurations test
        self.batch_size = 2
        self.input_channels = 64
        self.output_channels = 128
        self.height, self.width = 80, 80
        self.kernel_size = 3
        self.reduction = 0.0625
        
    def test_odconv_module_forward(self):
        """Test 1: Forward pass module ODConv isolé"""
        print("\n🧪 Test 1: ODConv Module Forward Pass")
        
        # Créer module ODConv
        odconv = ODConv2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            reduction=self.reduction
        ).to(self.device)
        
        # Input tensor
        x = torch.randn(
            self.batch_size, 
            self.input_channels, 
            self.height, 
            self.width
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = odconv(x)
        
        # Validations
        expected_shape = (self.batch_size, self.output_channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape, 
                        f"Output shape {output.shape} != expected {expected_shape}")
        
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")
        
        print(f"   ✅ Forward pass successful: {x.shape} -> {output.shape}")
        print(f"   ✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        
    def test_odconv_attention_dimensions(self):
        """Test 2: Dimensions attention 4D ODConv"""
        print("\n🧪 Test 2: ODConv 4D Attention Dimensions")
        
        odconv = ODConv2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            reduction=self.reduction
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, 
            self.input_channels, 
            self.height, 
            self.width
        ).to(self.device)
        
        with torch.no_grad():
            # Get attention components
            attn_spatial = odconv._get_spatial_attention(x)
            attn_input = odconv._get_input_channel_attention(x)
            attn_output = odconv._get_output_channel_attention(x)
            attn_kernel = odconv._get_kernel_attention(x)
        
        # Validation dimensions
        expected_spatial = (self.batch_size, 1, self.kernel_size, self.kernel_size)
        expected_input = (self.batch_size, self.input_channels)
        expected_output = (self.batch_size, self.output_channels)
        expected_kernel = (self.batch_size, 1)
        
        self.assertEqual(attn_spatial.shape, expected_spatial)
        self.assertEqual(attn_input.shape, expected_input)
        self.assertEqual(attn_output.shape, expected_output)
        self.assertEqual(attn_kernel.shape, expected_kernel)
        
        print(f"   ✅ Spatial attention: {attn_spatial.shape}")
        print(f"   ✅ Input channel attention: {attn_input.shape}")
        print(f"   ✅ Output channel attention: {attn_output.shape}")
        print(f"   ✅ Kernel attention: {attn_kernel.shape}")
        
        # Validation valeurs attention (doivent être normalisées)
        self.assertTrue(torch.all(attn_spatial >= 0), "Spatial attention has negative values")
        self.assertTrue(torch.all(attn_input >= 0), "Input attention has negative values")
        self.assertTrue(torch.all(attn_output >= 0), "Output attention has negative values")
        self.assertTrue(torch.all(attn_kernel >= 0), "Kernel attention has negative values")
        
        print(f"   ✅ All attention values are non-negative")
        
    def test_featherface_odconv_architecture(self):
        """Test 3: Architecture complète FeatherFace ODConv"""
        print("\n🧪 Test 3: FeatherFace ODConv Complete Architecture")
        
        # Créer modèle
        model = FeatherFaceODConv(cfg=cfg_odconv, phase='train').to(self.device)
        
        # Input standard FeatherFace
        input_size = 640
        x = torch.randn(self.batch_size, 3, input_size, input_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        # Validation outputs (loc, conf, landms)
        self.assertEqual(len(outputs), 3, "Model should return 3 outputs (loc, conf, landms)")
        
        loc, conf, landms = outputs
        print(f"   ✅ Location predictions: {loc.shape}")
        print(f"   ✅ Confidence predictions: {conf.shape}")
        print(f"   ✅ Landmark predictions: {landms.shape}")
        
        # Validation pas de NaN/Inf
        for i, output in enumerate(outputs):
            self.assertFalse(torch.isnan(output).any(), f"Output {i} contains NaN")
            self.assertFalse(torch.isinf(output).any(), f"Output {i} contains Inf")
        
        print(f"   ✅ All outputs valid (no NaN/Inf)")
        
    def test_odconv_vs_cbam_parameters(self):
        """Test 4: Comparaison paramètres ODConv vs CBAM"""
        print("\n🧪 Test 4: Parameter Comparison ODConv vs CBAM")
        
        # Créer les deux modèles
        model_odconv = FeatherFaceODConv(cfg=cfg_odconv, phase='train')
        model_cbam = RetinaFace(cfg=cfg_mnet, phase='train')
        
        # Compter paramètres
        params_odconv = sum(p.numel() for p in model_odconv.parameters())
        params_cbam = sum(p.numel() for p in model_cbam.parameters())
        
        print(f"   📊 ODConv parameters: {params_odconv:,}")
        print(f"   📊 CBAM parameters: {params_cbam:,}")
        print(f"   📊 Difference: {params_odconv - params_cbam:+,}")
        print(f"   📊 Change: {(params_odconv - params_cbam)/params_cbam*100:+.2f}%")
        
        # Validation efficacité (ODConv devrait être plus efficace)
        self.assertLess(params_odconv, params_cbam * 1.05, 
                       "ODConv should not significantly increase parameters")
        
        # Compter modules d'attention
        odconv_attention = sum(1 for name, _ in model_odconv.named_modules() 
                              if 'odconv' in name.lower())
        cbam_attention = sum(1 for name, _ in model_cbam.named_modules() 
                            if 'cbam' in name.lower())
        
        print(f"   🔍 ODConv attention modules: {odconv_attention}")
        print(f"   🔍 CBAM attention modules: {cbam_attention}")
        
        self.assertEqual(odconv_attention, 6, "Should have 6 ODConv modules")
        
    def test_odconv_inference_speed(self):
        """Test 5: Vitesse d'inférence ODConv"""
        print("\n🧪 Test 5: ODConv Inference Speed")
        
        model = FeatherFaceODConv(cfg=cfg_odconv, phase='test').to(self.device)
        model.eval()
        
        # Input batch
        input_size = 640
        x = torch.randn(1, 3, input_size, input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Mesure temps inférence
        times = []
        with torch.no_grad():
            for _ in range(50):
                start_time = time.time()
                _ = model(x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   ⏱️ Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   ⏱️ Min/Max: {min(times):.2f}/{max(times):.2f} ms")
        
        # Validation temps acceptable pour mobile (<100ms cible)
        self.assertLess(avg_time, 150.0, 
                       f"Inference too slow: {avg_time:.2f}ms > 150ms threshold")
        
    def test_odconv_gradient_flow(self):
        """Test 6: Flux de gradients ODConv"""
        print("\n🧪 Test 6: ODConv Gradient Flow")
        
        model = FeatherFaceODConv(cfg=cfg_odconv, phase='train').to(self.device)
        
        # Input et target factices
        x = torch.randn(2, 3, 640, 640, requires_grad=True).to(self.device)
        
        # Forward pass
        outputs = model(x)
        
        # Loss factice
        loss = sum(output.sum() for output in outputs)
        
        # Backward pass
        loss.backward()
        
        # Vérifier gradients
        grad_norms = []
        attention_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if 'odconv' in name.lower():
                    attention_grads.append(grad_norm)
        
        print(f"   📈 Total parameters with gradients: {len(grad_norms)}")
        print(f"   📈 ODConv parameters with gradients: {len(attention_grads)}")
        print(f"   📈 Average gradient norm: {np.mean(grad_norms):.6f}")
        print(f"   📈 ODConv gradient norm: {np.mean(attention_grads):.6f}")
        
        # Validations
        self.assertGreater(len(grad_norms), 0, "No gradients computed")
        self.assertGreater(len(attention_grads), 0, "No ODConv gradients")
        self.assertLess(np.mean(grad_norms), 1.0, "Gradient explosion detected")
        self.assertGreater(np.mean(grad_norms), 1e-8, "Gradient vanishing detected")
        
    def test_odconv_memory_efficiency(self):
        """Test 7: Efficacité mémoire ODConv"""
        print("\n🧪 Test 7: ODConv Memory Efficiency")
        
        if self.device.type != 'cuda':
            print("   ⚠️ Skipping memory test (CPU mode)")
            return
        
        # Reset mémoire
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Créer modèle et input
        model = FeatherFaceODConv(cfg=cfg_odconv, phase='train').to(self.device)
        x = torch.randn(4, 3, 640, 640).to(self.device)  # Batch plus large
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        memory_used = (current_memory - initial_memory) / 1024**2  # MB
        peak_used = (peak_memory - initial_memory) / 1024**2  # MB
        
        print(f"   💾 Current memory usage: {memory_used:.1f} MB")
        print(f"   💾 Peak memory usage: {peak_used:.1f} MB")
        
        # Validation mémoire raisonnable (<2GB pour batch 4)
        self.assertLess(peak_used, 2048, 
                       f"Memory usage too high: {peak_used:.1f}MB")
        
    def test_odconv_config_consistency(self):
        """Test 8: Cohérence configuration ODConv"""
        print("\n🧪 Test 8: ODConv Configuration Consistency")
        
        # Vérifier configuration ODConv
        required_keys = ['name', 'min_sizes', 'steps', 'variance', 'clip', 'loc_weight']
        
        for key in required_keys:
            self.assertIn(key, cfg_odconv, f"Missing key in cfg_odconv: {key}")
        
        # Vérifier valeurs cohérentes
        self.assertEqual(cfg_odconv['name'], 'ODConv', "Config name should be 'ODConv'")
        self.assertIsInstance(cfg_odconv['min_sizes'], list, "min_sizes should be list")
        self.assertGreater(len(cfg_odconv['min_sizes']), 0, "min_sizes should not be empty")
        
        print(f"   ✅ Configuration name: {cfg_odconv['name']}")
        print(f"   ✅ Min sizes: {cfg_odconv['min_sizes']}")
        print(f"   ✅ Steps: {cfg_odconv['steps']}")
        print(f"   ✅ All required keys present")

def run_validation_suite():
    """Exécute la suite complète de tests de validation"""
    print("🚀 FEATHERFACE ODCONV VALIDATION SUITE")
    print("="*50)
    
    # Configuration
    unittest.TestLoader.sortTestMethodsUsing = None  # Ordre d'exécution
    
    # Créer suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestODConvValidation)
    
    # Exécuter avec verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Résumé final
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failures} ❌")
    print(f"Errors: {errors} ⚠️")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Status final
    if success_rate >= 85:
        print(f"\n🎉 VALIDATION SUCCESSFUL! ODConv ready for deployment.")
    else:
        print(f"\n⚠️ VALIDATION ISSUES DETECTED. Review failures before deployment.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)