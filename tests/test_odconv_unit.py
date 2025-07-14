#!/usr/bin/env python3
"""
Tests unitaires pour les modules ODConv
Validation individuelle des composants ODConv

Auteur: FeatherFace ODConv Team  
Date: Juillet 2025
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.odconv import ODConv2d, AttentionGeneration

class TestODConvUnit(unittest.TestCase):
    """Tests unitaires modules ODConv individuels"""
    
    def setUp(self):
        """Configuration tests"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_attention_generation_module(self):
        """Test module génération attention"""
        print("\n🧪 Testing AttentionGeneration module")
        
        in_channels = 64
        attention_dim = 4  # 4 dimensions attention
        reduction = 0.0625
        
        attention_gen = AttentionGeneration(
            in_channels=in_channels,
            attention_dim=attention_dim,
            reduction=reduction
        ).to(self.device)
        
        # Test input
        batch_size = 2
        height, width = 32, 32
        x = torch.randn(batch_size, in_channels, height, width).to(self.device)
        
        # Forward pass
        attention = attention_gen(x)
        
        # Validations
        expected_shape = (batch_size, attention_dim)
        self.assertEqual(attention.shape, expected_shape)
        self.assertFalse(torch.isnan(attention).any())
        self.assertTrue(torch.all(attention >= 0))  # Attention positive
        
        print(f"   ✅ Input: {x.shape} -> Attention: {attention.shape}")
        
    def test_odconv_basic_functionality(self):
        """Test fonctionnalité de base ODConv"""
        print("\n🧪 Testing ODConv basic functionality")
        
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        reduction = 0.0625
        
        odconv = ODConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            reduction=reduction
        ).to(self.device)
        
        # Test différentes tailles input
        test_sizes = [(16, 16), (32, 32), (64, 64)]
        
        for h, w in test_sizes:
            with self.subTest(size=(h, w)):
                x = torch.randn(1, in_channels, h, w).to(self.device)
                
                with torch.no_grad():
                    output = odconv(x)
                
                expected_shape = (1, out_channels, h, w)
                self.assertEqual(output.shape, expected_shape)
                self.assertFalse(torch.isnan(output).any())
                
        print(f"   ✅ Tested sizes: {test_sizes}")
        
    def test_odconv_attention_components(self):
        """Test composants attention individuels"""
        print("\n🧪 Testing ODConv attention components")
        
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        
        odconv = ODConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        ).to(self.device)
        
        batch_size = 2
        height, width = 40, 40
        x = torch.randn(batch_size, in_channels, height, width).to(self.device)
        
        with torch.no_grad():
            # Test chaque composant attention
            spatial_attn = odconv._get_spatial_attention(x)
            input_attn = odconv._get_input_channel_attention(x)
            output_attn = odconv._get_output_channel_attention(x)
            kernel_attn = odconv._get_kernel_attention(x)
            
            # Validations dimensions
            self.assertEqual(spatial_attn.shape, (batch_size, 1, kernel_size, kernel_size))
            self.assertEqual(input_attn.shape, (batch_size, in_channels))
            self.assertEqual(output_attn.shape, (batch_size, out_channels))
            self.assertEqual(kernel_attn.shape, (batch_size, 1))
            
            # Validations valeurs
            self.assertTrue(torch.all(spatial_attn >= 0))
            self.assertTrue(torch.all(input_attn >= 0))
            self.assertTrue(torch.all(output_attn >= 0))
            self.assertTrue(torch.all(kernel_attn >= 0))
            
        print(f"   ✅ All 4D attention components validated")
        
    def test_odconv_parameter_efficiency(self):
        """Test efficacité paramétrique ODConv"""
        print("\n🧪 Testing ODConv parameter efficiency")
        
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        
        # ODConv avec différents reduction ratios
        reductions = [0.0625, 0.125, 0.25]
        
        base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        base_params = sum(p.numel() for p in base_conv.parameters())
        
        for reduction in reductions:
            with self.subTest(reduction=reduction):
                odconv = ODConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    reduction=reduction
                )
                
                odconv_params = sum(p.numel() for p in odconv.parameters())
                overhead = (odconv_params - base_params) / base_params * 100
                
                print(f"   📊 Reduction {reduction}: +{overhead:.2f}% params")
                
                # Validation overhead raisonnable
                self.assertLess(overhead, 50.0, f"Too much overhead: {overhead:.1f}%")
                
    def test_odconv_gradient_computation(self):
        """Test calcul gradients ODConv"""
        print("\n🧪 Testing ODConv gradient computation")
        
        odconv = ODConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        ).to(self.device)
        
        x = torch.randn(2, 32, 32, 32, requires_grad=True).to(self.device)
        
        # Forward pass
        output = odconv(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier gradients
        for name, param in odconv.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient for {name}")
            
        print(f"   ✅ All parameters have valid gradients")
        
    def test_odconv_numerical_stability(self):
        """Test stabilité numérique ODConv"""
        print("\n🧪 Testing ODConv numerical stability")
        
        odconv = ODConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        ).to(self.device)
        
        # Test avec différentes magnitudes input
        magnitudes = [1e-3, 1e-1, 1.0, 10.0, 100.0]
        
        for magnitude in magnitudes:
            with self.subTest(magnitude=magnitude):
                x = torch.randn(1, 64, 32, 32).to(self.device) * magnitude
                
                with torch.no_grad():
                    output = odconv(x)
                
                self.assertFalse(torch.isnan(output).any(), 
                               f"NaN output for magnitude {magnitude}")
                self.assertFalse(torch.isinf(output).any(), 
                               f"Inf output for magnitude {magnitude}")
                
        print(f"   ✅ Stable across magnitudes: {magnitudes}")
        
    def test_odconv_deterministic(self):
        """Test comportement déterministe ODConv"""
        print("\n🧪 Testing ODConv deterministic behavior")
        
        # Fixer seed pour reproductibilité
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        odconv = ODConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        ).to(self.device)
        
        x = torch.randn(1, 32, 32, 32).to(self.device)
        
        # Deux forward passes identiques
        with torch.no_grad():
            output1 = odconv(x)
            output2 = odconv(x)
        
        # Vérifier identité
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6),
                       "ODConv not deterministic")
        
        print(f"   ✅ Deterministic behavior confirmed")
        
    def test_odconv_memory_efficiency(self):
        """Test efficacité mémoire ODConv"""
        print("\n🧪 Testing ODConv memory efficiency")
        
        if self.device.type != 'cuda':
            print("   ⚠️ Skipping memory test (CPU mode)")
            return
        
        # Reset mémoire
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Créer ODConv
        odconv = ODConv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        ).to(self.device)
        
        x = torch.randn(4, 128, 64, 64).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = odconv(x)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB
        
        print(f"   💾 Memory used: {memory_used:.1f} MB")
        
        # Validation mémoire raisonnable
        self.assertLess(memory_used, 500, f"Memory usage too high: {memory_used:.1f}MB")

def run_unit_tests():
    """Exécute les tests unitaires ODConv"""
    print("🔬 ODCONV UNIT TESTS")
    print("="*30)
    
    # Créer suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestODConvUnit)
    
    # Exécuter
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)