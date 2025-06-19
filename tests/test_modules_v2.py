"""
Unit tests for FeatherFace V2 optimized modules
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modules_v2 import CBAM_Plus, SharedMultiHead, SharedCBAMManager, count_parameters
from models.net import CBAM


class TestModulesV2(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.height = 20
        self.width = 20
        
    def test_cbam_plus_parameter_reduction(self):
        """Test that CBAM_Plus reduces parameters compared to original CBAM"""
        channels_list = [32, 64, 128, 256]
        
        for channels in channels_list:
            with self.subTest(channels=channels):
                # Original CBAM
                original = CBAM(channels, reduction_ratio=16)
                original_params = count_parameters(original)
                
                # CBAM_Plus
                optimized = CBAM_Plus(channels, reduction_ratio=32)
                optimized_params = count_parameters(optimized)
                
                # Should have fewer parameters
                self.assertLess(optimized_params, original_params,
                               f"CBAM_Plus should have fewer params than original for channels={channels}")
                
                # Should achieve at least 40% reduction
                reduction_rate = (1 - optimized_params / original_params)
                self.assertGreaterEqual(reduction_rate, 0.40,
                                      f"CBAM_Plus should achieve at least 40% reduction for channels={channels}")
    
    def test_cbam_plus_forward_pass(self):
        """Test CBAM_Plus forward pass maintains tensor shape"""
        channels = 64
        cbam = CBAM_Plus(channels).to(self.device)
        
        x = torch.randn(self.batch_size, channels, self.height, self.width).to(self.device)
        output = cbam(x)
        
        # Output shape should match input
        self.assertEqual(output.shape, x.shape,
                        "CBAM_Plus should maintain input tensor shape")
        
        # Output should be different from input (attention applied)
        self.assertFalse(torch.allclose(output, x),
                        "CBAM_Plus should modify the input tensor")
    
    def test_shared_multihead_output_shapes(self):
        """Test SharedMultiHead produces correct output shapes"""
        in_channels = 64
        num_anchors = 3
        
        head = SharedMultiHead(in_channels, num_anchors).to(self.device)
        x = torch.randn(self.batch_size, in_channels, self.height, self.width).to(self.device)
        
        cls, bbox, ldm = head(x)
        
        # Check output shapes
        expected_num_predictions = self.height * self.width * num_anchors
        
        self.assertEqual(cls.shape, (self.batch_size, expected_num_predictions, 2),
                        "Classification output shape incorrect")
        self.assertEqual(bbox.shape, (self.batch_size, expected_num_predictions, 4),
                        "BBox output shape incorrect")
        self.assertEqual(ldm.shape, (self.batch_size, expected_num_predictions, 10),
                        "Landmark output shape incorrect")
    
    def test_shared_cbam_manager(self):
        """Test SharedCBAMManager functionality"""
        channel_configs = {
            'small': 32,
            'medium': 64,
            'large': 128
        }
        
        manager = SharedCBAMManager(channel_configs).to(self.device)
        
        # Test different channel sizes
        for name, channels in channel_configs.items():
            with self.subTest(config=name):
                x = torch.randn(self.batch_size, channels, self.height, self.width).to(self.device)
                output = manager(x, name)
                
                # Output shape should match input
                self.assertEqual(output.shape, x.shape,
                               f"SharedCBAMManager should maintain shape for {name}")
                
                # Output should be different (attention applied)
                self.assertFalse(torch.allclose(output, x),
                               f"SharedCBAMManager should modify input for {name}")
    
    def test_shared_cbam_manager_parameter_efficiency(self):
        """Test that SharedCBAMManager is more efficient than separate CBAMs"""
        # Simulate 6 CBAM instances
        channel_sizes = [32, 64, 128, 64, 64, 64]
        
        # Original: separate CBAMs
        original_params = 0
        for channels in channel_sizes:
            cbam = CBAM(channels, reduction_ratio=16)
            original_params += count_parameters(cbam)
        
        # Optimized: shared manager
        channel_configs = {
            'ch32': 32,
            'ch64': 64,
            'ch128': 128
        }
        manager = SharedCBAMManager(channel_configs, reduction_ratio=32)
        manager_params = count_parameters(manager)
        
        # Should use fewer parameters
        self.assertLess(manager_params, original_params,
                       "SharedCBAMManager should use fewer params than separate CBAMs")
        
        # Should achieve at least 50% reduction
        reduction_rate = (1 - manager_params / original_params)
        self.assertGreaterEqual(reduction_rate, 0.50,
                              "SharedCBAMManager should achieve at least 50% reduction")
    
    def test_backward_compatibility(self):
        """Test that modules support gradient computation"""
        # CBAM_Plus
        cbam = CBAM_Plus(64).to(self.device)
        x = torch.randn(1, 64, 10, 10, requires_grad=True).to(self.device)
        output = cbam(x)
        loss = output.mean()
        loss.backward()
        
        self.assertIsNotNone(x.grad, "CBAM_Plus should support backpropagation")
        
        # SharedMultiHead
        head = SharedMultiHead(64, 3).to(self.device)
        x = torch.randn(1, 64, 10, 10, requires_grad=True).to(self.device)
        cls, bbox, ldm = head(x)
        loss = cls.mean() + bbox.mean() + ldm.mean()
        
        # Clear previous gradients
        if x.grad is not None:
            x.grad.zero_()
            
        loss.backward()
        self.assertIsNotNone(x.grad, "SharedMultiHead should support backpropagation")


if __name__ == '__main__':
    unittest.main(verbosity=2)
