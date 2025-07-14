"""
Model Validation and Testing Utilities for FeatherFace
Provides comprehensive validation, error checking, and compatibility testing
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import time
from contextlib import contextmanager


class ModelValidator:
    """Comprehensive model validation and testing utilities"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_results = {}
        
    def validate_model_architecture(self, model: nn.Module, 
                                  expected_params: Optional[int] = None,
                                  target_tolerance: int = 5000) -> Dict[str, Any]:
        """Validate model architecture and parameter counts"""
        results = {
            'architecture_valid': True,
            'errors': [],
            'warnings': [],
            'parameter_info': {}
        }
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results['parameter_info'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': total_params - trainable_params,
                'parameter_size_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
            }
            
            # Check against expected parameter count
            if expected_params is not None:
                param_diff = abs(total_params - expected_params)
                if param_diff > target_tolerance:
                    results['errors'].append(
                        f"Parameter count mismatch: expected {expected_params:,}, "
                        f"got {total_params:,} (diff: {param_diff:,})"
                    )
                    results['architecture_valid'] = False
                else:
                    results['warnings'].append(
                        f"Parameter count close to target: {total_params:,} vs {expected_params:,} "
                        f"(diff: {param_diff:,})"
                    )
            
            # Check for common issues
            self._check_gradient_flow(model, results)
            self._check_layer_initialization(model, results)
            
            logging.info(f"Architecture validation: {total_params:,} parameters "
                        f"({'PASSED' if results['architecture_valid'] else 'FAILED'})")
            
        except Exception as e:
            results['errors'].append(f"Architecture validation failed: {str(e)}")
            results['architecture_valid'] = False
            
        return results
    
    def validate_forward_pass(self, model: nn.Module, 
                            input_shapes: List[Tuple[int, ...]],
                            expected_output_shapes: Optional[List[Tuple[int, ...]]] = None) -> Dict[str, Any]:
        """Validate model forward pass with different input sizes"""
        results = {
            'forward_pass_valid': True,
            'errors': [],
            'warnings': [],
            'output_info': {}
        }
        
        model.eval()
        
        for i, input_shape in enumerate(input_shapes):
            try:
                # Create test input
                test_input = torch.randn(input_shape).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(test_input)
                
                # Store output information
                if isinstance(outputs, (list, tuple)):
                    output_shapes = [out.shape for out in outputs]
                    output_dtypes = [str(out.dtype) for out in outputs]
                else:
                    output_shapes = [outputs.shape]
                    output_dtypes = [str(outputs.dtype)]
                
                results['output_info'][f'input_{i}'] = {
                    'input_shape': input_shape,
                    'output_shapes': output_shapes,
                    'output_dtypes': output_dtypes
                }
                
                # Check against expected shapes
                if expected_output_shapes and i < len(expected_output_shapes):
                    expected = expected_output_shapes[i]
                    if isinstance(outputs, (list, tuple)):
                        actual = outputs[0].shape
                    else:
                        actual = outputs.shape
                        
                    if actual != expected:
                        results['warnings'].append(
                            f"Output shape mismatch for input {i}: "
                            f"expected {expected}, got {actual}"
                        )
                
                logging.info(f"Forward pass {i+1}/{len(input_shapes)}: "
                           f"{input_shape} -> {output_shapes}")
                
            except Exception as e:
                error_msg = f"Forward pass failed for input shape {input_shape}: {str(e)}"
                results['errors'].append(error_msg)
                results['forward_pass_valid'] = False
                logging.error(error_msg)
        
        return results
    
    def validate_device_compatibility(self, model: nn.Module) -> Dict[str, Any]:
        """Validate model works on available devices"""
        results = {
            'device_compatibility': True,
            'errors': [],
            'warnings': [],
            'device_info': {}
        }
        
        # Test CPU
        try:
            model_cpu = model.cpu()
            test_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                _ = model_cpu(test_input)
            results['device_info']['cpu'] = 'PASSED'
            logging.info("CPU compatibility: PASSED")
        except Exception as e:
            results['errors'].append(f"CPU compatibility failed: {str(e)}")
            results['device_compatibility'] = False
            results['device_info']['cpu'] = 'FAILED'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                model_cuda = model.cuda()
                test_input = torch.randn(1, 3, 640, 640).cuda()
                with torch.no_grad():
                    _ = model_cuda(test_input)
                results['device_info']['cuda'] = 'PASSED'
                logging.info("CUDA compatibility: PASSED")
            except Exception as e:
                results['errors'].append(f"CUDA compatibility failed: {str(e)}")
                results['device_compatibility'] = False
                results['device_info']['cuda'] = 'FAILED'
        else:
            results['device_info']['cuda'] = 'NOT_AVAILABLE'
        
        return results
    
    def validate_memory_efficiency(self, model: nn.Module, 
                                 batch_sizes: List[int] = [1, 8, 16, 32]) -> Dict[str, Any]:
        """Validate memory usage across different batch sizes"""
        results = {
            'memory_efficient': True,
            'errors': [],
            'warnings': [],
            'memory_info': {}
        }
        
        if not torch.cuda.is_available():
            results['warnings'].append("CUDA not available, skipping memory validation")
            return results
        
        model = model.cuda()
        model.eval()
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Test forward pass
                test_input = torch.randn(batch_size, 3, 640, 640).cuda()
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Get memory statistics
                allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                
                results['memory_info'][f'batch_{batch_size}'] = {
                    'allocated_mb': allocated,
                    'peak_mb': peak,
                    'memory_per_sample_mb': peak / batch_size
                }
                
                logging.info(f"Batch size {batch_size}: {peak:.1f}MB peak memory "
                           f"({peak/batch_size:.1f}MB per sample)")
                
                # Check for excessive memory usage
                if peak > 8000:  # 8GB threshold
                    results['warnings'].append(
                        f"High memory usage for batch size {batch_size}: {peak:.1f}MB"
                    )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results['warnings'].append(f"OOM error at batch size {batch_size}")
                    break
                else:
                    results['errors'].append(f"Memory test failed for batch size {batch_size}: {str(e)}")
                    results['memory_efficient'] = False
        
        return results
    
    def validate_training_compatibility(self, model: nn.Module, 
                                      loss_fn: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Validate model is ready for training"""
        results = {
            'training_ready': True,
            'errors': [],
            'warnings': [],
            'training_info': {}
        }
        
        try:
            # Check if model has trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                results['errors'].append("No trainable parameters found")
                results['training_ready'] = False
            
            # Test gradient computation
            model.train()
            test_input = torch.randn(2, 3, 640, 640).to(self.device)
            
            # Forward pass
            outputs = model(test_input)
            
            # Create dummy loss
            if loss_fn is None:
                if isinstance(outputs, (list, tuple)):
                    dummy_loss = sum(out.sum() for out in outputs)
                else:
                    dummy_loss = outputs.sum()
            else:
                # Use provided loss function with dummy targets
                dummy_targets = torch.randn_like(outputs if not isinstance(outputs, (list, tuple)) else outputs[0])
                dummy_loss = loss_fn(outputs, dummy_targets)
            
            # Backward pass
            dummy_loss.backward()
            
            # Check gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                elif param.requires_grad:
                    results['warnings'].append(f"No gradient for parameter: {name}")
            
            if grad_norms:
                avg_grad_norm = np.mean(grad_norms)
                max_grad_norm = np.max(grad_norms)
                
                results['training_info'] = {
                    'trainable_parameters': trainable_params,
                    'avg_gradient_norm': avg_grad_norm,
                    'max_gradient_norm': max_grad_norm,
                    'gradient_flow_ok': max_grad_norm > 1e-7 and max_grad_norm < 1e3
                }
                
                # Check for gradient issues
                if max_grad_norm < 1e-7:
                    results['warnings'].append("Very small gradients detected (vanishing gradients?)")
                elif max_grad_norm > 1e3:
                    results['warnings'].append("Very large gradients detected (exploding gradients?)")
                
                logging.info(f"Training validation: {trainable_params:,} trainable params, "
                           f"grad norm: {avg_grad_norm:.2e}")
            else:
                results['errors'].append("No gradients computed")
                results['training_ready'] = False
            
        except Exception as e:
            results['errors'].append(f"Training compatibility test failed: {str(e)}")
            results['training_ready'] = False
        
        return results
    
    def _check_gradient_flow(self, model: nn.Module, results: Dict):
        """Check for potential gradient flow issues"""
        for name, module in model.named_modules():
            # Check for modules that might block gradients
            if isinstance(module, nn.ReLU) and not module.inplace:
                results['warnings'].append(f"Non-inplace ReLU found: {name} (consider inplace=True)")
            
            # Check for very deep sequences
            if isinstance(module, nn.Sequential) and len(module) > 20:
                results['warnings'].append(f"Very deep sequential module: {name} ({len(module)} layers)")
    
    def _check_layer_initialization(self, model: nn.Module, results: Dict):
        """Check layer initialization"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_std = param.std().item()
                param_mean = param.mean().item()
                
                # Check for potential initialization issues
                if param_std < 1e-6:
                    results['warnings'].append(f"Very small parameter variance: {name} (std: {param_std:.2e})")
                elif param_std > 1.0:
                    results['warnings'].append(f"Large parameter variance: {name} (std: {param_std:.2f})")
                
                if abs(param_mean) > 0.5:
                    results['warnings'].append(f"Large parameter mean: {name} (mean: {param_mean:.2f})")
    
    def run_comprehensive_validation(self, model: nn.Module, 
                                   expected_params: Optional[int] = None,
                                   save_results: bool = True) -> Dict[str, Any]:
        """Run all validation tests"""
        logging.info("🔍 Starting comprehensive model validation...")
        
        validation_results = {
            'timestamp': time.time(),
            'model_class': model.__class__.__name__,
            'validation_passed': True
        }
        
        # Architecture validation
        arch_results = self.validate_model_architecture(model, expected_params)
        validation_results['architecture'] = arch_results
        if not arch_results['architecture_valid']:
            validation_results['validation_passed'] = False
        
        # Forward pass validation
        input_shapes = [(1, 3, 640, 640), (1, 3, 416, 416), (2, 3, 640, 640)]
        forward_results = self.validate_forward_pass(model, input_shapes)
        validation_results['forward_pass'] = forward_results
        if not forward_results['forward_pass_valid']:
            validation_results['validation_passed'] = False
        
        # Device compatibility
        device_results = self.validate_device_compatibility(model)
        validation_results['device_compatibility'] = device_results
        if not device_results['device_compatibility']:
            validation_results['validation_passed'] = False
        
        # Memory efficiency
        memory_results = self.validate_memory_efficiency(model)
        validation_results['memory_efficiency'] = memory_results
        if not memory_results['memory_efficient']:
            validation_results['validation_passed'] = False
        
        # Training compatibility
        training_results = self.validate_training_compatibility(model)
        validation_results['training_compatibility'] = training_results
        if not training_results['training_ready']:
            validation_results['validation_passed'] = False
        
        # Save results
        if save_results:
            self._save_validation_results(validation_results)
        
        # Print summary
        self._print_validation_summary(validation_results)
        
        return validation_results
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        save_dir = Path("experiments/logs")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = results['timestamp']
        filename = f"validation_results_{int(timestamp)}.json"
        save_path = save_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Validation results saved to: {save_path}")
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print a summary of validation results"""
        print("\n" + "="*60)
        print("MODEL VALIDATION SUMMARY")
        print("="*60)
        
        overall_status = "✅ PASSED" if results['validation_passed'] else "❌ FAILED"
        print(f"Overall Status: {overall_status}")
        print(f"Model: {results['model_class']}")
        
        # Architecture summary
        arch = results['architecture']
        params = arch['parameter_info']['total_parameters']
        print(f"\n📊 Architecture:")
        print(f"  Parameters: {params:,} ({params/1e6:.3f}M)")
        print(f"  Status: {'✅' if arch['architecture_valid'] else '❌'}")
        
        # Device compatibility
        device = results['device_compatibility']
        print(f"\n💻 Device Compatibility:")
        for dev, status in device['device_info'].items():
            print(f"  {dev.upper()}: {'✅' if status == 'PASSED' else '❌' if status == 'FAILED' else '⚠️'}")
        
        # Memory efficiency
        memory = results['memory_efficiency']
        if 'memory_info' in memory:
            print(f"\n🧠 Memory Usage:")
            for batch_info in memory['memory_info'].values():
                batch_size = batch_info.get('memory_per_sample_mb', 0) * 8  # Estimate batch size
                peak_mb = batch_info.get('peak_mb', 0)
                print(f"  Batch size ~{int(batch_size)}: {peak_mb:.1f}MB")
        
        # Errors and warnings
        all_errors = []
        all_warnings = []
        
        for section in ['architecture', 'forward_pass', 'device_compatibility', 
                       'memory_efficiency', 'training_compatibility']:
            if section in results:
                all_errors.extend(results[section].get('errors', []))
                all_warnings.extend(results[section].get('warnings', []))
        
        if all_errors:
            print(f"\n❌ Errors ({len(all_errors)}):")
            for error in all_errors:
                print(f"  - {error}")
        
        if all_warnings:
            print(f"\n⚠️ Warnings ({len(all_warnings)}):")
            for warning in all_warnings:
                print(f"  - {warning}")
        
        print("\n" + "="*60)


def quick_model_validation(model: nn.Module, 
                         expected_params: Optional[int] = None,
                         device: Optional[torch.device] = None) -> bool:
    """Quick validation for notebook use"""
    validator = ModelValidator(device)
    results = validator.run_comprehensive_validation(model, expected_params)
    return results['validation_passed']


def validate_model_compatibility(model1: nn.Module, model2: nn.Module,
                                test_inputs: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
    """Validate compatibility between two models (e.g., CBAM baseline and ECA innovation)"""
    results = {
        'compatible': True,
        'errors': [],
        'warnings': [],
        'comparison': {}
    }
    
    # Parameter comparison
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    results['comparison']['parameters'] = {
        'model1': params1,
        'model2': params2,
        'ratio': params1 / params2 if params2 > 0 else float('inf')
    }
    
    # Output shape compatibility
    if test_inputs is None:
        test_inputs = [torch.randn(1, 3, 640, 640)]
    
    model1.eval()
    model2.eval()
    
    for i, test_input in enumerate(test_inputs):
        try:
            with torch.no_grad():
                out1 = model1(test_input)
                out2 = model2(test_input)
            
            # Compare output shapes
            if isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
                if len(out1) != len(out2):
                    results['errors'].append(f"Different number of outputs: {len(out1)} vs {len(out2)}")
                    results['compatible'] = False
                else:
                    for j, (o1, o2) in enumerate(zip(out1, out2)):
                        if o1.shape != o2.shape:
                            results['warnings'].append(f"Output {j} shape mismatch: {o1.shape} vs {o2.shape}")
            else:
                if isinstance(out1, (list, tuple)) != isinstance(out2, (list, tuple)):
                    results['errors'].append("Inconsistent output types (tensor vs list/tuple)")
                    results['compatible'] = False
                elif hasattr(out1, 'shape') and hasattr(out2, 'shape') and out1.shape != out2.shape:
                    results['warnings'].append(f"Output shape mismatch: {out1.shape} vs {out2.shape}")
        
        except Exception as e:
            results['errors'].append(f"Compatibility test failed for input {i}: {str(e)}")
            results['compatible'] = False
    
    logging.info(f"Model compatibility: {'✅ COMPATIBLE' if results['compatible'] else '❌ INCOMPATIBLE'}")
    
    return results


if __name__ == "__main__":
    # Demo usage
    import torch.nn as nn
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Run validation
    model = TestModel()
    is_valid = quick_model_validation(model, expected_params=None)
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")