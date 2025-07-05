#!/usr/bin/env python3
"""
FeatherFace Nano-B: Bayesian-Optimized Soft FPGM Pruning
Implementation inspired by B-FPGM (Kaparinos & Mezaris, WACVW 2025)

This module implements Bayesian-optimized structured pruning for FeatherFace Nano,
combining Filter Pruning via Geometric Median (FPGM), Soft Filter Pruning (SFP),
and Bayesian optimization for automatic pruning rate determination.

Scientific Foundation:
1. B-FPGM: Kaparinos & Mezaris, WACVW 2025
2. FPGM: He et al., ICML 2019 - Filter Pruning via Geometric Median
3. SFP: He et al., IJCAI 2018 - Soft Filter Pruning
4. Bayesian Optimization: Mockus, 1989 - Global optimization theory
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometricMedianPruner:
    """
    Filter Pruning via Geometric Median (FPGM) implementation
    
    Based on:
    He et al. "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration"
    """
    
    def __init__(self, distance_type: str = 'l2'):
        """
        Initialize FPGM pruner
        
        Args:
            distance_type: Distance metric for geometric median ('l2' or 'cosine')
        """
        self.distance_type = distance_type
    
    def compute_geometric_median(self, weight_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric median of filters
        
        Args:
            weight_tensor: Convolutional layer weights [out_channels, in_channels, H, W]
            
        Returns:
            Geometric median tensor
        """
        # Flatten filters to vectors
        filters = weight_tensor.view(weight_tensor.size(0), -1)  # [out_channels, flattened]
        
        # Compute pairwise distances
        if self.distance_type == 'l2':
            distances = torch.cdist(filters, filters, p=2)
        elif self.distance_type == 'cosine':
            # Cosine distance = 1 - cosine similarity
            normalized = torch.nn.functional.normalize(filters, p=2, dim=1)
            cosine_sim = torch.mm(normalized, normalized.t())
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        # Find filter with minimum sum of distances (geometric median approximation)
        sum_distances = distances.sum(dim=1)
        median_idx = sum_distances.argmin()
        
        return filters[median_idx]
    
    def rank_filters_by_importance(self, weight_tensor: torch.Tensor) -> torch.Tensor:
        """
        Rank filters by their distance to geometric median
        
        Args:
            weight_tensor: Convolutional layer weights [out_channels, in_channels, H, W]
            
        Returns:
            Indices of filters sorted by importance (ascending = least important first)
        """
        # Compute geometric median
        geometric_median = self.compute_geometric_median(weight_tensor)
        
        # Flatten filters
        filters = weight_tensor.view(weight_tensor.size(0), -1)
        
        # Compute distances to geometric median
        if self.distance_type == 'l2':
            distances = torch.norm(filters - geometric_median.unsqueeze(0), p=2, dim=1)
        elif self.distance_type == 'cosine':
            # Cosine distance
            normalized_filters = torch.nn.functional.normalize(filters, p=2, dim=1)
            normalized_median = torch.nn.functional.normalize(geometric_median, p=2, dim=0)
            cosine_sim = torch.mm(normalized_filters, normalized_median.unsqueeze(1)).squeeze()
            distances = 1 - cosine_sim
        
        # Sort by distance (ascending = least important first)
        importance_ranking = distances.argsort()
        
        return importance_ranking


class SoftFilterPruner:
    """
    Soft Filter Pruning implementation
    
    Based on:
    He et al. "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks"
    """
    
    def __init__(self, sparsity_schedule: str = 'polynomial'):
        """
        Initialize SFP pruner
        
        Args:
            sparsity_schedule: Sparsity scheduling strategy ('polynomial', 'exponential')
        """
        self.sparsity_schedule = sparsity_schedule
        self.pruned_filters = {}  # Store pruned filter masks
    
    def create_soft_mask(self, layer_name: str, target_sparsity: float, 
                        current_epoch: int, total_epochs: int) -> torch.Tensor:
        """
        Create soft pruning mask with gradual sparsity increase
        
        Args:
            layer_name: Name of the layer being pruned
            target_sparsity: Final target sparsity (0-1)
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            
        Returns:
            Soft pruning mask
        """
        # Calculate current sparsity based on schedule
        progress = current_epoch / total_epochs
        
        if self.sparsity_schedule == 'polynomial':
            # Polynomial schedule: s_t = s_f * (1 - (1-t)^3)
            current_sparsity = target_sparsity * (1 - (1 - progress) ** 3)
        elif self.sparsity_schedule == 'exponential':
            # Exponential schedule
            current_sparsity = target_sparsity * (1 - np.exp(-5 * progress))
        else:
            # Linear schedule
            current_sparsity = target_sparsity * progress
        
        return current_sparsity
    
    def apply_soft_pruning(self, weight_tensor: torch.Tensor, importance_ranking: torch.Tensor,
                          sparsity: float, temperature: float = 1.0) -> torch.Tensor:
        """
        Apply soft pruning to weights
        
        Args:
            weight_tensor: Layer weights to prune
            importance_ranking: Filter importance ranking (from FPGM)
            sparsity: Current sparsity level (0-1)
            temperature: Softmax temperature for soft pruning
            
        Returns:
            Soft-pruned weights
        """
        num_filters = weight_tensor.size(0)
        num_pruned = int(num_filters * sparsity)
        
        if num_pruned == 0:
            return weight_tensor
        
        # Create soft mask
        soft_mask = torch.ones(num_filters, device=weight_tensor.device)
        
        # Get filters to prune (least important)
        filters_to_prune = importance_ranking[:num_pruned]
        
        # Apply soft pruning with temperature
        for filter_idx in filters_to_prune:
            # Soft mask using sigmoid with temperature
            pruning_strength = 1.0 / (1.0 + torch.exp(temperature))
            soft_mask[filter_idx] = pruning_strength
        
        # Apply mask to weights
        soft_mask = soft_mask.view(-1, 1, 1, 1)  # Broadcast to weight shape
        pruned_weights = weight_tensor * soft_mask
        
        return pruned_weights


class BayesianOptimizer:
    """
    Bayesian optimization for automatic pruning rate determination
    
    Uses Gaussian Process regression to optimize pruning rates across layer groups.
    
    POURQUOI L'OPTIMISATION BAYÉSIENNE ?
    ===================================
    Au lieu de fixer manuellement les taux de pruning (qui donnerait un nombre 
    de paramètres fixe mais suboptimal), cette approche:
    
    1. Teste automatiquement différentes configurations
    2. Évalue les performances de chaque configuration  
    3. Utilise un modèle gaussien pour prédire les meilleures zones
    4. Converge vers la configuration optimale dans la plage cible
    
    RÉSULTAT: Nombre de paramètres variable (120K-180K) mais performances optimales
    vs nombre fixe avec performances dégradées.
    """
    
    def __init__(self, num_groups: int = 6, acquisition_function: str = 'ei'):
        """
        Initialize Bayesian optimizer
        
        Args:
            num_groups: Number of layer groups for optimization
            acquisition_function: Acquisition function ('ei', 'ucb', 'pi')
        """
        self.num_groups = num_groups
        self.acquisition_function = acquisition_function
        
        # Gaussian Process for modeling objective function
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                          normalize_y=True, n_restarts_optimizer=5)
        
        # History of evaluated configurations
        self.X_evaluated = []  # Pruning rate configurations
        self.y_evaluated = []  # Corresponding performance scores
        
        logger.info(f"Initialized Bayesian optimizer with {num_groups} groups")
    
    def suggest_pruning_rates(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Suggest next pruning rates to evaluate using acquisition function
        
        Args:
            bounds: List of (min, max) bounds for each group's pruning rate
            
        Returns:
            Suggested pruning rates for each group
        """
        if len(self.X_evaluated) < 5:  # Initial random exploration
            rates = []
            for low, high in bounds:
                rates.append(np.random.uniform(low, high))
            return np.array(rates)
        
        # Fit GP to current data
        X = np.array(self.X_evaluated)
        y = np.array(self.y_evaluated)
        self.gp.fit(X, y)
        
        # Optimize acquisition function
        def acquisition(x):
            x_reshaped = x.reshape(1, -1)
            
            if self.acquisition_function == 'ei':
                # Expected Improvement
                mu, sigma = self.gp.predict(x_reshaped, return_std=True)
                mu, sigma = mu[0], sigma[0]
                
                if sigma == 0:
                    return 0
                
                best_y = max(self.y_evaluated)
                z = (mu - best_y) / sigma
                ei = (mu - best_y) * self._normal_cdf(z) + sigma * self._normal_pdf(z)
                return -ei  # Minimize negative EI
                
            elif self.acquisition_function == 'ucb':
                # Upper Confidence Bound
                mu, sigma = self.gp.predict(x_reshaped, return_std=True)
                beta = 2.0  # Exploration parameter
                ucb = mu[0] + beta * sigma[0]
                return -ucb  # Minimize negative UCB
            
            else:  # Probability of Improvement
                mu, sigma = self.gp.predict(x_reshaped, return_std=True)
                mu, sigma = mu[0], sigma[0]
                
                if sigma == 0:
                    return 0
                
                best_y = max(self.y_evaluated)
                z = (mu - best_y) / sigma
                pi = self._normal_cdf(z)
                return -pi  # Minimize negative PI
        
        # Optimize acquisition function
        initial_guess = np.array([np.mean(bound) for bound in bounds])
        
        result = minimize(acquisition, initial_guess, bounds=bounds, 
                         method='L-BFGS-B')
        
        return result.x
    
    def update(self, pruning_rates: np.ndarray, performance_score: float):
        """
        Update optimizer with new evaluation
        
        Args:
            pruning_rates: Evaluated pruning rates
            performance_score: Corresponding performance (higher is better)
        """
        self.X_evaluated.append(pruning_rates.copy())
        self.y_evaluated.append(performance_score)
        
        logger.info(f"Updated BO with rates {pruning_rates} -> score {performance_score:.4f}")
    
    def _normal_cdf(self, x):
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class FeatherFaceNanoBPruner:
    """
    Main B-FPGM pruning class for FeatherFace Nano
    
    Integrates FPGM + SFP + Bayesian Optimization for automatic structured pruning
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize FeatherFace Nano-B pruner
        
        Args:
            model: FeatherFace Nano model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        
        # Initialize components
        self.fpgm_pruner = GeometricMedianPruner(
            distance_type=config.get('distance_type', 'l2')
        )
        self.sfp_pruner = SoftFilterPruner(
            sparsity_schedule=config.get('sparsity_schedule', 'polynomial')
        )
        self.bayesian_optimizer = BayesianOptimizer(
            num_groups=config.get('num_groups', 6),
            acquisition_function=config.get('acquisition_function', 'ei')
        )
        
        # Layer grouping for Bayesian optimization
        self.layer_groups = self._create_layer_groups()
        
        # Pruning state
        self.current_sparsities = {name: 0.0 for name in self.layer_groups.keys()}
        self.importance_rankings = {}
        
        logger.info("Initialized FeatherFace Nano-B pruner")
        logger.info(f"Layer groups: {list(self.layer_groups.keys())}")
    
    def _create_layer_groups(self) -> Dict[str, List[str]]:
        """
        Create layer groups for Bayesian optimization
        
        Groups layers based on FeatherFace Nano architecture:
        1. Backbone early layers
        2. Backbone late layers  
        3. CBAM layers (Woo et al. ECCV 2018)
        4. BiFPN layers (Tan et al. CVPR 2020)
        5. Grouped SSH layers
        6. Detection heads
        """
        groups = {
            'backbone_early': [],
            'backbone_late': [],
            'efficient_cbam': [],
            'efficient_bifpn': [],
            'grouped_ssh': [],
            'detection_heads': []
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'body' in name and ('conv1' in name or 'conv2' in name):
                    if 'stage1' in name or 'stage2' in name:
                        groups['backbone_early'].append(name)
                    else:
                        groups['backbone_late'].append(name)
                elif 'cbam' in name.lower():
                    groups['efficient_cbam'].append(name)
                elif 'bifpn' in name.lower() or 'fpn' in name.lower():
                    groups['efficient_bifpn'].append(name)
                elif 'ssh' in name.lower():
                    groups['grouped_ssh'].append(name)
                elif any(head in name for head in ['ClassHead', 'BboxHead', 'LandmarkHead']):
                    groups['detection_heads'].append(name)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def compute_layer_importances(self):
        """Compute FPGM importance rankings for all prunable layers"""
        logger.info("Computing FPGM importance rankings...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.size(0) > 1:
                importance_ranking = self.fpgm_pruner.rank_filters_by_importance(
                    module.weight.data
                )
                self.importance_rankings[name] = importance_ranking
                
                logger.debug(f"Computed importance for {name}: {len(importance_ranking)} filters")
    
    def evaluate_pruning_configuration(self, group_sparsities: Dict[str, float],
                                     validation_loader, criterion) -> float:
        """
        Evaluate a pruning configuration
        
        Args:
            group_sparsities: Sparsity levels for each group
            validation_loader: Validation data loader
            criterion: Loss criterion
            
        Returns:
            Performance score (higher is better)
        """
        # Apply temporary pruning
        original_weights = {}
        
        for group_name, sparsity in group_sparsities.items():
            for layer_name in self.layer_groups[group_name]:
                module = dict(self.model.named_modules())[layer_name]
                if isinstance(module, nn.Conv2d):
                    # Store original weights
                    original_weights[layer_name] = module.weight.data.clone()
                    
                    # Apply pruning
                    if layer_name in self.importance_rankings:
                        pruned_weights = self.sfp_pruner.apply_soft_pruning(
                            module.weight.data,
                            self.importance_rankings[layer_name],
                            sparsity,
                            temperature=self.config.get('soft_temperature', 1.0)
                        )
                        module.weight.data = pruned_weights
        
        # Evaluate performance
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(validation_loader):
                if batch_idx >= self.config.get('eval_batches', 50):  # Limit for speed
                    break
                
                data = data.cuda() if torch.cuda.is_available() else data
                outputs = self.model(data)
                
                # Compute loss (simplified for efficiency)
                if isinstance(outputs, (list, tuple)):
                    # Multi-output case (classification, bbox, landmarks)
                    loss = sum(criterion(out, tgt) for out, tgt in zip(outputs, targets))
                else:
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Restore original weights
        for layer_name, original_weight in original_weights.items():
            module = dict(self.model.named_modules())[layer_name]
            module.weight.data = original_weight
        
        # Return negative loss as performance score (higher is better)
        avg_loss = total_loss / max(num_batches, 1)
        performance_score = -avg_loss
        
        logger.info(f"Evaluated config {group_sparsities} -> score: {performance_score:.4f}")
        
        return performance_score
    
    def optimize_pruning_rates(self, validation_loader, criterion, 
                              num_iterations: int = 20) -> Dict[str, float]:
        """
        Use Bayesian optimization to find optimal pruning rates
        
        Args:
            validation_loader: Validation data loader
            criterion: Loss criterion
            num_iterations: Number of BO iterations
            
        Returns:
            Optimal pruning rates for each group
        """
        logger.info(f"Starting Bayesian optimization for {num_iterations} iterations...")
        
        # Define search bounds for each group
        group_names = list(self.layer_groups.keys())
        bounds = []
        
        for group_name in group_names:
            if 'detection_heads' in group_name:
                # More conservative pruning for detection heads
                bounds.append((0.0, 0.3))
            elif 'backbone_early' in group_name:
                # Conservative pruning for early layers
                bounds.append((0.0, 0.4))
            else:
                # More aggressive pruning for other layers
                bounds.append((0.1, 0.6))
        
        # Bayesian optimization loop
        best_config = None
        best_score = float('-inf')
        
        for iteration in range(num_iterations):
            logger.info(f"BO iteration {iteration + 1}/{num_iterations}")
            
            # Get suggested pruning rates
            suggested_rates = self.bayesian_optimizer.suggest_pruning_rates(bounds)
            
            # Convert to group configuration
            group_config = {
                group_name: rate 
                for group_name, rate in zip(group_names, suggested_rates)
            }
            
            # Evaluate configuration
            score = self.evaluate_pruning_configuration(
                group_config, validation_loader, criterion
            )
            
            # Update Bayesian optimizer
            self.bayesian_optimizer.update(suggested_rates, score)
            
            # Track best configuration
            if score > best_score:
                best_score = score
                best_config = group_config.copy()
                logger.info(f"New best config: {best_config} (score: {best_score:.4f})")
        
        logger.info(f"Optimization complete. Best config: {best_config}")
        return best_config
    
    def apply_final_pruning(self, optimal_rates: Dict[str, float]):
        """
        Apply final structured pruning using optimal rates
        
        Args:
            optimal_rates: Optimal pruning rates from Bayesian optimization
        """
        logger.info("Applying final structured pruning...")
        
        total_params_before = sum(p.numel() for p in self.model.parameters())
        
        for group_name, sparsity in optimal_rates.items():
            logger.info(f"Pruning group {group_name} with sparsity {sparsity:.3f}")
            
            for layer_name in self.layer_groups[group_name]:
                module = dict(self.model.named_modules())[layer_name]
                
                if isinstance(module, nn.Conv2d) and layer_name in self.importance_rankings:
                    # Get number of filters to prune
                    num_filters = module.weight.size(0)
                    num_to_prune = int(num_filters * sparsity)
                    
                    if num_to_prune > 0:
                        # Get filters to remove (least important)
                        importance_ranking = self.importance_rankings[layer_name]
                        filters_to_remove = importance_ranking[:num_to_prune]
                        
                        # Create mask for remaining filters
                        keep_mask = torch.ones(num_filters, dtype=torch.bool)
                        keep_mask[filters_to_remove] = False
                        
                        # Apply structural pruning
                        with torch.no_grad():
                            # Prune output channels
                            module.weight.data = module.weight.data[keep_mask]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[keep_mask]
                            
                            # Update module parameters
                            module.out_channels = keep_mask.sum().item()
                        
                        logger.info(f"Pruned {layer_name}: {num_filters} -> {module.out_channels} filters")
        
        total_params_after = sum(p.numel() for p in self.model.parameters())
        reduction = (total_params_before - total_params_after) / total_params_before * 100
        
        logger.info(f"Pruning complete: {total_params_before} -> {total_params_after} parameters")
        logger.info(f"Parameter reduction: {reduction:.2f}%")
        
        return {
            'params_before': total_params_before,
            'params_after': total_params_after,
            'reduction_percent': reduction
        }


def create_nano_b_config(target_reduction: float = 0.4) -> Dict:
    """
    Create configuration for FeatherFace Nano-B pruning
    
    IMPORTANT: Le nombre final de paramètres sera VARIABLE (120K-180K) car:
    - L'optimisation bayésienne trouve automatiquement les taux optimaux
    - Chaque groupe de couches est optimisé indépendamment
    - Le résultat dépend de l'importance calculée par FPGM
    - Cette variabilité garantit des performances optimales vs un taux fixe
    
    Args:
        target_reduction: Target parameter reduction (0-1)
                         Note: Le résultat final peut varier selon BO
        
    Returns:
        Pruning configuration dictionary
        
    Example:
        config = create_nano_b_config(0.5)  # Cible 50% réduction
        # Résultat possible: 120K-180K paramètres selon optimisation
    """
    config = {
        # FPGM settings
        'distance_type': 'l2',  # or 'cosine'
        
        # SFP settings  
        'sparsity_schedule': 'polynomial',  # 'polynomial', 'exponential', 'linear'
        'soft_temperature': 1.0,
        
        # Bayesian optimization settings
        'num_groups': 6,
        'acquisition_function': 'ei',  # 'ei', 'ucb', 'pi'
        
        # Evaluation settings
        'eval_batches': 50,  # Number of batches for quick evaluation
        
        # Target reduction
        'target_reduction': target_reduction,
        
        # Training settings
        'pruning_epochs': 10,  # Epochs for gradual pruning
        'fine_tune_epochs': 20,  # Epochs for fine-tuning after pruning
    }
    
    return config


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = create_nano_b_config(target_reduction=0.4)
    print("FeatherFace Nano-B Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")