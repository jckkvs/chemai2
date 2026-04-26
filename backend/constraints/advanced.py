"""
Advanced Constraint Engine - chemai2/backend/constraints/advanced.py
Mathematically rigorous constraint enforcement with differentiable penalties
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats, interpolate
from scipy.sparse import csr_matrix, diags
from typing import Dict, List, Optional, Union, Tuple, Literal, Callable
from dataclasses import dataclass, field
import warnings

from backend.constraints.base import ConstraintSpec, ConstraintEvaluation
from backend.utils.logger import logger


@dataclass
class SmoothMonotonicPenalty:
    """
    Differentiable penalty function for monotonic constraints
    
    Uses softplus-based smooth approximation to enable gradient-based optimization
    """
    feature_name: str
    direction: Literal['increasing', 'decreasing']
    sigma_range: float = 3.0
    strength: Literal['strong', 'weak'] = 'weak'
    smoothness: float = 10.0  # Higher = sharper transition (closer to hard constraint)
    weight: float = 1.0
    
    def penalty_value(self, x: np.ndarray, y_pred: np.ndarray, 
                     feature_stats: Dict[str, float]) -> float:
        """
        Calculate smooth monotonicity penalty
        
        Penalty = weight * Σ softplus(-direction * dy/dx)
        where softplus(z) = log(1 + exp(z))
        """
        if len(x) < 2:
            return 0.0
        
        # Sort by feature value
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y_pred[sorted_idx]
        
        # Calculate numerical derivatives
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        
        # Avoid division by zero
        mask = dx > 1e-10
        if not np.any(mask):
            return 0.0
        
        slopes = dy[mask] / dx[mask]
        
        # Direction multiplier: +1 for increasing, -1 for decreasing
        dir_mult = 1.0 if self.direction == 'increasing' else -1.0
        
        # Softplus penalty: log(1 + exp(-smoothness * dir_mult * slope))
        # This is ~0 when constraint satisfied, grows when violated
        penalty_terms = np.log1p(np.exp(-self.smoothness * dir_mult * slopes))
        
        # Apply sigma-range weighting
        mean = feature_stats.get('mean', np.mean(x))
        std = feature_stats.get('std', np.std(x))
        if std > 0:
            z_scores = np.abs((x_sorted[:-1][mask] - mean) / std)
            # Weight more heavily within sigma_range
            weights = np.exp(-0.5 * np.maximum(0, z_scores - self.sigma_range) ** 2)
            penalty_terms *= weights
        
        return self.weight * np.mean(penalty_terms)
    
    def gradient(self, x: np.ndarray, y_pred: np.ndarray, 
                feature_stats: Dict[str, float]) -> np.ndarray:
        """
        Calculate gradient of penalty w.r.t. predictions
        
        Used for gradient-based optimization with constraints
        """
        if len(x) < 2:
            return np.zeros_like(y_pred)
        
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y_pred[sorted_idx]
        
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        
        mask = dx > 1e-10
        if not np.any(mask):
            return np.zeros_like(y_pred)
        
        slopes = dy[mask] / dx[mask]
        dir_mult = 1.0 if self.direction == 'increasing' else -1.0
        
        # Derivative of softplus: sigmoid(-smoothness * dir_mult * slope)
        sigmoid_terms = 1.0 / (1.0 + np.exp(self.smoothness * dir_mult * slopes))
        
        # Chain rule: d(penalty)/d(y_pred[i]) affects slopes[i-1] and slopes[i]
        grad = np.zeros(len(y_pred))
        
        mean = feature_stats.get('mean', np.mean(x))
        std = feature_stats.get('std', np.std(x))
        
        for i, m in enumerate(mask):
            if not m:
                continue
            z = np.abs((x_sorted[i] - mean) / std) if std > 0 else 0
            w = np.exp(-0.5 * np.maximum(0, z - self.sigma_range) ** 2)
            
            # Gradient flows to both endpoints of the slope
            grad[sorted_idx[i]] -= self.weight * w * sigmoid_terms[i] / dx[i] * dir_mult
            grad[sorted_idx[i + 1]] += self.weight * w * sigmoid_terms[i] / dx[i] * dir_mult
        
        # Reorder to original index
        inv_sort = np.argsort(sorted_idx)
        return grad[inv_sort]


@dataclass
class LinearDeviationPenalty:
    """
    Penalty for deviation from linearity
    
    Measures how much partial dependence deviates from best-fit line
    """
    feature_name: str
    strength: Literal['strong', 'weak'] = 'weak'
    reference_range: Tuple[float, float] = None  # (min, max) for linearity check
    weight: float = 1.0
    
    def penalty_value(self, x: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate linearity deviation penalty"""
        if len(x) < 3:
            return 0.0
        
        # Filter to reference range if specified
        if self.reference_range:
            mask = (x >= self.reference_range[0]) & (x <= self.reference_range[1])
            if np.sum(mask) < 3:
                return 0.0
            x_filtered = x[mask]
            y_filtered = y_pred[mask]
        else:
            x_filtered, y_filtered = x, y_pred
        
        # Fit linear model
        slope, intercept, r_value, _, _ = stats.linregress(x_filtered, y_filtered)
        y_linear = slope * x_filtered + intercept
        
        # Calculate normalized deviation
        residuals = y_filtered - y_linear
        rmse = np.sqrt(np.mean(residuals ** 2))
        y_range = np.ptp(y_filtered)  # peak-to-peak
        
        if y_range < 1e-10:
            return 0.0
        
        # Normalized deviation penalty
        deviation = rmse / y_range
        
        if self.strength == 'strong':
            # Hard penalty: exponential growth beyond threshold
            threshold = 0.05  # 5% deviation allowed
            return self.weight * np.exp(10 * max(0, deviation - threshold))
        else:
            # Soft penalty: quadratic growth
            return self.weight * deviation ** 2
    
    def target_linearity(self, x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return target linear predictions for projection"""
        if len(x) < 2:
            return y_pred
        
        slope, intercept, _, _, _ = stats.linregress(x, y_pred)
        return slope * x + intercept


@dataclass
class InteractionMonotonicitySpec:
    """
    Specification for monotonicity of feature interactions
    
    E.g., "when feature_A increases, the effect of feature_B on output should be monotonic"
    """
    primary_feature: str
    secondary_feature: str
    interaction_direction: Literal['increasing', 'decreasing']
    sigma_range: float = 3.0
    weight: float = 0.5  # Usually weaker than main effect constraints


class AdvancedConstraintEngine:
    """
    Enhanced constraint engine with differentiable penalties and interaction support
    """
    
    def __init__(
        self,
        monotonic_constraints: Dict[str, SmoothMonotonicPenalty] = None,
        linearity_constraints: Dict[str, LinearDeviationPenalty] = None,
        interaction_constraints: List[InteractionMonotonicitySpec] = None,
        feature_stats: Dict[str, Dict[str, float]] = None
    ):
        self.monotonic = monotonic_constraints or {}
        self.linearity = linearity_constraints or {}
        self.interactions = interaction_constraints or []
        self.feature_stats = feature_stats or {}
    
    def total_penalty(self, X: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate total constraint penalty for given predictions"""
        total = 0.0
        
        # Monotonic penalties
        for feat_name, constraint in self.monotonic.items():
            if feat_name in X.columns:
                x_vals = X[feat_name].values
                stats = self.feature_stats.get(feat_name, {})
                total += constraint.penalty_value(x_vals, y_pred, stats)
        
        # Linearity penalties
        for feat_name, constraint in self.linearity.items():
            if feat_name in X.columns:
                x_vals = X[feat_name].values
                total += constraint.penalty_value(x_vals, y_pred)
        
        # Interaction penalties (simplified pairwise check)
        for spec in self.interactions:
            if spec.primary_feature in X.columns and spec.secondary_feature in X.columns:
                penalty = self._interaction_penalty(
                    X[spec.primary_feature].values,
                    X[spec.secondary_feature].values,
                    y_pred,
                    spec
                )
                total += penalty
        
        return total
    
    def _interaction_penalty(
        self,
        x_primary: np.ndarray,
        x_secondary: np.ndarray,
        y_pred: np.ndarray,
        spec: InteractionMonotonicitySpec
    ) -> float:
        """
        Calculate penalty for interaction monotonicity violation
        
        Checks if the partial effect of secondary feature changes monotonically
        with primary feature value
        """
        if len(x_primary) < 10:
            return 0.0
        
        # Bin primary feature and check secondary effect in each bin
        n_bins = min(10, len(x_primary) // 5)
        bins = np.percentile(x_primary, np.linspace(0, 100, n_bins + 1))
        
        effects = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = (x_primary >= bins[i]) & (x_primary < bins[i + 1])
            if np.sum(mask) < 3:
                continue
            
            # Fit simple model: y ~ secondary in this bin
            x_sec = x_secondary[mask]
            y_bin = y_pred[mask]
            
            if len(np.unique(x_sec)) < 2:
                continue
            
            # Estimate effect size via correlation or simple slope
            slope, _, _, _, _ = stats.linregress(x_sec, y_bin)
            effects.append(slope)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
        
        if len(effects) < 2:
            return 0.0
        
        # Check if effects change monotonically with bin center
        effects = np.array(effects)
        centers = np.array(bin_centers)
        
        sorted_idx = np.argsort(centers)
        effects_sorted = effects[sorted_idx]
        
        # Count violations of expected monotonicity
        dir_mult = 1.0 if spec.interaction_direction == 'increasing' else -1.0
        violations = np.sum(np.diff(effects_sorted) * dir_mult < -1e-8)
        
        violation_ratio = violations / max(1, len(effects_sorted) - 1)
        return spec.weight * violation_ratio
    
    def project_to_constraints(self, X: pd.DataFrame, y_pred: np.ndarray, 
                              max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Project predictions to satisfy strong constraints via iterative optimization
        
        Uses gradient descent on penalty function
        """
        y_current = y_pred.copy()
        lr = 0.01  # Learning rate
        
        for iteration in range(max_iter):
            # Calculate gradient of total penalty
            grad = np.zeros_like(y_current)
            
            for feat_name, constraint in self.monotonic.items():
                if feat_name in X.columns and constraint.strength == 'strong':
                    x_vals = X[feat_name].values
                    stats = self.feature_stats.get(feat_name, {})
                    grad += constraint.gradient(x_vals, y_current, stats)
            
            # Update predictions
            y_new = y_current - lr * grad
            
            # Check convergence
            if np.max(np.abs(y_new - y_current)) < tol:
                logger.debug(f"Constraint projection converged in {iteration + 1} iterations")
                return y_new
            
            y_current = y_new
            # Adaptive learning rate
            if iteration % 20 == 0 and iteration > 0:
                lr *= 0.9
        
        logger.warning(f"Constraint projection did not fully converge after {max_iter} iterations")
        return y_current
