"""
Constraint Enforcement Engine - chemai2/backend/constraints/engine.py
Mathematical enforcement of monotonicity and linearity constraints
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union, Literal, Callable
from dataclasses import dataclass, field
import warnings

from sklearn.base import BaseEstimator, is_regressor, is_classifier
from sklearn.isotonic import IsotonicRegression

from backend.utils.logger import logger


@dataclass
class ConstraintSpec:
    """Specification for a single feature constraint"""
    feature_name: str
    monotonic: Optional[Literal['increasing', 'decreasing', 'either']] = None
    linearity: Optional[Literal['strong', 'weak', 'none']] = 'none'
    sigma_range: float = 3.0  # ±nσ enforcement range
    strength: Literal['strong', 'weak'] = 'weak'
    weight: float = 1.0  # Relative importance for weak constraints
    
    def __post_init__(self):
        if self.monotonic is None and self.linearity == 'none':
            warnings.warn(f"ConstraintSpec for {self.feature_name} has no active constraints")


@dataclass
class ConstraintEvaluation:
    """Result of constraint evaluation on a model"""
    feature_name: str
    monotonic_violations: int = 0
    monotonic_violation_ratio: float = 0.0  # 0.0 = perfect, 1.0 = all violated
    linearity_r2: Optional[float] = None  # R² of linear fit to partial dependence
    linearity_rmse: Optional[float] = None
    sigma_range_min: Optional[float] = None
    sigma_range_max: Optional[float] = None
    passed: bool = True
    details: Dict[str, any] = field(default_factory=dict)


class ConstraintEngine:
    """
    Core engine for evaluating and enforcing constraints
    
    Features:
    - Monotonicity verification via partial dependence profiles
    - Linearity assessment via R² of linear fit
    - Sigma-range enforcement (±nσ from training distribution)
    - Weak constraint penalty calculation for custom loss functions
    - Post-hoc prediction correction for strong constraints
    """
    
    def __init__(
        self,
        constraints: Dict[str, ConstraintSpec],
        n_grid_points: int = 50,
        random_state: int = 42
    ):
        self.constraints = constraints
        self.n_grid_points = n_grid_points
        self.random_state = random_state
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._rng = np.random.default_rng(random_state)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: List[str] = None):
        """Store feature statistics for sigma-range calculations"""
        if isinstance(X, pd.DataFrame):
            df = X
            feature_names = feature_names or list(X.columns)
        else:
            df = pd.DataFrame(X, columns=feature_names)
            feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        for feat in feature_names:
            if feat in self.constraints and df[feat].dtype in [np.float64, np.int64, float, int]:
                values = df[feat].dropna()
                if len(values) > 1:
                    self._feature_stats[feat] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'q01': float(values.quantile(0.01)),
                        'q99': float(values.quantile(0.99)),
                    }
        
        return self
    
    def evaluate_constraints(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str] = None
    ) -> Dict[str, ConstraintEvaluation]:
        """
        Evaluate how well a trained model satisfies constraints
        
        Uses partial dependence profiles to assess monotonicity/linearity
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feature_names = feature_names or list(X.columns)
        else:
            df = pd.DataFrame(X, columns=feature_names)
            feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        results = {}
        
        for feat_name, spec in self.constraints.items():
            if feat_name not in df.columns:
                logger.warning(f"Feature {feat_name} not found in data, skipping constraint evaluation")
                continue
            
            evaluation = self._evaluate_single_constraint(
                model, df, y, feat_name, spec, feature_names
            )
            results[feat_name] = evaluation
        
        return results
    
    def _evaluate_single_constraint(
        self,
        model: BaseEstimator,
        df: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        feature_name: str,
        spec: ConstraintSpec,
        all_features: List[str]
    ) -> ConstraintEvaluation:
        """Evaluate constraints for a single feature"""
        eval_result = ConstraintEvaluation(feature_name=feature_name)
        
        # Calculate sigma range
        stats = self._feature_stats.get(feature_name, {})
        if stats:
            eval_result.sigma_range_min = stats['mean'] - spec.sigma_range * stats['std']
            eval_result.sigma_range_max = stats['mean'] + spec.sigma_range * stats['std']
        
        # Generate partial dependence profile
        pdp_values = self._compute_partial_dependence(
            model, df, feature_name, all_features,
            x_range=(eval_result.sigma_range_min, eval_result.sigma_range_max)
        )
        
        if pdp_values is None or len(pdp_values) < 2:
            eval_result.passed = False
            eval_result.details['error'] = 'Could not compute partial dependence'
            return eval_result
        
        x_grid = pdp_values[feature_name].values
        y_pdp = pdp_values['prediction'].values
        
        # Evaluate monotonicity
        if spec.monotonic:
            violations, ratio = self._check_monotonicity(
                x_grid, y_pdp, spec.monotonic, spec.sigma_range
            )
            eval_result.monotonic_violations = violations
            eval_result.monotonic_violation_ratio = ratio
            
            if ratio > 0.1:  # >10% violations = failed
                eval_result.passed = False
                eval_result.details['monotonicity_issue'] = f'{ratio*100:.1f}% violations'
        
        # Evaluate linearity
        if spec.linearity and spec.linearity != 'none':
            r2, rmse = self._check_linearity(x_grid, y_pdp)
            eval_result.linearity_r2 = r2
            eval_result.linearity_rmse = rmse
            
            if spec.linearity == 'strong' and r2 < 0.95:
                eval_result.passed = False
                eval_result.details['linearity_issue'] = f'R²={r2:.3f} < 0.95'
            elif spec.linearity == 'weak' and r2 < 0.7:
                eval_result.passed = False
                eval_result.details['linearity_issue'] = f'R²={r2:.3f} < 0.7 (weak threshold)'
        
        eval_result.details['pdp_sample'] = {
            'x_min': float(x_grid.min()),
            'x_max': float(x_grid.max()),
            'y_min': float(y_pdp.min()),
            'y_max': float(y_pdp.max()),
        }
        
        return eval_result
    
    def _compute_partial_dependence(
        self,
        model: BaseEstimator,
        df: pd.DataFrame,
        feature_name: str,
        all_features: List[str],
        x_range: Tuple[Optional[float], Optional[float]] = None,
        n_samples: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Compute partial dependence profile for a single feature
        
        Uses Monte Carlo averaging over other features
        """
        if feature_name not in df.columns:
            return None
        
        # Get feature range
        values = df[feature_name].dropna()
        if x_range[0] is not None and x_range[1] is not None:
            x_min, x_max = x_range
        else:
            x_min, x_max = values.min(), values.max()
        
        if x_min == x_max:
            return None
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, self.n_grid_points)
        
        # Sample background data for marginalization
        n_background = min(n_samples, len(df))
        background = df.sample(n=n_background, random_state=self.random_state)
        
        predictions = []
        
        for x_val in x_grid:
            # Create modified dataset with feature fixed to x_val
            X_modified = background.copy()
            X_modified[feature_name] = x_val
            
            # Predict
            try:
                preds = model.predict(X_modified)
                predictions.append(np.mean(preds))
            except Exception as e:
                logger.debug(f"Prediction failed at x={x_val}: {e}")
                predictions.append(np.nan)
        
        return pd.DataFrame({
            feature_name: x_grid,
            'prediction': predictions
        }).dropna()
    
    def _check_monotonicity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        direction: Literal['increasing', 'decreasing', 'either'],
        sigma_range: float
    ) -> Tuple[int, float]:
        """
        Check monotonicity of y with respect to x
        
        Returns: (violation_count, violation_ratio)
        """
        if len(x) < 2:
            return 0, 0.0
        
        # Sort by x
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Calculate differences
        dy = np.diff(y_sorted)
        
        if direction == 'increasing':
            violations = np.sum(dy < -1e-8)  # Allow small numerical tolerance
        elif direction == 'decreasing':
            violations = np.sum(dy > 1e-8)
        else:  # 'either' - just check for any non-monotonic behavior
            # Count sign changes in differences
            sign_changes = np.sum(np.diff(np.sign(dy)) != 0)
            violations = sign_changes
        
        ratio = violations / max(1, len(dy))
        return int(violations), float(ratio)
    
    def _check_linearity(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Assess linearity via R² of linear regression fit
        
        Returns: (r_squared, root_mean_squared_error)
        """
        if len(x) < 3:
            return 0.0, float('inf')
        
        # Fit linear model
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2
        
        # Calculate RMSE
        y_pred = slope * x + intercept
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return float(r2), float(rmse)
    
    def calculate_weak_penalty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        constraints: Dict[str, ConstraintSpec],
        base_loss: float
    ) -> float:
        """
        Calculate penalty term for weak constraints
        
        Adds to base loss function during training
        """
        penalty = 0.0
        
        for feat_name, spec in constraints.items():
            if feat_name not in X.columns or spec.strength != 'weak':
                continue
            
            # Compute partial dependence
            pdp = self._compute_partial_dependence(model, X, feat_name, list(X.columns))
            if pdp is None:
                continue
            
            x_grid = pdp[feat_name].values
            y_pdp = pdp['prediction'].values
            
            # Monotonicity penalty
            if spec.monotonic:
                _, violation_ratio = self._check_monotonicity(
                    x_grid, y_pdp, spec.monotonic, spec.sigma_range
                )
                penalty += spec.weight * violation_ratio * 10.0  # Scale factor
            
            # Linearity penalty
            if spec.linearity == 'weak':
                r2, _ = self._check_linearity(x_grid, y_pdp)
                # Penalty increases as R² decreases below 0.9
                if r2 < 0.9:
                    penalty += spec.weight * (0.9 - r2) * 5.0
        
        return base_loss + penalty
    
    def enforce_strong_constraints_posthoc(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        predictions: np.ndarray,
        constraints: Dict[str, ConstraintSpec]
    ) -> np.ndarray:
        """
        Post-hoc correction of predictions to enforce strong constraints
        
        Uses isotonic regression per constrained feature
        """
        corrected = predictions.copy()
        
        for feat_name, spec in constraints.items():
            if feat_name not in X.columns or spec.strength != 'strong':
                continue
            if not spec.monotonic:
                continue
            
            # Group predictions by feature value bins
            feature_values = X[feat_name].values
            unique_vals = np.unique(feature_values)
            
            if len(unique_vals) < 10:
                continue  # Not enough variation
            
            # Apply isotonic regression
            try:
                iso_reg = IsotonicRegression(
                    increasing=(spec.monotonic == 'increasing'),
                    out_of_bounds='clip'
                )
                iso_reg.fit(feature_values, predictions)
                corrected = iso_reg.predict(feature_values)
            except Exception as e:
                logger.warning(f"Isotonic regression failed for {feat_name}: {e}")
                continue
        
        return corrected
    
    def generate_constraint_report(
        self,
        evaluations: Dict[str, ConstraintEvaluation]
    ) -> Dict[str, any]:
        """Generate human-readable constraint evaluation report"""
        report = {
            'summary': {
                'total_constraints': len(evaluations),
                'passed': sum(1 for e in evaluations.values() if e.passed),
                'failed': sum(1 for e in evaluations.values() if not e.passed),
            },
            'details': {}
        }
        
        for feat_name, eval_result in evaluations.items():
            report['details'][feat_name] = {
                'passed': eval_result.passed,
                'monotonicity': {
                    'direction': self.constraints[feat_name].monotonic,
                    'violations': eval_result.monotonic_violations,
                    'violation_ratio': f"{eval_result.monotonic_violation_ratio*100:.1f}%",
                } if eval_result.monotonic_violation_ratio is not None else None,
                'linearity': {
                    'target': self.constraints[feat_name].linearity,
                    'r2': f"{eval_result.linearity_r2:.3f}" if eval_result.linearity_r2 else None,
                    'rmse': f"{eval_result.linearity_rmse:.3f}" if eval_result.linearity_rmse else None,
                } if eval_result.linearity_r2 is not None else None,
                'sigma_range': f"[{eval_result.sigma_range_min:.2f}, {eval_result.sigma_range_max:.2f}]" 
                    if eval_result.sigma_range_min else None,
                'issues': eval_result.details.get('monotonicity_issue') or eval_result.details.get('linearity_issue'),
            }
        
        return report
