"""
Constrained Estimator Registry - chemai2/backend/ml/estimators.py
Registry of all supported estimators with constraint-aware wrappers
"""
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Literal, Type, Callable
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)

from backend.ml.constraints import ConstraintSpec, ConstraintEngine
from backend.utils.logger import logger


# ========== Base Constraint Wrapper ==========
class ConstrainedEstimatorMixin:
    """Mixin for adding constraint support to any estimator"""
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        constraints: Dict[str, ConstraintSpec],
        task_type: Literal['regression', 'classification'] = 'regression',
        sigma_multiplier: float = 3.0,
        constraint_engine: Optional[ConstraintEngine] = None
    ):
        self.base_estimator = clone(base_estimator)
        self.constraints = constraints or {}
        self.task_type = task_type
        self.sigma_multiplier = sigma_multiplier
        self._constraint_engine = constraint_engine
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False
        
    def _store_feature_stats(self, X: Union[pd.DataFrame, np.ndarray], feature_names: List[str] = None):
        """Store feature statistics for sigma-range constraint enforcement"""
        if isinstance(X, pd.DataFrame):
            df = X
            feature_names = feature_names or list(X.columns)
        else:
            df = pd.DataFrame(X, columns=feature_names)
            feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        for feat in feature_names:
            if feat in self.constraints and feat in df.columns:
                values = df[feat].dropna()
                if len(values) > 1 and pd.api.types.is_numeric_dtype(values):
                    self._feature_stats[feat] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                    }
    
    def _get_sigma_range(self, feature_name: str) -> tuple:
        """Get ±nσ range for a feature"""
        stats = self._feature_stats.get(feature_name, {})
        if not stats:
            return None, None
        return (
            stats['mean'] - self.sigma_multiplier * stats['std'],
            stats['mean'] + self.sigma_multiplier * stats['std']
        )
    
    def _apply_native_monotonic_constraints(self) -> Optional[tuple]:
        """Apply native monotonic constraints if estimator supports them"""
        # XGBoost/LightGBM style monotonic_constraints
        if hasattr(self.base_estimator, 'set_params') and 'monotonic_constraints' in self.base_estimator.get_params():
            # Build constraint array
            # This requires knowing feature order - handled in fit
            pass
        return None
    
    def _posthoc_constraint_correction(self, X: Union[pd.DataFrame, np.ndarray], predictions: np.ndarray) -> np.ndarray:
        """Apply post-hoc correction for strong constraints"""
        corrected = predictions.copy()
        
        for feat_name, spec in self.constraints.items():
            if spec.strength != 'strong' or not spec.monotonic:
                continue
            if feat_name not in self._feature_stats:
                continue
            
            # Get feature values and predictions
            if isinstance(X, pd.DataFrame):
                if feat_name not in X.columns:
                    continue
                feature_values = X[feat_name].values
            else:
                continue  # Need column names for non-DataFrame
            
            # Apply isotonic regression for monotonic correction
            try:
                increasing = (spec.monotonic == 'increasing')
                iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
                
                # Fit on feature vs prediction pairs
                iso.fit(feature_values, predictions)
                corrected = iso.predict(feature_values)
            except Exception as e:
                logger.debug(f"Isotonic correction failed for {feat_name}: {e}")
                continue
        
        return corrected
    
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit with constraint-aware preprocessing"""
        # Store feature stats for sigma-range
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self._store_feature_stats(X, feature_names)
        
        # Apply native constraints if supported
        native_constraints = self._apply_native_monotonic_constraints()
        if native_constraints:
            self.base_estimator.set_params(monotonic_constraints=native_constraints)
        
        # For weak constraints: could add custom sample weights or callbacks
        # (Implementation depends on estimator)
        
        # Fit base estimator
        if sample_weight is not None:
            self.base_estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        else:
            self.base_estimator.fit(X, y, **fit_params)
        
        self._fitted = True
        return self
    
    def predict(self, X):
        """Predict with optional post-hoc constraint enforcement"""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before prediction")
        
        predictions = self.base_estimator.predict(X)
        
        # Apply strong constraint post-processing
        has_strong = any(c.strength == 'strong' for c in self.constraints.values() if c.monotonic)
        if has_strong:
            predictions = self._posthoc_constraint_correction(X, predictions)
        
        return predictions
    
    def __getattr__(self, name):
        """Delegate all other attributes to base_estimator"""
        if name.startswith('_') or name in ['base_estimator', 'constraints', 'task_type']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.base_estimator, name)
    
    def __sklearn_is_fitted__(self):
        return self._fitted


class ConstrainedEstimatorWrapper(ConstrainedEstimatorMixin, BaseEstimator):
    """Concrete wrapper class for constrained estimators"""
    pass


# ========== Specific Estimator Wrappers ==========
class ConstrainedTreeBasedEstimator(ConstrainedEstimatorMixin, BaseEstimator):
    """Specialized wrapper for tree-based estimators with enhanced constraint support"""
    
    def _apply_native_monotonic_constraints(self) -> Optional[tuple]:
        """Apply native monotonic constraints for tree-based models"""
        # HistGradientBoosting supports monotonic_cst
        if hasattr(self.base_estimator, 'monotonic_cst'):
            return self._build_monotonic_array()
        return None
    
    def _build_monotonic_array(self) -> Optional[tuple]:
        """Build monotonic constraint array for tree models"""
        # This requires knowing the feature order at fit time
        # Return placeholder - actual implementation in fit method
        return None
    
    def fit(self, X, y, **fit_params):
        # For HistGradientBoosting, set monotonic_cst before fit
        if hasattr(self.base_estimator, 'monotonic_cst'):
            constraints_array = self._build_monotonic_cst_array(X)
            if constraints_array:
                self.base_estimator.set_params(monotonic_cst=constraints_array)
        return super().fit(X, y, **fit_params)
    
    def _build_monotonic_cst_array(self, X) -> Optional[tuple]:
        """Build monotonic_cst array for HistGradientBoosting"""
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        if not feature_names:
            return None
        
        constraints = []
        for feat in feature_names:
            spec = self.constraints.get(feat)
            if spec and spec.monotonic and spec.strength == 'strong':
                if spec.monotonic == 'increasing':
                    constraints.append(1)
                elif spec.monotonic == 'decreasing':
                    constraints.append(-1)
                else:
                    constraints.append(0)
            else:
                constraints.append(0)
        
        return tuple(constraints) if constraints else None


class ConstrainedLinearEstimator(ConstrainedEstimatorMixin, BaseEstimator):
    """Wrapper for linear models with constraint-aware regularization"""
    
    def _add_constraint_penalty(self, X, y, sample_weight=None):
        """Add penalty terms for weak constraints to linear model loss"""
        # For weak linearity constraints: add ridge-like penalty on non-linear terms
        # This is a simplified approach; full implementation would modify the loss
        pass
    
    def fit(self, X, y, **fit_params):
        # Could add constraint-based regularization here
        return super().fit(X, y, **fit_params)


# ========== Estimator Registry ==========
ESTIMATOR_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    # Linear models
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    
    # Tree-based (with native constraint support)
    'RandomForestRegressor': RandomForestRegressor,
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesRegressor': ExtraTreesRegressor,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
}

# Wrapper mapping: which wrapper to use for each estimator
WRAPPER_MAPPING: Dict[str, Type[ConstrainedEstimatorMixin]] = {
    'HistGradientBoostingRegressor': ConstrainedTreeBasedEstimator,
    'HistGradientBoostingClassifier': ConstrainedTreeBasedEstimator,
    'LinearRegression': ConstrainedLinearEstimator,
    'Ridge': ConstrainedLinearEstimator,
    'Lasso': ConstrainedLinearEstimator,
    'ElasticNet': ConstrainedLinearEstimator,
    # Default for others
    'default': ConstrainedEstimatorWrapper,
}


def get_estimator_class(name: str) -> Type[BaseEstimator]:
    """Get estimator class from registry"""
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator: {name}. Available: {list(ESTIMATOR_REGISTRY.keys())}")
    return ESTIMATOR_REGISTRY[name]


def get_constrained_wrapper(estimator_name: str, base_estimator: BaseEstimator, 
                           constraints: Dict[str, ConstraintSpec],
                           task_type: str) -> ConstrainedEstimatorMixin:
    """Get appropriate constrained wrapper for an estimator"""
    wrapper_class = WRAPPER_MAPPING.get(estimator_name, WRAPPER_MAPPING['default'])
    return wrapper_class(
        base_estimator=base_estimator,
        constraints=constraints,
        task_type=task_type
    )


def create_constrained_estimator(
    estimator_name: str,
    estimator_params: Dict[str, Any],
    constraints: Dict[str, ConstraintSpec],
    task_type: str = 'regression'
) -> ConstrainedEstimatorMixin:
    """Factory function to create constrained estimator"""
    base_class = get_estimator_class(estimator_name)
    base_estimator = base_class(**estimator_params)
    return get_constrained_wrapper(estimator_name, base_estimator, constraints, task_type)
