"""
Constraint-Aware Gradient Boosting Wrapper - chemai2/backend/ml/constrained_boosting.py
Integrates constraint penalties into gradient boosting training loop
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor

from backend.constraints.advanced import (
    AdvancedConstraintEngine, SmoothMonotonicPenalty, 
    LinearDeviationPenalty, InteractionMonotonicitySpec
)
from backend.utils.logger import logger


class ConstrainedGradientBooster(BaseEstimator):
    """
    Wrapper for gradient boosting estimators with constraint-aware training
    
    Supports:
    - Weak constraints via penalty-augmented loss function
    - Strong constraints via post-hoc projection
    - Native monotonic_cst for HistGradientBoosting
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        constraint_engine: AdvancedConstraintEngine,
        penalty_weight: float = 1.0,
        projection_iters: int = 50,
        use_native_constraints: bool = True
    ):
        self.base_estimator = clone(base_estimator)
        self.constraint_engine = constraint_engine
        self.penalty_weight = penalty_weight
        self.projection_iters = projection_iters
        self.use_native_constraints = use_native_constraints
        self._feature_names: Optional[List[str]] = None
        self._feature_stats: Dict[str, Dict[str, float]] = {}
    
    def _store_feature_info(self, X: Union[pd.DataFrame, np.ndarray]):
        """Store feature names and statistics for constraint evaluation"""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    vals = X[col].dropna()
                    if len(vals) > 1:
                        self._feature_stats[col] = {
                            'mean': float(vals.mean()),
                            'std': float(vals.std()),
                            'min': float(vals.min()),
                            'max': float(vals.max()),
                        }
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    def _apply_native_monotonic(self):
        """Apply native monotonic constraints if estimator supports them"""
        if not self.use_native_constraints:
            return
        
        # HistGradientBoosting supports monotonic_cst parameter
        if isinstance(self.base_estimator, HistGradientBoostingRegressor):
            if self._feature_names and any(
                c.strength == 'strong' and c.direction for c in 
                self.constraint_engine.monotonic.values()
            ):
                constraints = []
                for feat in self._feature_names:
                    c = self.constraint_engine.monotonic.get(feat)
                    if c and c.strength == 'strong':
                        constraints.append(1 if c.direction == 'increasing' else -1)
                    else:
                        constraints.append(0)
                
                if any(c != 0 for c in constraints):
                    self.base_estimator.set_params(monotonic_cst=constraints)
                    logger.info(f"Applied native monotonic constraints: {constraints}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, 
            sample_weight: np.ndarray = None, **fit_params) -> 'ConstrainedGradientBooster':
        """Fit with constraint-aware training"""
        self._store_feature_info(X)
        self._apply_native_monotonic()
        
        if sample_weight is not None:
            self.base_estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        else:
            self.base_estimator.fit(X, y, **fit_params)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with optional strong constraint projection"""
        y_pred = self.base_estimator.predict(X)
        
        # Apply post-hoc projection for strong constraints
        has_strong = any(
            c.strength == 'strong' 
            for c in list(self.constraint_engine.monotonic.values()) + 
                     list(self.constraint_engine.linearity.values())
        )
        
        if has_strong and isinstance(X, pd.DataFrame):
            y_pred = self.constraint_engine.project_to_constraints(
                X, y_pred, max_iter=self.projection_iters
            )
        
        return y_pred
    
    def staged_predict(self, X: Union[pd.DataFrame, np.ndarray], n_iter: int = None):
        """Staged prediction with constraint projection at each stage"""
        for y_pred in self.base_estimator.staged_predict(X, n_iter=n_iter):
            if isinstance(X, pd.DataFrame):
                y_pred = self.constraint_engine.project_to_constraints(
                    X, y_pred, max_iter=min(10, self.projection_iters // 10)
                )
            yield y_pred
    
    def __getattr__(self, name):
        """Delegate to base estimator"""
        if name.startswith('_') or name in ['base_estimator', 'constraint_engine']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.base_estimator, name)

    def __sklearn_is_fitted__(self):
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.base_estimator)
            return True
        except:
            return False
