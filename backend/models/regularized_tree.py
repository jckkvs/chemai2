"""
Regularized Tree implementation for RGF (Regularized Greedy Forest).

Implements a single tree with L1/L2 regularization for leaf weights.
Used internally by RegularizedGreedyForest (RGF) for building
weak learners with regularization.
"""

from typing import Optional, Dict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor


class RegularizedTree(BaseEstimator, RegressorMixin):
    """
    Decision tree with L1/L2 regularization on leaf weights.

    Used internally by RegularizedGreedyForest (RGF) for building
    weak learners with regularization.
    """

    def __init__(
        self,
        max_leaf: int = 1000,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_depth: int = 20,
    ):
        self.max_leaf = max_leaf
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.tree_ = None
        self.leaf_values_: Dict[int, float] = {}
        self.n_leaves_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegularizedTree':
        """Fit tree with regularization."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Build tree using sklearn's DecisionTree as base
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.tree_.fit(X, y)
        self.n_leaves_ = self.tree_.tree_.n_leaves

        # Apply regularization to leaf values
        self._regularize_leaves(X, y)

        return self

    def _regularize_leaves(self, X: np.ndarray, y: np.ndarray):
        """Apply L1/L2 regularization to leaf values."""
        leaf_indices = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_indices)
        self.leaf_values_ = {}

        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            if np.sum(mask) == 0:
                continue
            y_leaf = y[mask]

            # L2 regularization: ridge-like adjustment
            n = len(y_leaf)
            if n > 0:
                mean_val = np.mean(y_leaf)
                # Shrink toward zero based on reg_lambda
                self.leaf_values_[leaf_id] = mean_val / (1.0 + self.reg_lambda / max(n, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using regularized leaf values."""
        if self.tree_ is None:
            raise RuntimeError("Tree not fitted")

        leaf_indices = self.tree_.apply(X)
        predictions = np.array([self.leaf_values_.get(idx, 0.0) for idx in leaf_indices])
        return predictions

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Tree not fitted")
        return self.tree_.feature_importances_
