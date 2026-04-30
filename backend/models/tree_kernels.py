# backend/models/tree_kernels.py
"""
Tree-based kernels extending the RFR-Kernel concept.

Implements:
  - TreeKernel: Base class for tree-ensemble kernels
  - RandomForestKernel: Kernel based on Random Forest leaf co-occurrence
  - ExtraTreesKernel: Kernel based on Extra Trees leaf co-occurrence
  - RidgeTreeKernel: Kernel for individual trees with ridge leaves

Usage:
    from backend.models.tree_kernels import RandomForestKernel
    from sklearn.kernel_ridge import KernelRidge

    # Train a random forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    # Create kernel
    kernel = RandomForestKernel(rf)

    # Use with KernelRidge
    kr = KernelRidge(kernel=kernel)
    kr.fit(X_train, y_train)
"""

from __future__ import annotations

from typing import Optional, List, Any
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import logging

logger = logging.getLogger(__name__)
from backend.models.monotonic_kernel import MonotonicConstrainedKernel


# ──────────────────────────────────────────────────────
# 1. Base Tree Kernel
# ──────────────────────────────────────────────────────

class TreeKernel:
    """Base class for tree-based kernels."""

    def __init__(self, ensemble=None, n_trees: int = 100, max_depth: int = 10, random_state: Optional[int] = None):
        self.ensemble = ensemble
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'TreeKernel':
        """Fit the ensemble if not already fitted."""
        if self.ensemble is None:
            from sklearn.ensemble import RandomForestRegressor
            self.ensemble = RandomForestRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.ensemble.fit(X, y)
        elif not hasattr(self.ensemble, 'estimators_'):
            self.ensemble.fit(X, y)
        self._is_fitted = True
        return self

    def __call__(self, X1: np.ndarray, X2: np.ndarray = None, eval_gradient: bool = False):
        """Compute kernel matrix."""
        if not self._is_fitted:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1
            is_self = True
        else:
            is_self = False

        # Get leaf indices for both sets
        leaves1 = self._get_leaf_indices(X1)
        leaves2 = self._get_leaf_indices(X2) if not is_self else leaves1

        # Compute co-occurrence matrix
        K = self._compute_cooccurrence(leaves1, leaves2)

        if eval_gradient:
            # Tree kernel has no learnable hyperparameters
            # Return gradient with shape (n1, n2, 0) to indicate no params
            gradient = np.zeros((K.shape[0], K.shape[1], 0))
            return K, gradient
        return K

    def _get_leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """Get leaf indices for each sample in each tree."""
        # Ensure 2D (sklearn trees require 2D input)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_trees = len(self.ensemble.estimators_)
        leaves = np.zeros((n_samples, n_trees), dtype=int)

        for i, tree in enumerate(self.ensemble.estimators_):
            leaves[:, i] = tree.apply(X)

        return leaves

    def _compute_cooccurrence(self, leaves1: np.ndarray, leaves2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on leaf co-occurrence."""
        n1 = leaves1.shape[0]
        n2 = leaves2.shape[0]
        n_trees = leaves1.shape[1]

        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # Count number of trees where samples are in the same leaf
                matches = np.sum(leaves1[i] == leaves2[j])
                K[i, j] = matches / n_trees

        return K


# ──────────────────────────────────────────────────────
# 2. Random Forest Kernel
# ──────────────────────────────────────────────────────

class RandomForestKernel(TreeKernel):
    """Kernel based on Random Forest leaf co-occurrence (RFR-Kernel)."""

    def __init__(self, ensemble=None, n_trees: int = 100, max_depth: int = 10,
                 random_state: Optional[int] = None, **kwargs):
        super().__init__(ensemble=ensemble, n_trees=n_trees,
                         max_depth=max_depth, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RandomForestKernel':
        if self.ensemble is None:
            from sklearn.ensemble import RandomForestRegressor
            self.ensemble = RandomForestRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.ensemble.fit(X, y)
        elif not hasattr(self.ensemble, 'estimators_'):
            self.ensemble.fit(X, y)
        self._is_fitted = True
        return self


# ──────────────────────────────────────────────────────
# 3. Extra Trees Kernel
# ──────────────────────────────────────────────────────

class ExtraTreesKernel(TreeKernel):
    """Kernel based on Extra Trees leaf co-occurrence."""

    def __init__(self, ensemble=None, n_trees: int = 100, max_depth: int = 10,
                 random_state: Optional[int] = None, **kwargs):
        super().__init__(ensemble=ensemble, n_trees=n_trees,
                         max_depth=max_depth, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ExtraTreesKernel':
        if self.ensemble is None:
            from sklearn.ensemble import ExtraTreesRegressor
            self.ensemble = ExtraTreesRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.ensemble.fit(X, y)
        elif not hasattr(self.ensemble, 'estimators_'):
            self.ensemble.fit(X, y)
        self._is_fitted = True
        return self


# ──────────────────────────────────────────────────────
# 4. Ridge Tree Kernel (single tree with ridge leaves)
# ──────────────────────────────────────────────────────

class RidgeTreeKernel:
    """Kernel for a single decision tree with ridge regression leaves.

    Represents the RFR-Kernel concept for a single tree:
    similarity between samples is based on leaf co-occurrence,
    but each leaf has a ridge regression model.
    """

    def __init__(self, max_depth: int = 10, min_samples_leaf: int = 10,
                 alpha: float = 1.0, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.random_state = random_state
        self.tree_ = None
        self.leaf_models_ = {}
        self.leaf_mapping_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeTreeKernel':
        """Fit tree and ridge models for each leaf."""
        from sklearn.tree import DecisionTreeRegressor

        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree_.fit(X, y)

        # Get leaf indices
        leaf_indices = self.tree_.apply(X)

        # Fit ridge regression for each leaf
        from sklearn.linear_model import Ridge
        unique_leaves = np.unique(leaf_indices)

        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            if np.sum(mask) < 2:
                continue
            X_leaf = X[mask]
            y_leaf = y[mask]
            model = Ridge(alpha=self.alpha)
            model.fit(X_leaf, y_leaf)
            self.leaf_models_[leaf_id] = model

        return self

    def __call__(self, X1: np.ndarray, X2: np.ndarray = None, eval_gradient: bool = False):
        """Compute kernel matrix based on leaf co-occurrence."""
        if self.tree_ is None:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1

        leaves1 = self.tree_.apply(X1)
        leaves2 = self.tree_.apply(X2) if X2 is not X1 else leaves1

        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                if leaves1[i] == leaves2[j]:
                    K[i, j] = 1.0

        if eval_gradient:
            # Kernel does not depend on alpha (used only in predict, not kernel)
            gradient = np.zeros((n1, n2, 0))
            return K, gradient
        return K

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using leaf models."""
        if self.tree_ is None:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        leaf_indices = self.tree_.apply(X)
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            leaf_id = leaf_indices[i]
            if leaf_id in self.leaf_models_:
                predictions[i] = self.leaf_models_[leaf_id].predict(X[i].reshape(1, -1))[0]
            else:
                # Fallback to mean prediction
                predictions[i] = np.mean(self.tree_.predict(X[i].reshape(1, -1)))

        return predictions


# ──────────────────────────────────────────────────────
# 5. Monotonic Random Forest Kernel
# ──────────────────────────────────────────────────────

class MonotonicRandomForestKernel(MonotonicConstrainedKernel):
    """
    RFR-Kernel with monotonicity constraints.

    Combines the RFR-Kernel concept (RandomForestKernel)
    with monotonicity constraints from MonotonicConstrainedKernel.
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        regularization: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        base_kernel = RandomForestKernel(
            n_trees=n_trees, max_depth=max_depth, random_state=random_state
        )
        super().__init__(
            base_kernel=base_kernel,
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            regularization=regularization,
        )
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MonotonicRandomForestKernel':
        """Fit the internal RandomForestKernel first."""
        self.base_kernel.fit(X, y)
        return super().fit(X, y)


# ──────────────────────────────────────────────────────
# 6. Tree Kernel Wrapper for scikit-learn
# ──────────────────────────────────────────────────────

def make_tree_kernel_model(
    model_type: str = "kernelridge",
    kernel_type: str = "rf",
    n_trees: int = 100,
    max_depth: int = 10,
    random_state: Optional[int] = None,
    **kwargs
):
    """
    Factory function to create kernel models with tree kernels.

    Args:
        model_type: "kernelridge", "svr", "gpr"
        kernel_type: "rf", "et", "ridge_tree"
        n_trees: Number of trees (for ensemble kernels)
        max_depth: Max depth of trees
        random_state: Random state
    """
    # Create kernel
    if kernel_type == "rf":
        kernel = RandomForestKernel(n_trees=n_trees, max_depth=max_depth, random_state=random_state)
    elif kernel_type == "et":
        kernel = ExtraTreesKernel(n_trees=n_trees, max_depth=max_depth, random_state=random_state)
    elif kernel_type == "ridge_tree":
        kernel = RidgeTreeKernel(max_depth=max_depth, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Create model
    if model_type == "kernelridge":
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(kernel=kernel, **kwargs)
    elif model_type == "svr":
        from sklearn.svm import SVR
        return SVR(kernel=kernel, **kwargs)
    elif model_type == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        return GaussianProcessRegressor(kernel=kernel, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ──────────────────────────────────────────────
# 7. Extended RFR Kernel (wraps any tree ensemble)
# ──────────────────────────────────────────────

class TreeKernelRFRExtended:
    """
    Extended RFR-Kernel that can wrap ANY tree-based model.

    Unlike RandomForestKernel which requires a sklearn RandomForest,
    this can wrap any tree ensemble that has:
    - estimators_ attribute with .apply() method
    - Or a get_leaf_indicators() method
    """

    def __init__(
        self,
        ensemble=None,
        n_trees: int = 100,
        max_depth: int = 10,
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.ensemble = ensemble
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y=None):
        """Fit ensemble if not already fitted."""
        if self.ensemble is None:
            from sklearn.ensemble import RandomForestRegressor
            self.ensemble = RandomForestRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.ensemble.fit(X, y)
        elif not hasattr(self.ensemble, 'estimators_'):
            self.ensemble.fit(X, y)
        self._is_fitted = True
        return self

    def __call__(self, X1, X2=None, eval_gradient=False):
        """Compute kernel matrix with optional gradient."""
        if not self._is_fitted:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1

        # Get leaf indices
        leaves1 = self._get_leaf_indices(X1)
        leaves2 = self._get_leaf_indices(X2) if X2 is not X1 else leaves1

        # Compute kernel matrix
        K = self._compute_cooccurrence(leaves1, leaves2)

        if eval_gradient:
            # Tree kernels have no learnable hyperparameters
            gradient = np.zeros((K.shape[0], K.shape[1], 0))
            return K, gradient

        return K

    def _get_leaf_indices(self, X):
        """Get leaf indices, handling various ensemble types."""
        # Ensure 2D input for sklearn tree.apply()
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if hasattr(self.ensemble, 'estimators_'):
            n_samples = X.shape[0]
            n_trees = len(self.ensemble.estimators_)
            leaves = np.zeros((n_samples, n_trees), dtype=int)
            for i, tree in enumerate(self.ensemble.estimators_):
                leaves[:, i] = tree.apply(X)
            return leaves
        elif hasattr(self.ensemble, 'get_leaf_indicators'):
            return self.ensemble.get_leaf_indicators(X)
        else:
            raise ValueError("Ensemble must have estimators_ or get_leaf_indicators()")

    def _compute_cooccurrence(self, leaves1, leaves2):
        """Compute kernel matrix based on leaf co-occurrence."""
        n1 = leaves1.shape[0]
        n2 = leaves2.shape[0]
        n_trees = leaves1.shape[1]

        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                matches = np.sum(leaves1[i] == leaves2[j])
                K[i, j] = matches / n_trees
        return K


# ──────────────────────────────────────────────
# 8. Tree Kernel Decision Tree
# ──────────────────────────────────────────────

class TreeKernelDecisionTree(BaseEstimator, RegressorMixin):
    """
    Decision tree with built-in RFR-Kernel concept.

    Each leaf has a ridge regression model, and the tree itself
    computes kernel similarity based on leaf co-occurrence.

    Can be used as:
    1. Standalone tree regressor
    2. Kernel for KernelRidge, SVR, etc. (via __call__ method)
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        alpha: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.random_state = random_state

        self.tree_ = None
        self.leaf_models_ = {}
        self._is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Fit base tree
        from sklearn.tree import DecisionTreeRegressor
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_.fit(X, y)

        # Fit ridge models for each leaf
        leaf_indices = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_indices)

        from sklearn.linear_model import Ridge
        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            if np.sum(mask) < 2:
                continue
            X_leaf = X[mask]
            y_leaf = y[mask]
            model = Ridge(alpha=self.alpha)
            model.fit(X_leaf, y_leaf)
            self.leaf_models_[leaf_id] = model

        self._is_fitted = True
        return self

    def __call__(self, X1, X2=None, eval_gradient=False):
        """Compute kernel matrix (RFR-Kernel concept)."""
        if not self._is_fitted:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1

        leaves1 = self.tree_.apply(X1)
        leaves2 = self.tree_.apply(X2) if X2 is not X1 else leaves1

        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                if leaves1[i] == leaves2[j]:
                    K[i, j] = 1.0

        if eval_gradient:
            # No learnable hyperparameters for the kernel matrix
            gradient = np.zeros((n1, n2, 0))
            return K, gradient

        return K

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        leaf_indices = self.tree_.apply(X)
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            leaf_id = leaf_indices[i]
            if leaf_id in self.leaf_models_:
                predictions[i] = self.leaf_models_[leaf_id].predict(X[i].reshape(1, -1))[0]
            else:
                predictions[i] = np.mean(self.tree_.predict(X[i].reshape(1, -1)))

        return predictions

    @property
    def feature_importances_(self):
        if self.tree_ is None:
            raise RuntimeError("Model not fitted")
        return self.tree_.feature_importances_
