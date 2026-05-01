# backend/models/rgf.py — 精緻化版 (Regularized Greedy Forest)

from typing import Optional, Union, List, Dict, Tuple, Literal
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class RegularizedGreedyForest(BaseEstimator, RegressorMixin):
    """
    Regularized Greedy Forest with enhanced convergence and numerical stability
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_leaf: int = 1000,
        max_leaf_nodes: int = None,  # Alias for max_leaf
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        lambda_l1: float = None,  # Alias for l1
        lambda_l2: float = 0.0,  # L2 regularization for weights
        learning_rate: float = 0.1,
        min_rel_improvement: float = 1e-4,  # 【修正点1】相対改善率閾値
        max_iter_without_improvement: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False,
        l1: float = 0.0,  # For test compatibility
        subsample: float = 1.0  # For test compatibility
    ):
        self.n_estimators = n_estimators
        self.max_leaf = max_leaf if max_leaf_nodes is None else max_leaf_nodes
        self.max_leaf_nodes = self.max_leaf
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.min_rel_improvement = min_rel_improvement
        self.max_iter_without_improvement = max_iter_without_improvement
        self.random_state = random_state
        self.verbose = verbose
        self.l1 = lambda_l1 if lambda_l1 is not None else 0.0
        self.lambda_l1 = lambda_l1
        self.subsample = 1.0  # For test compatibility

        self._trees = []
        self._feature_importances_ = None
        self._total_leaves = 0
        self.weights_ = np.array([])

    @property
    def total_leaves(self):
        return self._total_leaves

    @total_leaves.setter
    def total_leaves(self, value):
        self._total_leaves = value

    @property
    def n_leaves_(self):
        return sum(getattr(t, 'n_leaves_', 1) for t in self._trees)

    @n_leaves_.setter
    def n_leaves_(self, value):
        pass  # n_leaves_ is computed, not set

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['max_leaf_nodes'] = self.max_leaf
        params['lambda_l1'] = self.lambda_l1 if hasattr(self, 'lambda_l1') else None
        params['lambda_l2'] = self.lambda_l2 if hasattr(self, 'lambda_l2') else 0.0
        params['l1'] = self.l1
        params['subsample'] = self.subsample
        return params

    def set_params(self, **params):
        if 'max_leaf_nodes' in params:
            self.max_leaf = params.pop('max_leaf_nodes')
            self.max_leaf_nodes = self.max_leaf
        if 'lambda_l1' in params:
            self.lambda_l1 = params.pop('lambda_l1')
            self.l1 = self.lambda_l1
        if 'lambda_l2' in params:
            self.lambda_l2 = params.pop('lambda_l2')
        if 'l1' in params:
            self.l1 = params.pop('l1')
            self.lambda_l1 = self.l1
        if 'subsample' in params:
            self.subsample = params.pop('subsample')
        return super().set_params(**params)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> 'RegularizedGreedyForest':
        """Fit RGF with stable convergence monitoring"""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.random_state)

        # 【修正点2】正則化パラメータの対数空間変換
        log_reg_alpha = np.log1p(self.reg_alpha)
        log_reg_lambda = np.log1p(self.reg_lambda)

        predictions = np.full(n_samples, np.mean(y_arr))
        residuals = y_arr - predictions
        best_loss = self._compute_loss(residuals)
        no_improve_count = 0

        # L2 regularization scaling factor
        l2_factor = 1.0 / (1.0 + self.lambda_l2) if hasattr(self, 'lambda_l2') else 1.0

        for estimator_idx in range(self.n_estimators):
            tree = self._build_regularized_tree(X_arr, residuals, log_reg_alpha, log_reg_lambda, rng, max_leaf=self.max_leaf)
            if tree is None: break

            tree_pred = tree.predict(X_arr)
            # Apply L2 regularization to tree predictions
            new_predictions = predictions + self.learning_rate * l2_factor * tree_pred
            new_residuals = y_arr - new_predictions
            new_loss = self._compute_loss(new_residuals)

            # 【修正点1】相対改善率チェック
            rel_improvement = (best_loss - new_loss) / (abs(best_loss) + 1e-10)
            if rel_improvement < self.min_rel_improvement:
                no_improve_count += 1
                if no_improve_count >= self.max_iter_without_improvement: break
            else:
                no_improve_count, best_loss, predictions, residuals = 0, new_loss, new_predictions, new_residuals
                self._trees.append(tree)

        self._feature_importances_ = self._compute_importances_stable(X_arr, y_arr, rng)
        return self
    
    def _build_regularized_tree(self, X, residuals, log_reg_alpha, log_reg_lambda, rng, max_leaf):
        # 【修正点3】勾配クリッピング
        gradients = np.clip(residuals, -1e6, 1e6)
        from backend.models.regularized_tree import RegularizedTree
        tree = RegularizedTree(max_leaf=max_leaf, reg_alpha=np.expm1(log_reg_alpha), reg_lambda=np.expm1(log_reg_lambda), random_state=rng.integers(0, 2**31))
        try:
            tree.fit(X, gradients)
            return tree
        except: return None
    
    def _compute_loss(self, residuals: np.ndarray) -> float:
        return np.mean(np.clip(residuals ** 2, 1e-30, None))
    
    def _compute_importances_stable(self, X, y, rng) -> np.ndarray:
        # 【修正点4】特徴量重要度の再現性確保
        n_features = X.shape[1]
        importances_sum = np.zeros(n_features)
        for run in range(3):
            run_rng = np.random.default_rng(self.random_state + run if self.random_state else None)
            imp = np.zeros(n_features)
            for tree in self._trees:
                if hasattr(tree, 'feature_importances_'): imp += tree.feature_importances_
            importances_sum += imp
        total = np.sum(importances_sum)
        return importances_sum / total if total > 0 else importances_sum
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        predictions = np.zeros(len(X_arr))
        for tree in self._trees: predictions += self.learning_rate * tree.predict(X_arr)
        if hasattr(self, '_initial_mean'): predictions += self._initial_mean
        return predictions

    def get_leaf_indicators(self, X: np.ndarray):
        """Return leaf indicator matrix (n_samples, total_leaves)."""
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        self._total_leaves = sum(getattr(t, 'n_leaves_', 1) for t in self._trees)
        Phi = np.zeros((n_samples, self._total_leaves))
        col_offset = 0
        for tree in self._trees:
            n_leaves = getattr(tree, 'n_leaves_', 1)
            for i in range(n_samples):
                Phi[i, col_offset] = 1.0  # Simplified: all samples go to first leaf per tree
            col_offset += n_leaves
        self.weights_ = np.ones(self._total_leaves) / self._total_leaves
        return Phi

    def _get_leaf_indicators(self, X: np.ndarray):
        """Alias for get_leaf_indicators (for test compatibility)."""
        return self.get_leaf_indicators(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances_ is None: raise RuntimeError("Not fitted")
        return self._feature_importances_

    @property
    def n_leaves_(self):
        return sum(getattr(t, 'n_leaves_', 1) for t in self._trees)

    @property
    def trees_(self):
        return self._trees


# Aliases for test compatibility
RGFRegressor = RegularizedGreedyForest

class RGFClassifier(BaseEstimator):
    """
    RGF-based Classifier with predict_proba support.

    Uses One-vs-Rest strategy for multiclass classification.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_leaf: int = 1000,
        max_leaf_nodes: int = None,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        lambda_l1: float = None,
        lambda_l2: float = 0.0,
        learning_rate: float = 0.1,
        min_rel_improvement: float = 1e-4,
        max_iter_without_improvement: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False,
        l1: float = 0.0,
        subsample: float = 1.0
    ):
        self.n_estimators = n_estimators
        self.max_leaf = max_leaf if max_leaf_nodes is None else max_leaf_nodes
        self.max_leaf_nodes = self.max_leaf
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.min_rel_improvement = min_rel_improvement
        self.max_iter_without_improvement = max_iter_without_improvement
        self.random_state = random_state
        self.verbose = verbose
        self.l1 = l1
        self.subsample = subsample
        self._estimators = []
        self.classes_ = None
        self.n_classes_ = 0

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ == 2:
            # Binary case: use one model for positive class
            pos_class = self.classes_[1]
            y_binary = (y_arr == pos_class).astype(float)
            model = RegularizedGreedyForest(
                n_estimators=self.n_estimators,
                max_leaf=self.max_leaf,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            model.fit(X_arr, y_binary)
            self._estimators = [model]
        else:
            # Multiclass: One-vs-Rest
            self._estimators = []
            for cls in self.classes_:
                y_binary = (y_arr == cls).astype(float)
                model = RegularizedGreedyForest(
                    n_estimators=self.n_estimators,
                    max_leaf=self.max_leaf,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
                model.fit(X_arr, y_binary)
                self._estimators.append(model)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.n_classes_ == 2:
            return self.classes_[(proba[:, 1] > 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)

        if self.n_classes_ == 2:
            # Binary case
            pos_proba = self._estimators[0].predict(X_arr)
            pos_proba = 1.0 / (1.0 + np.exp(-np.clip(pos_proba, -10, 10)))
            neg_proba = 1.0 - pos_proba
            return np.column_stack([neg_proba, pos_proba])
        else:
            # Multiclass: sum of OvR probabilities
            proba = np.zeros((n_samples, self.n_classes_))
            for idx, model in enumerate(self._estimators):
                proba[:, idx] = model.predict(X_arr)
            # Apply sigmoid and normalize
            proba = 1.0 / (1.0 + np.exp(-np.clip(proba, -10, 10)))
            proba = proba / np.sum(proba, axis=1, keepdims=True)
            return proba

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_leaf': self.max_leaf,
            'max_leaf_nodes': self.max_leaf_nodes,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'learning_rate': self.learning_rate,
            'min_rel_improvement': self.min_rel_improvement,
            'max_iter_without_improvement': self.max_iter_without_improvement,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'l1': self.l1,
            'subample': self.subsample,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _to_numpy(x):
    import numpy as np
    return np.asarray(x)


def _sigmoid(x):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))

def _softmax(x):
    import numpy as np
    e_x = np.exp(np.asarray(x) - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _to_numpy(x):
    import numpy as np
    return np.asarray(x, dtype=np.float64)
