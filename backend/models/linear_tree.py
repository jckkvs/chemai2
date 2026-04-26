# backend/models/linear_tree.py — 精緻化版 (線形決定木コア)

from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import _tree

logger = logging.getLogger(__name__)


class LinearTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree with linear models at leaves, with numerical stability enhancements
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_improvement_ratio: float = 0.01,  # 【修正点3】早期終了閾値
        linear_alpha: float = 1.0,  # Ridge regularization for leaf models
        check_collinearity: bool = True,  # 【修正点1】多重共線性チェック
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement_ratio = min_improvement_ratio
        self.linear_alpha = linear_alpha
        self.check_collinearity = check_collinearity
        self.random_state = random_state
        self.verbose = verbose
        
        self._tree = None
        self._feature_importances_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> 'LinearTreeRegressor':
        """Fit linear tree with stability checks"""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        
        n_samples, n_features = X_arr.shape
        
        if n_samples < self.min_samples_split:
            logger.warning(f"Too few samples ({n_samples}) for splitting. Fitting constant model.")
            self._tree = _ConstantLeafModel(y_arr.mean())
            self._feature_importances_ = np.zeros(n_features)
            return self
        
        rng = np.random.default_rng(self.random_state)
        self._rng = rng
        
        self._tree = self._build_tree(
            X_arr, y_arr, 
            depth=0, 
            feature_indices=np.arange(n_features)
        )
        
        # 【修正点4】重要度計算（複数ラン平均で再現性確保）
        self._feature_importances_ = self._compute_importances_stable(
            X_arr, y_arr, n_runs=3
        )
        
        return self
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        feature_indices: np.ndarray,
        parent_impurity: Optional[float] = None
    ):
        """Recursive tree construction with early stopping"""
        n_samples = len(y)
        
        # 【修正点3】早期終了条件: 改善率閾値チェック
        if parent_impurity is not None and depth > 0:
            current_impurity = np.var(y)
            if current_impurity > parent_impurity * (1 - self.min_improvement_ratio):
                return _LinearLeafModel(X, y, alpha=self.linear_alpha, 
                                       check_collinearity=self.check_collinearity)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            np.var(y) < 1e-10):
            return _LinearLeafModel(X, y, alpha=self.linear_alpha,
                                   check_collinearity=self.check_collinearity)
        
        best_gain = -np.inf
        best_split = None
        
        for feat_idx in feature_indices:
            thresholds = self._get_candidate_thresholds(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask
                
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                impurity_parent = self._stable_variance(y)
                impurity_left = self._stable_variance(y[left_mask])
                impurity_right = self._stable_variance(y[right_mask])
                
                gain = (impurity_parent - 
                       (n_left/n_samples)*impurity_left - 
                       (n_right/n_samples)*impurity_right)
                
                if gain > best_gain:
                    best_gain, best_split = gain, (feat_idx, thresh, left_mask, right_mask)
        
        if best_split is None or best_gain < 1e-6:
            return _LinearLeafModel(X, y, alpha=self.linear_alpha,
                                   check_collinearity=self.check_collinearity)
        
        feat_idx, thresh, left_mask, right_mask = best_split
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices, parent_impurity=self._stable_variance(y[left_mask]))
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices, parent_impurity=self._stable_variance(y[right_mask]))
        
        return _SplitNode(feat_idx, thresh, left_child, right_child)
    
    def _stable_variance(self, y: np.ndarray) -> float:
        """Compute variance with numerical stability (Welford's algorithm)"""
        if len(y) < 2: return 0.0
        mean, M2 = 0.0, 0.0
        for i, val in enumerate(y, 1):
            delta = val - mean
            mean += delta / i
            delta2 = val - mean
            M2 += delta * delta2
        return M2 / (len(y) - 1) if len(y) > 1 else 0.0
    
    def _get_candidate_thresholds(self, values: np.ndarray, max_candidates: int = 50) -> np.ndarray:
        unique_vals = np.unique(values)
        if len(unique_vals) <= 1: return np.array([])
        if len(unique_vals) <= max_candidates:
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        else:
            quantiles = np.linspace(0, 1, max_candidates + 2)[1:-1]
            thresholds = np.percentile(values, quantiles * 100)
        return thresholds
    
    def _compute_importances_stable(self, X: np.ndarray, y: np.ndarray, n_runs: int = 3) -> np.ndarray:
        importances_sum = np.zeros(X.shape[1])
        for run in range(n_runs):
            run_seed = self.random_state + run if self.random_state is not None else None
            rng = np.random.default_rng(run_seed)
            importances_sum += self._compute_importances_single(X, y, rng)
        importances = importances_sum / n_runs
        total = np.sum(importances)
        if total > 0: importances /= total
        return importances
    
    def _compute_importances_single(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        importances = np.zeros(X.shape[1])
        def traverse(node, X_sub, y_sub):
            if isinstance(node, _SplitNode):
                left_mask = X_sub[:, node.feature_idx] <= node.threshold
                right_mask = ~left_mask
                imp_p, imp_l, imp_r = np.var(y_sub), np.var(y_sub[left_mask]) if np.any(left_mask) else 0, np.var(y_sub[right_mask]) if np.any(right_mask) else 0
                n, n_l, n_r = len(y_sub), np.sum(left_mask), np.sum(right_mask)
                gain = imp_p - (n_l/n)*imp_l - (n_r/n)*imp_r
                importances[node.feature_idx] += max(0, gain)
                if np.any(left_mask): traverse(node.left, X_sub[left_mask], y_sub[left_mask])
                if np.any(right_mask): traverse(node.right, X_sub[right_mask], y_sub[right_mask])
        traverse(self._tree, X, y)
        return importances
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array([self._tree.predict(x) for x in X_arr])
    
    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances_ is None: raise RuntimeError("Model not fitted")
        return self._feature_importances_


class _LinearLeafModel:
    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0, check_collinearity: bool = True):
        # 【修正点1】多重共線性チェックと自動正則化
        if check_collinearity and X.shape[1] > 1 and len(X) > X.shape[1]:
            try:
                cond_num = np.linalg.cond(X)
                if cond_num > 1e10: alpha = max(alpha, 10.0)
            except np.linalg.LinAlgError: pass
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.model.fit(X, y)
    def predict(self, x: np.ndarray) -> float: return float(self.model.predict([x])[0])

class _SplitNode:
    def __init__(self, feature_idx: int, threshold: float, left, right):
        self.feature_idx, self.threshold, self.left, self.right = feature_idx, threshold, left, right
    def predict(self, x: np.ndarray) -> float:
        return self.left.predict(x) if x[self.feature_idx] <= self.threshold else self.right.predict(x)

class _ConstantLeafModel:
    def __init__(self, value: float): self.value = value
    def predict(self, x: np.ndarray) -> float: return self.value
