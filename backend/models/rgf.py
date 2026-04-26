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
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        learning_rate: float = 0.1,
        min_rel_improvement: float = 1e-4,  # 【修正点1】相対改善率閾値
        max_iter_without_improvement: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_estimators = n_estimators
        self.max_leaf = max_leaf
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.min_rel_improvement = min_rel_improvement
        self.max_iter_without_improvement = max_iter_without_improvement
        self.random_state = random_state
        self.verbose = verbose
        
        self._trees = []
        self._feature_importances_ = None
    
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
        
        for estimator_idx in range(self.n_estimators):
            tree = self._build_regularized_tree(X_arr, residuals, log_reg_alpha, log_reg_lambda, rng, max_leaf=self.max_leaf)
            if tree is None: break
            
            tree_pred = tree.predict(X_arr)
            new_predictions = predictions + self.learning_rate * tree_pred
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
    
    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances_ is None: raise RuntimeError("Not fitted")
        return self._feature_importances_
