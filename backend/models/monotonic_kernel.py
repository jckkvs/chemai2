# backend/models/monotonic_kernel.py — 精緻化版 (制約カーネル実装)

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

logger = logging.getLogger(__name__)

class MonotonicConstrainedKernel(BaseEstimator):
    """
    Gaussian Process kernel with monotonicity constraints via kernel modification
    
    Mathematical formulation:
    K_mono(x, x') = K(x, x') * exp(-λ * max(0, f'(x))^2)
    where f'(x) is estimated via finite differences on training points.
    """
    
    def __init__(
        self,
        base_kernel=None,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        regularization: float = 1e-6,
        verbose: bool = False
    ):
        self.base_kernel = base_kernel or RBF()
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.regularization = regularization
        self.verbose = verbose
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MonotonicConstrainedKernel':
        """Validate constraints and prepare internal matrices"""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        self.n_features = self.X_train.shape[1]
        
        # 【修正点4】制約インデックスの検証
        valid_idx = [i for i in self.monotonic_features if 0 <= i < self.n_features]
        invalid = [i for i in self.monotonic_features if i not in valid_idx]
        if invalid:
            logger.warning(f"Ignoring invalid feature indices for monotonicity: {invalid}")
            self.monotonic_features = valid_idx
        
        if not self.monotonic_features:
            if self.verbose:
                logger.info("No valid monotonic features. Using base kernel.")
            return self
        
        # 【修正点4】事前検証: 訓練データ内の単調性チェック
        self._validate_training_monotonicity()
        
        return self
    
    def _validate_training_monotonicity(self):
        """Check if training data respects monotonicity constraints"""
        for feat_idx in self.monotonic_features:
            x_col = self.X_train[:, feat_idx]
            sorted_idx = np.argsort(x_col)
            y_sorted = self.y_train[sorted_idx]
            
            # 単調増加を仮定（減少の場合は符号反転で同等）
            diffs = np.diff(y_sorted)
            violations = np.sum(diffs < -1e-6)
            violation_ratio = violations / max(1, len(diffs))
            
            if violation_ratio > 0.3:
                logger.warning(
                    f"High monotonicity violation in training data for feature {feat_idx}: "
                    f"{violation_ratio:.1%}. Constraints may be hard to satisfy."
                )
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, eval_gradient: bool = False):
        """Compute kernel matrix with monotonic penalty"""
        X = np.asarray(X, dtype=np.float64)
        if Y is None:
            Y = X
            is_self = True
        else:
            Y = np.asarray(Y, dtype=np.float64)
            is_self = False
        
        # 基本カーネル
        K_base = self.base_kernel(X, Y)
        
        if not self.monotonic_features:
            if eval_gradient:
                return K_base, []
            return K_base
        
        # 【修正点1】数値安定性のため制約ペナルティを指数関数的に適用
        # 有限差分で勾配を推定（訓練データのみ）
        penalty_matrix = np.ones_like(K_base)
        
        for feat_idx in self.monotonic_features:
            grad_X = self._estimate_gradient(X[:, feat_idx], self.y_train)
            grad_Y = self._estimate_gradient(Y[:, feat_idx], self.y_train) if Y is not X else grad_X
            
            # 【修正点2】勾配のクリッピングと安定化
            grad_X = np.clip(grad_X, -10.0, 10.0)
            grad_Y = np.clip(grad_Y, -10.0, 10.0)
            
            # 正の勾配に対するペナルティ（単調増加制約）
            interaction = grad_X[:, None] * grad_Y[None, :]
            penalty = np.exp(-self.constraint_strength * np.maximum(0, interaction))
            penalty_matrix *= penalty
        
        K_constrained = K_base * penalty_matrix
        
        # 【修正点1】対角に正則化追加（数値安定性）
        if is_self:
            K_constrained += self.regularization * np.eye(len(X))
        
        if eval_gradient:
            return K_constrained, []
        return K_constrained
    
    def _estimate_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate local gradient via moving average finite differences
        """
        n = len(x)
        if n < 2:
            return np.zeros(n)
        
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        
        # 【修正点3】ゼロ除算防止
        dx_safe = np.where(np.abs(dx) < 1e-10, 1e-10, dx)
        grads_sorted = dy / dx_safe
        
        # 境界は隣接値で補完
        grads = np.empty(n)
        grads[sorted_idx] = np.concatenate([
            [grads_sorted[0]], 
            (grads_sorted[:-1] + grads_sorted[1:]) / 2.0, 
            [grads_sorted[-1]]
        ])
        
        return grads
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Wrap GPR predict with constraint-aware kernel"""
        K_star = self(X, self.X_train)
        
        try:
            K_train = self(self.X_train)
            alpha = np.linalg.solve(K_train, self.y_train)
        except np.linalg.LinAlgError:
            logger.warning("Kernel matrix singular. Adding regularization.")
            K_train += 1e-4 * np.eye(len(K_train))
            alpha = np.linalg.solve(K_train, self.y_train)
        
        return K_star @ alpha
