# backend/models/monotonic_kernel.py — 精緻化版 (制約カーネル実装)

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel, WhiteKernel

logger = logging.getLogger(__name__)

class MonotonicConstrainedKernel(Kernel):
    """
    Gaussian Process kernel with monotonicity constraints via kernel modification.
    Inherits from sklearn.gaussian_process.kernels.Kernel so it can be
    used directly with GaussianProcessRegressor.

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

    def is_stationary(self):
        """Return whether the kernel is stationary (it is not, due to monotonic penalty)."""
        return False

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        diag_base = self.base_kernel.diag(X) if hasattr(self.base_kernel, 'diag') else np.ones(X.shape[0])
        if not self.monotonic_features:
            return diag_base
        # Monotonic penalty is 1 for diagonal (x_i same as x_i)
        return diag_base
    
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
        # Handle 1D arrays (single sample) from sklearn's pairwise_kernels
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y is None:
            Y = X
            is_self = True
        else:
            Y = np.asarray(Y, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)
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


class ConstrainedEstimatorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper that applies monotonicity constraints to an estimator.
    For supported estimators (GBM, RF), set monotonic_constraints param.
    For others, falls back to unconstrained.
    """
    def __init__(self, estimator, column_meta: dict):
        self.estimator = estimator
        self.column_meta = column_meta

    def fit(self, X, y):
        # Apply monotonic constraints if estimator supports it
        constraints = {}
        for col, meta in self.column_meta.items():
            if hasattr(meta, 'monotonic') and meta.monotonic != 0:
                idx = list(X.columns).index(col) if hasattr(X, 'columns') else int(col)
                constraints[idx] = meta.monotonic
        if constraints and hasattr(self.estimator, 'set_params'):
            try:
                self.estimator.set_params(monotonic_constraints=constraints)
            except Exception:
                logger.warning("Estimator does not support monotonic_constraints")
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def get_params(self, deep=True):
        return {"estimator": self.estimator, "column_meta": self.column_meta}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)


class MonotonicKernelWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper that applies monotonicity constraints to any estimator.

    Uses penalty sample expansion method (similar to MonotonicConstraintRegressor).
    Works with SVR, KernelRidge, GaussianProcessRegressor, etc.
    """

    def __init__(self, base_estimator=None, monotonic_constraints=None, constraint_strength=1.0,
                 penalty_weight=10.0, max_iter=5, n_grid=20, sigma_factor=3.0):
        self.base_estimator = base_estimator
        self.monotonic_constraints = monotonic_constraints
        self.constraint_strength = constraint_strength
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.n_grid = n_grid
        self.sigma_factor = sigma_factor
        self.estimator_ = None
        self.monotonic_violation_ = 0.0

    def fit(self, X, y):
        X_aug = np.asarray(X, dtype=np.float64)
        y_aug = np.asarray(y, dtype=np.float64).ravel()

        # Initial fit
        current_estimator = clone(self.base_estimator) if self.base_estimator else None
        if current_estimator is None:
            from sklearn.svm import SVR
            current_estimator = SVR()

        current_estimator.fit(X_aug, y_aug)
        self.estimator_ = current_estimator

        # Compute monotonicity violation
        self.monotonic_violation_ = self._compute_violation(X_aug, y_aug)

        # Apply penalty samples for monotonicity (simplified version)
        if self.monotonic_constraints is not None:
            for iteration in range(self.max_iter):
                X_pen, y_pen, w_pen = self._generate_penalty_samples(X_aug, y_aug)
                if X_pen is None:
                    break
                X_aug = np.vstack([X_aug, X_pen])
                y_aug = np.concatenate([y_aug, y_pen])
                new_estimator = clone(self.base_estimator) if self.base_estimator else SVR()
                new_estimator.fit(X_aug, y_aug)
                self.estimator_ = new_estimator
                new_violation = self._compute_violation(X_aug, y_aug)
                if new_violation >= self.monotonic_violation_:
                    break
                self.monotonic_violation_ = new_violation

        return self

    def _generate_penalty_samples(self, X, y):
        """Generate penalty samples for monotonicity constraints."""
        if self.monotonic_constraints is None:
            return None, None, None

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        X_pen_list = []
        y_pen_list = []

        for feat_idx, constraint in enumerate(self.monotonic_constraints):
            if constraint == 0 or feat_idx >= n_features:
                continue

            # Create grid points for this feature
            x_col = X_arr[:, feat_idx]
            mean_val = np.mean(x_col)
            std_val = np.std(x_col) + 1e-10
            lo = mean_val - self.sigma_factor * std_val
            hi = mean_val + self.sigma_factor * std_val
            grid = np.linspace(lo, hi, self.n_grid)

            # Fix other features at their median
            x_median = np.median(X_arr, axis=0)

            for i in range(len(grid) - 1):
                x0 = x_median.copy()
                x0[feat_idx] = grid[i]
                x1 = x_median.copy()
                x1[feat_idx] = grid[i + 1]

                # Predict at these points
                pred0 = self.estimator_.predict([x0])[0]
                pred1 = self.estimator_.predict([x1])[0]

                # Check violation
                if constraint > 0 and pred1 < pred0:  # Should increase
                    X_pen_list.extend([x0, x1])
                    y_pen_list.extend([pred0, pred0 + abs(pred0 - pred1) + 0.01])
                elif constraint < 0 and pred1 > pred0:  # Should decrease
                    X_pen_list.extend([x0, x1])
                    y_pen_list.extend([pred0, pred0 - abs(pred0 - pred1) - 0.01])

        if not X_pen_list:
            return None, None, None

        return np.array(X_pen_list), np.array(y_pen_list), np.full(len(y_pen_list), self.penalty_weight)

    def _compute_violation(self, X, y):
        """Compute monotonicity violation score."""
        if self.monotonic_constraints is None:
            return 0.0
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        violation_count = 0
        total_count = 0
        for feat_idx, constraint in enumerate(self.monotonic_constraints):
            if constraint == 0 or feat_idx >= X_arr.shape[1]:
                continue
            x_col = X_arr[:, feat_idx]
            sorted_idx = np.argsort(x_col)
            y_sorted = y_arr[sorted_idx]
            diffs = np.diff(y_sorted)
            if constraint > 0:
                violations = np.sum(diffs < 0)
            else:
                violations = np.sum(diffs > 0)
            violation_count += violations
            total_count += len(diffs)
        return violation_count / max(1, total_count)

    def predict(self, X):
        if self.estimator_ is None:
            raise RuntimeError("Model not fitted")
        return self.estimator_.predict(X)


def wrap_with_soft_monotonic(estimator, constraints):
    """
    Wrap estimator with MonotonicKernelWrapper if constraints are non-zero.
    Returns original estimator if all constraints are 0.
    """
    if constraints is None:
        return estimator
    # Check if any constraint is non-zero
    if hasattr(constraints, '__iter__'):
        if all(v == 0 for v in constraints):
            return estimator
    # Wrap with MonotonicKernelWrapper
    return MonotonicKernelWrapper(
        base_kernel=estimator,
        monotonic_constraints=constraints,
        max_iter=5
    )


def is_soft_monotonic_candidate(estimator):
    """
    Check if estimator is a candidate for soft monotonic constraints.
    Returns True for kernel-based models (SVR, SVC, KernelRidge, GaussianProcessRegressor).
    """
    from sklearn.svm import SVR, SVC
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import Ridge
    
    soft_models = (SVR, SVC, KernelRidge, GaussianProcessRegressor)
    return isinstance(estimator, soft_models)
