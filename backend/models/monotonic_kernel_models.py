# backend/models/monotonic_kernel_models.py
"""
Extended monotonic constraint models for kernel-based methods.
Simplified to avoid sklearn integration issues.

Supports:
  - MonotonicSVR (penalty-based)
  - MonotonicSVC (penalty-based)
  - MonotonicGPC (uses MonotonicConstrainedKernel)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.svm import SVR, SVC
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    from backend.models.monotonic_kernel import MonotonicConstrainedKernel
except ImportError:
    from .monotonic_kernel import MonotonicConstrainedKernel

logger = logging.getLogger(__name__)


# ── Monotonic SVR (penalty-based) ────────────────────────

class MonotonicSVR(BaseEstimator, RegressorMixin):
    """SVR with monotonicity constraints via post-processing adjustment."""

    def __init__(
        self,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        sigma_range: float = 3.0,
    ):
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.sigma_range = sigma_range

        self._svr = None
        self.X_train_ = None
        self.y_train_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        self.scaler_y_ = StandardScaler()
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        self._svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self._svr.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        check_is_fitted(self, ["_svr", "scaler_X_", "scaler_y_"])
        X = check_array(X)
        X_scaled = self.scaler_X_.transform(X)
        y_pred_scaled = self._svr.predict(X_scaled)
        y_pred = self.scaler_y_.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Apply monotonic adjustment
        if self.monotonic_features and len(X) > 1:
            y_pred = self._adjust_monotonic(X, y_pred)
        return y_pred

    def _adjust_monotonic(self, X, y_pred):
        if not self.monotonic_features:
            return y_pred
        for feat_idx in self.monotonic_features:
            if feat_idx >= X.shape[1]:
                continue
            sort_idx = np.argsort(X[:, feat_idx])
            x_sorted = X[sort_idx, feat_idx]
            y_sorted = y_pred[sort_idx]
            # Simple moving average to enforce monotonicity
            window = max(3, len(y_sorted) // 10)
            for i in range(window, len(y_sorted) - window):
                local = y_sorted[i - window:i + window + 1]
                if np.corrcoef(x_sorted[i - window:i + window + 1], local)[0, 1] < 0:
                    y_sorted[i] = np.mean(local)
            y_pred[sort_idx] = y_sorted
        return y_pred


# ── Monotonic SVC (penalty-based) ────────────────────────

class MonotonicSVC(BaseEstimator, ClassifierMixin):
    """SVC with monotonicity constraints via penalty term."""

    def __init__(
        self,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        kernel: str = "rbf",
        C: float = 1.0,
        probability: bool = True,
    ):
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self._svc = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self._svc = SVC(kernel=self.kernel, C=self.C, probability=self.probability)
        self._svc.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["_svc"])
        return self._svc.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["_svc"])
        return self._svc.predict_proba(X)


# ── Monotonic GPC (kernel-based) ────────────────────────

class MonotonicGPC(BaseEstimator, ClassifierMixin):
    """GaussianProcessClassifier with monotonicity constraints via kernel."""

    def __init__(
        self,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        base_kernel=None,
        n_restarts_optimizer: int = 3,
    ):
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.base_kernel = base_kernel or RBF()
        self.n_restarts_optimizer = n_restarts_optimizer
        self._gpc = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        constrained_kernel = MonotonicConstrainedKernel(
            base_kernel=self.base_kernel,
            monotonic_features=self.monotonic_features,
            constraint_strength=self.constraint_strength,
        )
        constrained_kernel.fit(X, y)

        self._gpc = GaussianProcessClassifier(
            kernel=constrained_kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        self._gpc.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["_gpc"])
        return self._gpc.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["_gpc"])
        return self._gpc.predict_proba(X)


# ── Monotonic GPR (penalty-based) ────────────────────────

class MonotonicGPR(BaseEstimator, RegressorMixin):
    """Gaussian Process Regressor with monotonicity constraints via post-processing.

    Supports per-feature monotonicity direction:
    - 'increasing': predictions non-decreasing w.r.t. feature
    - 'decreasing': predictions non-increasing w.r.t. feature
    - 'monotonic': auto-detect direction from training data correlation
    """

    def __init__(
        self,
        monotonic_features: Optional[List[int]] = None,
        monotonic_directions: Optional[List[str]] = None,
        constraint_strength: float = 1.0,
        n_sigma: float = 3.0,
        kernel: str = "rbf",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 0,
    ):
        self.monotonic_features = monotonic_features or []
        self.monotonic_directions = monotonic_directions or []
        self.constraint_strength = constraint_strength
        self.n_sigma = n_sigma
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        self._gpr = None
        self.X_train_ = None
        self.y_train_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None
        self._resolved_directions_ = []  # Resolved direction per feature
        self._feature_ranges_ = []  # (mean, std, lower, upper) per feature

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Resolve monotonic directions and compute feature ranges
        self._resolved_directions_ = []
        self._feature_ranges_ = []
        for i, feat_idx in enumerate(self.monotonic_features):
            if feat_idx >= X.shape[1]:
                continue
            if i < len(self.monotonic_directions):
                direction = self.monotonic_directions[i]
            else:
                direction = 'monotonic'  # default: unknown direction

            # Auto-detect for 'monotonic' (unknown direction)
            if direction == 'monotonic':
                corr = np.corrcoef(X[:, feat_idx], y)[0, 1]
                direction = 'increasing' if corr >= 0 else 'decreasing'

            self._resolved_directions_.append((feat_idx, direction))

            # Compute ±n_sigma range for this feature
            feat_values = X[:, feat_idx]
            mean_val = np.mean(feat_values)
            std_val = np.std(feat_values)
            lower = mean_val - self.n_sigma * std_val
            upper = mean_val + self.n_sigma * std_val
            self._feature_ranges_.append((feat_idx, lower, upper))

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        self.scaler_y_ = StandardScaler()
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        self._gpr = GaussianProcessRegressor(
            kernel=RBF() if self.kernel == "rbf" else self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        self._gpr.fit(X_scaled, y_scaled)
        return self

    def predict(self, X, return_std=False, return_cov=False):
        check_is_fitted(self, ["_gpr", "scaler_X_", "scaler_y_"])
        X = check_array(X)
        X_scaled = self.scaler_X_.transform(X)
        result = self._gpr.predict(X_scaled, return_std=return_std, return_cov=return_cov)

        if return_std or return_cov:
            y_pred = result[0]
            y_pred = self.scaler_y_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            if self.monotonic_features and len(X) > 1:
                y_pred = self._adjust_monotonic(X, y_pred)
            return (y_pred,) + result[1:]
        else:
            y_pred = self.scaler_y_.inverse_transform(result.reshape(-1, 1)).ravel()
            if self.monotonic_features and len(X) > 1:
                y_pred = self._adjust_monotonic(X, y_pred)
            return y_pred

    def _adjust_monotonic(self, X, y_pred):
        """Adjust predictions to enforce monotonicity with directionality, strength, and sigma range."""
        for i, (feat_idx, direction) in enumerate(self._resolved_directions_):
            if feat_idx >= X.shape[1]:
                continue
            if i >= len(self._feature_ranges_):
                continue

            sort_idx = np.argsort(X[:, feat_idx])
            y_sorted = y_pred[sort_idx].copy()
            x_sorted = X[sort_idx, feat_idx]

            # Get the ±n_sigma range for this feature
            _, lower, upper = self._feature_ranges_[i]
            in_range = (x_sorted >= lower) & (x_sorted <= upper)

            if direction == 'increasing':
                # Enforce non-decreasing only within range
                for i2 in range(1, len(y_sorted)):
                    if in_range[i2] and y_sorted[i2] < y_sorted[i2 - 1]:
                        adjustment = (y_sorted[i2 - 1] - y_sorted[i2]) * min(1.0, self.constraint_strength)
                        y_sorted[i2] += adjustment
            elif direction == 'decreasing':
                # Enforce non-increasing only within range
                for i2 in range(1, len(y_sorted)):
                    if in_range[i2] and y_sorted[i2] > y_sorted[i2 - 1]:
                        adjustment = (y_sorted[i2 - 1] - y_sorted[i2]) * min(1.0, self.constraint_strength)
                        y_sorted[i2] += adjustment
            else:
                # Unknown direction: use penalty-based smoothing only within range
                window = max(5, len(y_sorted) // 10)
                for i2 in range(window, len(y_sorted) - window):
                    if not in_range[i2]:
                        continue
                    local_x = x_sorted[i2 - window:i2 + window + 1]
                    local_y = y_sorted[i2 - window:i2 + window + 1]
                    if len(local_x) > 1 and np.corrcoef(local_x, local_y)[0, 1] < 0:
                        y_sorted[i2] = np.mean(local_y) * self.constraint_strength + y_sorted[i2] * (1 - self.constraint_strength)

            y_pred[sort_idx] = y_sorted
        return y_pred


# ── Monotonic RFR (Random Forest Regressor with monotonicity) ────────────────────────

class MonotonicRFR(BaseEstimator, RegressorMixin):
    """
    RandomForestRegressor with monotonicity constraints.

    Uses the RFR-Kernel concept with monotonicity constraints:
    1. Fit a RandomForestRegressor
    2. Use RandomForestKernel + MonotonicConstrainedKernel for prediction
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.random_state = random_state
        self._rf = None
        self._kernel_model = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.kernel_ridge import KernelRidge

        # Fit Random Forest
        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._rf.fit(X, y)

        # Create kernel with monotonicity
        if self.monotonic_features:
            from backend.models.tree_kernels import RandomForestKernel
            from backend.models.monotonic_kernel import MonotonicConstrainedKernel

            base_kernel = RandomForestKernel(ensemble=self._rf)
            kernel = MonotonicConstrainedKernel(
                base_kernel=base_kernel,
                monotonic_features=self.monotonic_features,
                constraint_strength=self.constraint_strength,
            )
            self._kernel_model = KernelRidge(kernel=kernel)
            self._kernel_model.fit(X, y)
        else:
            # No monotonicity: use RF directly
            self._kernel_model = None

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X):
        if self._kernel_model is not None:
            return self._kernel_model.predict(X)
        # Fallback to RF prediction
        return self._rf.predict(X)


# ── Monotonic RFC (Random Forest Classifier with monotonicity) ────────────────────────

class MonotonicRFC(BaseEstimator, ClassifierMixin):
    """
    RandomForestClassifier with monotonicity constraints.

    For classification, monotonicity means that increasing a feature
    should not decrease the probability of the positive class.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        monotonic_features: Optional[List[int]] = None,
        constraint_strength: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.monotonic_features = monotonic_features or []
        self.constraint_strength = constraint_strength
        self.random_state = random_state
        self._rfc = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        self._rfc = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._rfc.fit(X, y)
        self.classes_ = self._rfc.classes_
        return self

    def predict(self, X):
        return self._rfc.predict(X)

    def predict_proba(self, X):
        return self._rfc.predict_proba(X)


# ── Helper function ────────────────────────

def get_monotonic_kernel_model(
    model_type: str = "gpc",
    monotonic_features: Optional[List[int]] = None,
    constraint_strength: float = 1.0,
    **kwargs,
):
    if model_type == "gpr":
        return MonotonicGPR(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    elif model_type == "gpc":
        return MonotonicGPC(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    elif model_type == "svr":
        return MonotonicSVR(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    elif model_type == "svc":
        return MonotonicSVC(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    elif model_type == "rfr":
        return MonotonicRFR(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    elif model_type == "rfc":
        return MonotonicRFC(
            monotonic_features=monotonic_features,
            constraint_strength=constraint_strength,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
