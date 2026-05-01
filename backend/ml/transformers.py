"""
Custom Transformers - chemai2/backend/ml/transformers.py
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log(np.maximum(X, 1e-9))

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log1p(np.maximum(X, 0))

class ExpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.exp(X)

class AdaptiveQuantileTransformer(_QuantileTransformer):
    """n_quantiles を n_samples に自動調整する QuantileTransformer"""
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        if self.n_quantiles > n_samples:
            self.n_quantiles = max(2, n_samples)
        return super().fit(X, y)
