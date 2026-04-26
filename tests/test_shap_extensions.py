# tests/test_shap_extensions.py — SHAP拡張機能テスト

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from backend.interpret.shap_extensions import (
    compute_shap_batch,
    adjust_shap_for_correlation,
    compute_shap_confidence_intervals
)

@pytest.fixture
def regression_data():
    X = pd.DataFrame(np.random.randn(50, 4), columns=['f0', 'f1', 'f2', 'f3'])
    y = X['f0'] * 2 + np.random.randn(50)
    model = RandomForestRegressor(n_estimators=5, max_depth=3).fit(X, y)
    return model, X, y

class TestComputeShapBatch:
    def test_returns_dataframe(self, regression_data):
        model, X, y = regression_data
        result = compute_shap_batch(model, X, batch_size=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

class TestAdjustShapForCorrelation:
    def test_equal_split_adjustment(self):
        X = pd.DataFrame(np.random.randn(20, 2), columns=['f0', 'f1'])
        X['f1'] = X['f0'] * 0.99 # Highly correlated
        shap_values = pd.DataFrame({'f0': [1.0]*20, 'f1': [0.0]*20})
        adjusted = adjust_shap_for_correlation(shap_values, X, correlation_threshold=0.8)
        assert np.allclose(adjusted['f0'], 0.5)
        assert np.allclose(adjusted['f1'], 0.5)

class TestComputeShapConfidenceIntervals:
    def test_returns_expected_structure(self, regression_data):
        model, X, y = regression_data
        result = compute_shap_confidence_intervals(model, X, n_bootstrap=3)
        assert 'f0' in result
        assert 'mean' in result['f0']
        assert 'std' in result['f0']
