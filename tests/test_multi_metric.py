# tests/test_multi_metric.py — 多指標評価テスト

import numpy as np
import pandas as pd
import pytest
from backend.evaluation.multi_metric import (
    compute_metrics,
    compute_chemistry_specific_metrics,
    analyze_metric_correlations
)

class TestComputeMetrics:
    def test_regression_metrics(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 1.9, 3.2]
        results = compute_metrics(y_true, y_pred, metric_set='regression_basic')
        assert 'r2' in results
        assert 'mae' in results
        assert 'rmse' in results

class TestChemistryMetrics:
    def test_property_validity(self):
        y_true = [1.0, 2.0]
        y_pred = [1.5, 15.0] # 15 is out of logp range (-10, 10)
        results = compute_chemistry_specific_metrics(y_true, y_pred, property_name='logp')
        assert 'logp_validity' in results
        assert results['logp_validity'] == 0.5

class TestCorrelationAnalysis:
    def test_returns_matrix(self):
        results = {
            'm1': {'r2': 0.9, 'mae': 0.1},
            'm2': {'r2': 0.8, 'mae': 0.2},
            'm3': {'r2': 0.85, 'mae': 0.15}
        }
        analysis = analyze_metric_correlations(results)
        assert 'correlation_matrix' in analysis
        assert 'r2' in analysis['correlation_matrix']
