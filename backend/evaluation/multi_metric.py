# backend/evaluation/multi_metric.py — guikit-learn 互換の多指標評価

"""
Multi-metric evaluation utilities for ChemAI ML Studio
Provides:
- Batch computation of multiple sklearn metrics
- Chemistry-specific metrics (chemical validity, interpretability)
- Metric correlation analysis and trade-off visualization
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Union, Literal, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

logger = logging.getLogger(__name__)

METRIC_SETS = {
    'regression_basic': {'r2': r2_score, 'mae': mean_absolute_error, 'rmse': lambda t, p: np.sqrt(mean_squared_error(t, p))},
    'classification_basic': {'accuracy': accuracy_score, 'f1': f1_score, 'roc_auc': roc_auc_score},
}

def compute_metrics(y_true, y_pred, y_proba=None, metric_set='regression_basic') -> Dict[str, float]:
    metrics = METRIC_SETS.get(metric_set, METRIC_SETS['regression_basic'])
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    results = {}
    for name, func in metrics.items():
        try:
            if 'roc_auc' in name and y_proba is not None:
                results[name] = func(yt, y_proba if y_proba.ndim == 1 else y_proba[:, 1])
            else: results[name] = func(yt, yp)
        except: results[name] = np.nan
    return results

def compute_chemistry_specific_metrics(y_true, y_pred, property_name=None) -> Dict[str, float]:
    results = {}
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if property_name:
        ranges = {'logp': (-10, 10), 'solubility': (-15, 5)}
        if property_name.lower() in ranges:
            mi, ma = ranges[property_name.lower()]
            results[f'{property_name}_validity'] = float(((yp >= mi) & (yp <= ma)).mean())
    if len(yt) >= 10:
        from scipy import stats
        corr, _ = stats.spearmanr(yt, yp)
        results['spearman_corr'] = float(corr)
    return results

def analyze_metric_correlations(metric_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if len(metric_results) < 3: return {'warning': 'Insufficient data'}
    df = pd.DataFrame(metric_results).T.dropna(axis=1)
    if df.shape[1] < 2: return {'warning': 'Insufficient metrics'}
    return {'correlation_matrix': df.corr().abs().to_dict()}
