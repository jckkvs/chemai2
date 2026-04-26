# backend/interpret/shap_extensions.py — guikit-learn 互換のSHAP拡張

"""
SHAP integration extensions for ChemAI ML Studio
Provides:
- Memory-efficient batch processing for large datasets
- Partial result return on failure (graceful degradation)
- Correlation bias adjustment for feature importance
- Confidence interval estimation via bootstrapping
"""

from __future__ import annotations
import logging
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple, Literal
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def compute_shap_batch(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    background_size: int = 100,
    batch_size: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    return_dataframe: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """Compute SHAP values with memory-efficient batching"""
    import psutil
    if isinstance(X, pd.DataFrame):
        X_arr, feature_names = X.values, list(X.columns) if feature_names is None else feature_names
    else:
        X_arr = np.asarray(X)
        if feature_names is None: feature_names = [f'f_{i}' for i in range(X_arr.shape[1])]
    
    n_samples = len(X_arr)
    if batch_size is None:
        avail = psutil.virtual_memory().available / (1024 * 1024)
        batch_size = max(10, min(500, int(avail * 0.3 / 0.5)))
    
    bg_idx = np.random.default_rng(42).choice(n_samples, size=min(n_samples, background_size), replace=False)
    background = X_arr[bg_idx]
    
    try: explainer = shap.TreeExplainer(model, background)
    except: explainer = shap.KernelExplainer(model.predict, background[:50])
    
    shap_vals_list = []
    for i in range(0, n_samples, batch_size):
        try:
            batch_shap = explainer.shap_values(X_arr[i:i+batch_size])
            if isinstance(batch_shap, list): batch_shap = batch_shap[0]
            shap_vals_list.append(batch_shap)
        except Exception as e:
            logger.warning(f"Batch {i} failed: {e}"); continue
    
    if not shap_vals_list: return pd.DataFrame(np.zeros((n_samples, len(feature_names))), columns=feature_names) if return_dataframe else np.zeros((n_samples, len(feature_names)))
    shap_values = np.vstack(shap_vals_list)
    return pd.DataFrame(shap_values, columns=feature_names) if return_dataframe else shap_values

def adjust_shap_for_correlation(
    shap_values: Union[pd.DataFrame, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    correlation_threshold: float = 0.7,
    method: Literal['equal_split', 'proportional'] = 'equal_split',
) -> Union[pd.DataFrame, np.ndarray]:
    """Adjust SHAP values for feature correlation bias"""
    if isinstance(shap_values, pd.DataFrame):
        names, shap_arr = list(shap_values.columns), shap_values.values
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=names)
    else:
        shap_arr = shap_values
        names = [f'f_{i}' for i in range(shap_arr.shape[1])]
        X_df = pd.DataFrame(X, columns=names)
    
    corr = X_df[names].corr().abs().values
    adj, processed = shap_arr.copy(), set()
    for i in range(len(names)):
        if i in processed: continue
        correlated = [j for j in range(len(names)) if i != j and corr[i, j] > correlation_threshold and j not in processed]
        if not correlated: continue
        group = [i] + correlated
        if method == 'equal_split':
            val = np.mean(adj[:, group], axis=1, keepdims=True)
            for idx in group: adj[:, idx] = val.flatten()
        processed.update(group)
    return pd.DataFrame(adj, columns=names) if isinstance(shap_values, pd.DataFrame) else adj

def compute_shap_confidence_intervals(model, X, n_bootstrap=10):
    """Compute SHAP values with bootstrap confidence intervals"""
    names = list(X.columns) if isinstance(X, pd.DataFrame) else [f'f_{i}' for i in range(X.shape[1])]
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    boot_results = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_arr), size=int(len(X_arr)*0.8), replace=True)
        vals = compute_shap_batch(model, X_arr[idx], return_dataframe=False)
        boot_results.append(np.mean(np.abs(vals), axis=0))
    boot_arr = np.array(boot_results)
    return {f: {'mean': float(np.mean(boot_arr[:, i])), 'std': float(np.std(boot_arr[:, i]))} for i, f in enumerate(names)}
