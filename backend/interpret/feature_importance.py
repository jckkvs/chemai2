# backend/interpret/feature_importance.py — 精緻化版 (特徴量重要度エンジン)

from typing import Union, List, Dict, Optional, Literal, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    random_state: Optional[int] = 42,
    scoring: Optional[str] = None,
    n_jobs: int = 1,
    sample_ratio: float = 1.0,
    normalize: Literal['sum', 'max', 'none'] = 'sum'
) -> Dict[str, Dict[str, float]]:
    """
    Compute permutation feature importance with reproducibility and correlation awareness
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feature_names = [f'feature_{i}' for i in range(X_arr.shape[1])]
    
    y_arr = np.asarray(y).ravel()
    
    # 【修正点3】大規模データのサンプリング
    if sample_ratio < 1.0 and len(X_arr) > 1000:
        n_samples = max(1000, int(len(X_arr) * sample_ratio))
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_arr), size=n_samples, replace=False)
        X_sample, y_sample = X_arr[indices], y_arr[indices]
        logger.info(f"Using {n_samples}/{len(X_arr)} samples")
    else:
        X_sample, y_sample = X_arr, y_arr
    
    # 【修正点1】再現性確保のためrandom_stateを明示
    result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=n_repeats, random_state=random_state,
        scoring=scoring, n_jobs=n_jobs
    )
    
    importances, stds = result.importances_mean, result.importances_std
    
    # 【修正点4】正規化処理
    if normalize == 'sum' and np.sum(np.abs(importances)) > 0:
        total = np.sum(np.abs(importances))
        importances, stds = importances / total, stds / total
    elif normalize == 'max' and np.max(np.abs(importances)) > 0:
        max_val = np.max(np.abs(importances))
        importances, stds = importances / max_val, stds / max_val
    
    output = {name: {'importance': float(imp), 'std': float(std), 'n_repeats': n_repeats} 
              for name, imp, std in zip(feature_names, importances, stds)}
    
    # 【修正点2】相関バイアスの検出
    if isinstance(X, pd.DataFrame) and len(feature_names) > 2:
        _check_correlation_bias(output, X, feature_names)
    
    return output


def _check_correlation_bias(importance_result: Dict[str, Dict[str, float]], X: pd.DataFrame, feature_names: List[str], correlation_threshold: float = 0.8):
    """Detect and warn about importance bias due to feature correlation"""
    corr = X[feature_names].corr().abs()
    biased_pairs = []
    for i, f1 in enumerate(feature_names):
        for f2 in feature_names[i+1:]:
            if corr.loc[f1, f2] > correlation_threshold:
                if importance_result.get(f1, {}).get('importance', 0) > 0.01 and importance_result.get(f2, {}).get('importance', 0) > 0.01:
                    biased_pairs.append((f1, f2, corr.loc[f1, f2]))
    if biased_pairs:
        msg = f"High correlation (>{correlation_threshold}) detected for {len(biased_pairs)} pairs. Importance may be biased."
        logger.warning(msg)


def compute_model_based_importance(model: BaseEstimator, feature_names: Optional[List[str]] = None, normalize: Literal['sum', 'max', 'none'] = 'sum') -> Dict[str, float]:
    """Extract model-based feature importance (tree-based or linear)"""
    importances = None
    if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim == 2 else np.abs(model.coef_)
    
    if importances is None: return {}
    if feature_names is None:
        feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importances))]
    
    if normalize == 'sum' and np.sum(np.abs(importances)) > 0: importances /= np.sum(np.abs(importances))
    elif normalize == 'max' and np.max(np.abs(importances)) > 0: importances /= np.max(np.abs(importances))
    return dict(zip(feature_names, importances.tolist()))


def aggregate_importance(results: List[Dict[str, Dict[str, float]]], method: Literal['mean', 'median', 'max'] = 'mean') -> Dict[str, Dict[str, float]]:
    """Aggregate importance results from multiple runs/folds"""
    if not results: return {}
    all_features = set().union(*(r.keys() for r in results))
    aggregated = {}
    for feat in all_features:
        vals = [r[feat]['importance'] for r in results if feat in r]
        if not vals: continue
        if method == 'mean': agg_val, agg_std = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
        elif method == 'median': agg_val, agg_std = np.median(vals), np.percentile(vals, [25, 75]).ptp() / 2
        else: agg_val, agg_std = np.max(vals), 0
        aggregated[feat] = {'importance': float(agg_val), 'std': float(agg_std), 'n_runs': len(vals)}
    return aggregated
