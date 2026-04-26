# backend/interpret/sage_wrapper.py — 精緻化版 (SAGE計算エンジン)

from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


def compute_sage_values(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    n_samples: Optional[int] = None,
    n_permutations: int = 20,
    n_bootstrap: int = 10,  # 【修正点1】分散推定用ブートストラップ数
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
    correlate_adjustment: bool = True  # 【修正点4】相関バイアス補正
) -> Dict[str, Dict[str, float]]:
    """
    Compute SAGE (Shapley Additive Global Importance) values with statistical rigor
    """
    if X.empty: return {}
    
    # 【修正点2】大規模データ時の層化サンプリング
    if n_samples is not None and len(X) > n_samples:
        if y is not None and y.dtype == 'object':
            X_sample, y_sample = _stratified_sample(X, y, n_samples, random_state)
        else:
            X_sample, y_sample = _quantile_stratified_sample(X, y, n_samples, random_state)
    else:
        X_sample, y_sample = X.copy(), y.copy() if y is not None else None
    
    feature_names = list(X_sample.columns)
    base_seed = random_state if random_state is not None else 42
    rng = np.random.default_rng(base_seed)
    
    # 基本SAGE値計算（点推定）
    sage_point = _compute_sage_point_estimate(model, X_sample, y_sample, feature_names, n_permutations, rng)
    
    # 【修正点1】ブートストラップ分散推定
    if n_bootstrap > 1:
        sage_bootstrap = _compute_sage_bootstrap(model, X_sample, y_sample, feature_names, n_permutations, n_bootstrap, base_seed, n_jobs)
        results = {}
        for feat in feature_names:
            values = np.array([r[feat] for r in sage_bootstrap])
            results[feat] = {
                'sage_value': sage_point[feat],
                'std_error': float(np.std(values, ddof=1)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5)),
                'n_bootstrap': n_bootstrap
            }
    else:
        results = {feat: {'sage_value': val, 'std_error': None} for feat, val in sage_point.items()}
    
    # 【修正点4】相関バイアス補正
    if correlate_adjustment and len(feature_names) > 1:
        results = _adjust_for_correlation_bias(results, X_sample, feature_names)
    
    return results


def _compute_sage_point_estimate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    feature_names: List[str],
    n_permutations: int,
    rng: np.random.Generator
) -> Dict[str, float]:
    """Compute point estimates of SAGE values"""
    n_samples = len(X)
    baseline_loss = _compute_loss(model, X, y) if y is not None else None
    sage_values = {}
    
    for feat_name in feature_names:
        loss_diffs = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            X_perm[feat_name] = rng.choice(X[feat_name].values, size=n_samples, replace=True)
            perm_loss = _compute_loss(model, X_perm, y) if y is not None else 0
            if baseline_loss is not None:
                loss_diffs.append(perm_loss - baseline_loss)
            else:
                orig_pred, perm_pred = model.predict(X), model.predict(X_perm)
                loss_diffs.append(np.mean((orig_pred - perm_pred) ** 2))
        sage_values[feat_name] = -np.mean(loss_diffs)
    return sage_values


def _compute_sage_bootstrap(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    feature_names: List[str],
    n_permutations: int,
    n_bootstrap: int,
    base_seed: int,
    n_jobs: int
) -> List[Dict[str, float]]:
    """Compute bootstrap samples for uncertainty estimation"""
    n_samples = len(X)
    def bootstrap_iteration(seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return _compute_sage_point_estimate(model, X.iloc[indices], y.iloc[indices] if y is not None else None, feature_names, n_permutations, rng)
    
    seeds = [base_seed + i for i in range(n_bootstrap)]
    if n_jobs == 1 or n_bootstrap == 1:
        return [bootstrap_iteration(seed) for seed in seeds]
    else:
        max_workers = min(n_jobs if n_jobs > 0 else mp.cpu_count(), n_bootstrap)
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(bootstrap_iteration, seed): seed for seed in seeds}
            for future in as_completed(futures):
                try: results.append(future.result(timeout=300))
                except Exception as e:
                    seed = futures[future]
                    logger.warning(f"Bootstrap {seed} failed: {e}")
                    results.append(_compute_sage_point_estimate(model, X, y, feature_names, n_permutations, np.random.default_rng(seed)))
        return results


def _adjust_for_correlation_bias(
    sage_results: Dict[str, Dict[str, float]],
    X: pd.DataFrame,
    feature_names: List[str],
    correlation_threshold: float = 0.7
) -> Dict[str, Dict[str, float]]:
    """Adjust SAGE values for feature correlation bias"""
    if len(feature_names) < 2: return sage_results
    corr_matrix = X[feature_names].corr().values
    adjusted = {}
    for i, feat in enumerate(feature_names):
        orig_sage = sage_results[feat]['sage_value']
        high_corr = [j for j, f in enumerate(feature_names) if i != j and abs(corr_matrix[i, j]) > correlation_threshold]
        if not high_corr:
            adjusted[feat] = sage_results[feat].copy(); continue
        n_corr = len(high_corr) + 1
        adjusted[feat] = sage_results[feat].copy()
        adjusted[feat]['sage_value'] = orig_sage / n_corr
        adjusted[feat]['correlation_adjusted'] = True
        adjusted[feat]['correlated_with'] = [feature_names[j] for j in high_corr]
    return adjusted


def _stratified_sample(X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=random_state, shuffle=True)
    return X_sample, y_sample


def _quantile_stratified_sample(X: pd.DataFrame, y: Optional[pd.Series], n_samples: int, random_state: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    rng = np.random.default_rng(random_state)
    if y is None:
        indices = rng.choice(len(X), size=n_samples, replace=False)
        return X.iloc[indices], None
    n_strata = min(10, n_samples // 10)
    if n_strata < 2:
        indices = rng.choice(len(X), size=n_samples, replace=False)
        return X.iloc[indices], y.iloc[indices]
    y_quantiles = pd.qcut(y, q=n_strata, labels=False, duplicates='drop')
    indices = []
    for stratum in range(y_quantiles.nunique()):
        stratum_idx = y_quantiles[y_quantiles == stratum].index
        n_stratum = max(1, int(n_samples * len(stratum_idx) / len(y)))
        n_stratum = min(n_stratum, len(stratum_idx))
        indices.extend(rng.choice(stratum_idx, size=n_stratum, replace=False))
    return X.iloc[indices], y.iloc[indices]


def _compute_loss(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    from sklearn.metrics import mean_squared_error, log_loss
    pred = model.predict(X)
    if y.dtype == 'object' or y.nunique() < 10:
        if hasattr(model, 'predict_proba'): return log_loss(y, model.predict_proba(X))
        else: return np.mean(pred != y)
    else: return mean_squared_error(y, pred)
