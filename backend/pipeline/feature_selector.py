# backend/pipeline/feature_selector.py — 精緻化版 (特徴選択実行コア)

from typing import List, Dict, Optional, Union, Tuple, Literal, Any
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_score, KFold, RFECV
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    SelectKBest,
    f_regression,
    f_classif,
    mutual_info_regression,
    mutual_info_classif,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from dataclasses import dataclass, field
import gc

from backend.utils.config import RANDOM_STATE, AUTOML_N_JOBS
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# オプショナルライブラリ
_skrebate = safe_import("skrebate", "relieff")
_boruta = safe_import("boruta", "boruta")
_genetic = safe_import("sklearn_genetic", "sklearn-genetic-opt")
_xgb = safe_import("xgboost", "xgboost")

@dataclass
class FeatureSelectorConfig:
    """特徴量選択の設定。"""
    method: str = "none"
    task: str = "regression"
    threshold: str | float = "mean"
    max_features: int | None = None
    percentile: int = 50
    k: int | str = 10
    score_func: str = "f_regression"
    relieff_n_features: int = 10
    relieff_n_neighbors: int = 100
    boruta_n_estimators: int = 100
    boruta_max_iter: int = 50
    stability_n_iterations: int = 5
    stability_threshold: float = 0.7
    random_state: int = 42

def select_features_stable(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal['boruta', 'rfecv', 'mutual_info', 'lasso', 'variance'] = 'boruta',
    n_iterations: int = 5,
    stability_threshold: float = 0.7,
    max_features: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[List[str], Dict[str, float]]:
    """
    Perform feature selection with stability assessment across multiple runs
    """
    if X.empty or len(X) < 10:
        logger.warning("Insufficient data for feature selection")
        return list(X.columns), {col: 1.0 for col in X.columns}
    
    rng = np.random.default_rng(random_state)
    selection_runs = []
    
    for run in range(n_iterations):
        seed = int(rng.integers(0, 2**31))
        try:
            if method == 'boruta':
                selected = _run_boruta(X, y, seed=seed)
            elif method == 'rfecv':
                selected = _run_rfecv(X, y, seed=seed)
            elif method == 'mutual_info':
                selected = _run_mutual_info(X, y, max_features, seed=seed)
            else:
                selected = list(X.columns)
            
            selection_runs.append(set(selected))
            if verbose:
                logger.debug(f"Run {run+1}/{n_iterations}: selected {len(selected)} features")
        except Exception as e:
            logger.error(f"Feature selection run {run+1} failed: {e}")
            selection_runs.append(set(X.columns))
        gc.collect()
    
    if not selection_runs:
        return list(X.columns), {col: 0.0 for col in X.columns}
    
    all_features = set(X.columns)
    stability_scores = {feat: sum(1 for run in selection_runs if feat in run) / n_iterations for feat in all_features}
    stable_features = [f for f, s in stability_scores.items() if s >= stability_threshold]
    
    if not stable_features:
        logger.warning("No features met stability threshold. Using most frequent selection.")
        freq = {}
        for run in selection_runs:
            for f in run: freq[f] = freq.get(f, 0) + 1
        stable_features = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    
    if max_features and len(stable_features) > max_features:
        stable_features = stable_features[:max_features]
    
    final_stability = {f: stability_scores.get(f, 0.0) for f in stable_features}
    logger.info(f"Feature selection completed: {len(stable_features)}/{len(X.columns)} selected, avg stability: {np.mean(list(final_stability.values())):.2f}")
    return stable_features, final_stability

def _run_boruta(X: pd.DataFrame, y: pd.Series, seed: int) -> List[str]:
    from boruta import BorutaPy
    is_class = y.dtype == 'object' or y.nunique() < 10
    estimator = RandomForestClassifier(n_estimators=100, random_state=seed) if is_class else RandomForestRegressor(n_estimators=100, random_state=seed)
    boruta = BorutaPy(estimator, n_estimators='auto', max_iter=50, random_state=seed, verbose=0, two_step=True)
    boruta.fit(X.values.astype(np.float32), y.values.ravel())
    selected = X.columns[boruta.support_].tolist()
    return selected if selected else list(X.columns[:5])

def _run_rfecv(X: pd.DataFrame, y: pd.Series, seed: int) -> List[str]:
    estimator = GradientBoostingRegressor(random_state=seed, n_estimators=50)
    rfe = RFECV(estimator, step=1, cv=3, scoring='r2', min_features_to_select=2, n_jobs=1)
    rfe.fit(X.values, y.values)
    selected = X.columns[rfe.support_].tolist()
    return selected if selected else list(X.columns[:3])

def _run_mutual_info(X: pd.DataFrame, y: pd.Series, max_features: int | None, seed: int) -> List[str]:
    from sklearn.feature_selection import mutual_info_regression, SelectKBest
    k = max_features if max_features else min(10, X.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()

class FeatureSelector(BaseEstimator, TransformerMixin):
    """特徴量選択 Transformer。"""
    def __init__(self, config: FeatureSelectorConfig | None = None, column_meta: dict | None = None) -> None:
        self.config = config or FeatureSelectorConfig()
        self.column_meta = column_meta or {}
        self._feature_names_in: list[str] = []
        self._selected_features: list[str] = []
        self._stability_scores: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        self._feature_names_in = X.columns.tolist()
        if self.config.method == "none":
            self._selected_features = self._feature_names_in
            return self
        
        self._selected_features, self._stability_scores = select_features_stable(
            X, y,
            method=self.config.method,
            n_iterations=self.config.stability_n_iterations,
            stability_threshold=self.config.stability_threshold,
            max_features=self.config.max_features,
            random_state=self.config.random_state
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self._selected_features]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._selected_features)
