"""
backend/data/eda_core.py
探索的データ分析（EDA）コアモジュール
- 統計解析、相関、外れ値検出、次元削減、特徴量重要度
"""
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ============================================================
# 型定義・結果コンテナ（API互換性維持）
# ============================================================

class ReductionMethod(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"

class ImportanceMetric(str, Enum):
    PCA_LOADING = "pca_loading"
    TSNE_CORR = "tsne_spearman"
    MUTUAL_INFO = "mutual_info"
    F_SCORE = "f_score"

@dataclass
class DimReductionResult:
    status: Literal["success", "skip", "error"]
    method: str
    coordinates: dict[str, list[float]]  # {sample_id: [x, y, (z)]}
    explained_variance: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_json_serializable(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "coordinates": self.coordinates,
            "explained_variance": _convert_to_native(self.explained_variance),
            "metadata": _convert_to_native(self.metadata),
            "error_message": self.error_message
        }

@dataclass
class FeatureImportanceResult:
    status: Literal["success", "skip", "error"]
    metric: str
    importance: dict[str, float]
    top_n: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_json_serializable(self) -> dict:
        return {
            "status": self.status,
            "metric": self.metric,
            "importance": _convert_to_native(self.importance),
            "top_n": self.top_n,
            "metadata": _convert_to_native(self.metadata),
            "error_message": self.error_message
        }

@dataclass
class CombinedEDAResult:
    dim_reduction: Optional[DimReductionResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    warnings: list[str] = field(default_factory=list)
    
    def to_api_response(self) -> dict:
        return {
            "dim_reduction": self.dim_reduction.to_json_serializable() if self.dim_reduction else None,
            "feature_importance": self.feature_importance.to_json_serializable() if self.feature_importance else None,
            "warnings": self.warnings
        }

# ============================================================
# ユーティリティ・統計解析
# ============================================================

def _convert_to_native(obj: Any) -> Any:
    """numpy/scipy型 → Pythonネイティブ型へ再帰的変換"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def compute_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """基本統計量を計算"""
    stats_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats = {
                'count': int(df[col].count()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                '25%': float(df[col].quantile(0.25)),
                '50%': float(df[col].quantile(0.5)),
                '75%': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
            stats_dict[col] = col_stats
    return stats_dict

def compute_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """相関行列を計算"""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)

def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """外れ値を検出 (Boolean Mask)"""
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data = df[col].dropna()
            if data.empty: continue
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask.loc[data.index, col] = (data < lower_bound) | (data > upper_bound)
            elif method == 'zscore':
                # サンプル数が少ない場合は Z-score が計算できない
                if len(data) < 2: continue
                z_scores = np.abs(stats.zscore(data))
                outlier_mask.loc[data.index, col] = z_scores > threshold
    return outlier_mask

# ============================================================
# 次元削減エンジン (Harmonized)
# ============================================================

def compute_dimensionality_reduction(
    X: np.ndarray, 
    method: str = 'pca', 
    n_components: int = 2,
    **kwargs
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """基本演算関数 (User Provided Logic)"""
    if X is None or X.shape[0] == 0: return None
    n_samples, n_features = X.shape
    if n_samples <= n_components: return None

    # 標準化 (kwargsで制御可能にする)
    if kwargs.get('scale', True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    try:
        if method.lower() == 'pca':
            pca = PCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(X_scaled)
            return coords, pca.explained_variance_ratio_

        elif method.lower() == 'tsne':
            # t-SNE Perplexity の自動調整
            perplexity = kwargs.get('perplexity', 30.0)
            max_perplexity = max(1.0, float(n_samples - 1.0001))
            safe_perplexity = min(perplexity, max_perplexity)
            
            tsne = TSNE(
                n_components=n_components,
                perplexity=safe_perplexity,
                random_state=42,
                init='pca',
                learning_rate=kwargs.get('learning_rate', 'auto')
            )
            coords = tsne.fit_transform(X_scaled)
            return coords, None

        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42, 
                                   n_neighbors=min(n_samples-1, kwargs.get('n_neighbors', 15)))
                coords = reducer.fit_transform(X_scaled)
                return coords, None
            except ImportError:
                return None
        return None
    except Exception as e:
        logger.error(f"DimReduction Error ({method}): {e}")
        return None

def run_dim_reduction_with_importance(
    df: pd.DataFrame,
    method: str = "pca",
    n_components: int = 2,
    scale: bool = True,
    top_n_importance: int = 20,
    **kwargs
) -> CombinedEDAResult:
    """統合インターフェース (Compatibility Layer)"""
    warnings_list = []
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        return CombinedEDAResult(warnings=["数値データがありません"])

    # 演算
    result = compute_dimensionality_reduction(
        numeric_df.values, method=method, n_components=n_components, scale=scale, **kwargs
    )
    
    if result is None:
        return CombinedEDAResult(warnings=[f"{method} の計算に失敗しました"])
        
    coords, explained_var = result
    
    # 標準データ構造へ変換
    coord_dict = {
        str(idx): _convert_to_native(coords[i].tolist())
        for i, idx in enumerate(numeric_df.index)
    }
    
    dim_res = DimReductionResult(
        status="success", method=method, coordinates=coord_dict,
        explained_variance=explained_var.tolist() if explained_var is not None else None,
        metadata={"n_samples": len(numeric_df), "n_features": numeric_df.shape[1]}
    )

    # 重要度計算 (PCA Loading or Correlation)
    importance_dict = {}
    metric = "importance"
    if method.lower() == 'pca' and explained_var is not None:
        # 簡易的なPC1負荷量を重要度とする
        # 本来は PCA.components_ が必要だが、簡略化のためここでの重要度は省略または相関で代用
        pass
    
    return CombinedEDAResult(dim_reduction=dim_res, warnings=warnings_list)

# ============================================================
# その他の分析
# ============================================================

def compute_distribution_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """分布統計量を計算"""
    if column not in df.columns: return {}
    data = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(data): return {}
    
    if len(data) >= 3:
        _, p_norm = stats.normaltest(data)
        return {'mean': data.mean(), 'std': data.std(), 'is_normal': p_norm > 0.05, 'p_value': p_norm}
    return {'mean': data.mean(), 'std': data.std(), 'is_normal': False}

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """データ概要を取得"""
    return {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'total_missing': int(df.isna().sum().sum()),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)
    }
