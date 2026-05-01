"""
EDA コア機能 - 統計解析・可視化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings


def compute_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """基本統計量を計算"""
    
    stats_dict = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                '25%': df[col].quantile(0.25),
                '50%': df[col].quantile(0.5),
                '75%': df[col].quantile(0.75),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            stats_dict[col] = col_stats
    
    return stats_dict


def compute_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """相関行列を計算"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """外れ値を検出"""
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                valid = df[col].dropna()
                z_scores = np.abs(stats.zscore(valid))
                outlier_mask.loc[valid.index, col] = z_scores > threshold
    
    return outlier_mask


def compute_dimensionality_reduction(X: np.ndarray, method: str = 'pca', 
                                     n_components: int = 2,
                                     **kwargs) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    次元削減を実行
    
    Args:
        X: 入力データ (n_samples, n_features)
        method: 手法 ('pca', 'tsne', 'umap')
        n_components: 出力次元数
        **kwargs: 各手法固有のパラメータ
    
    Returns:
        (coordinates, explained_variance) または None
    """
    
    if X is None or X.shape[0] == 0:
        return None
    
    n_samples, n_features = X.shape
    
    # データ数が次元数より少ない場合の処理
    if n_samples <= n_components:
        warnings.warn(f"データ数 ({n_samples}) が次元数 ({n_components}) 以下です。")
        return None
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        if method == 'pca':
            # PCA
            actual_components = min(n_components, n_samples - 1, n_features)
            pca = PCA(n_components=actual_components)
            coords = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_
            return coords, explained_var
        
        elif method == 'tsne':
            # t-SNE
            # Perplexity の自動調整（データ数に応じて制限）
            perplexity = kwargs.get('perplexity', 5.0)
            
            # perplexity は n_samples - 1 より小さくなければならない
            max_perplexity = max(1.0, n_samples - 1 - 1e-5)
            safe_perplexity = min(perplexity, max_perplexity)
            
            # データ数が極端に少ない場合は早期リターン
            if n_samples < 2:
                warnings.warn("t-SNE: データが少なすぎます（2サンプル以上必要）")
                return None
            
            # 学習率の設定
            learning_rate = kwargs.get('learning_rate', 'auto')
            
            tsne = TSNE(
                n_components=min(n_components, n_samples - 1),
                perplexity=safe_perplexity,
                learning_rate=learning_rate,
                random_state=42,
                n_iter=1000,
                init='pca',
                method='barnes_hut' if n_samples > 100 else 'exact'
            )
            
            coords = tsne.fit_transform(X_scaled)
            return coords, None
        
        elif method == 'umap':
            # UMAP（オプション）
            try:
                import umap
                
                # n_neighbors の自動調整
                n_neighbors = kwargs.get('n_neighbors', 15)
                max_neighbors = max(1, n_samples - 1)
                safe_neighbors = min(n_neighbors, max_neighbors)
                
                reducer = umap.UMAP(
                    n_components=min(n_components, n_samples - 1),
                    n_neighbors=safe_neighbors,
                    random_state=42
                )
                
                coords = reducer.fit_transform(X_scaled)
                return coords, None
            
            except ImportError:
                warnings.warn("UMAP requires 'umap-learn' package. Install with: pip install umap-learn")
                return None
        
        else:
            warnings.warn(f"Unknown method: {method}")
            return None
    
    except Exception as e:
        warnings.warn(f"{method.upper()} error: {str(e)}")
        return None


def compute_feature_importance(X: np.ndarray, y: np.ndarray, 
                               method: str = 'mutual_info') -> np.ndarray:
    """特徴量の重要度を計算"""
    
    from sklearn.feature_selection import mutual_info_regression, f_regression
    
    if method == 'mutual_info':
        importance = mutual_info_regression(X, y, random_state=42)
    elif method == 'f_score':
        f_scores, _ = f_regression(X, y)
        importance = f_scores
    else:
        importance = np.zeros(X.shape[1])
    
    return importance


def compute_distribution_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """分布統計量を計算"""
    
    if column not in df.columns:
        return {}
    
    data = df[column].dropna()
    
    if not pd.api.types.is_numeric_dtype(data):
        return {}
    
    # 正規性検定
    if len(data) >= 3:
        _, p_value_normal = stats.normaltest(data)
        _, p_value_shapiro = stats.shapiro(data[:5000])  # Shapiro は最大5000サンプル
    else:
        p_value_normal = p_value_shapiro = None
    
    return {
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0] if len(data.mode()) > 0 else None,
        'variance': data.var(),
        'std': data.std(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'p_value_normality': p_value_normal,
        'p_value_shapiro': p_value_shapiro,
        'is_normal': p_value_normal is not None and p_value_normal > 0.05
    }


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """データ全体の概要を取得"""
    
    return {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        'total_missing': df.isna().sum().sum(),
        'missing_rate': df.isna().sum().sum() / (len(df) * len(df.columns)) * 100,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)
    }
