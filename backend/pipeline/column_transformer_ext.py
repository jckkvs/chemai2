"""
backend/pipeline/column_transformer_ext.py — 精緻化版 (型判定・前処理ロジック)

統計的により厳密な型判定と前処理戦略の自動選択を行うモジュール。
"""

from typing import List, Dict, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def detect_column_type(
    series: pd.Series,
    col_name: str,
    min_categorical_ratio: float = 0.05,
    max_categorical_unique: int = 50,
    skew_threshold: float = 3.0,
    kurtosis_threshold: float = 10.0
) -> Literal[
    'numeric_continuous', 'numeric_discrete', 'numeric_exponential', 
    'numeric_lognormal', 'binary', 'categorical_low', 'categorical_high',
    'text', 'datetime', 'smiles'
]:
    """
    Detect variable type with statistically robust heuristics
    """
    # 【修正点1】事前チェック: 空/全欠損のケース
    if series.empty or series.isna().all():
        logger.debug(f"Column '{col_name}': empty or all NaN, defaulting to 'text'")
        return 'text'
    
    non_null = series.dropna()
    n_non_null = len(non_null)
    n_unique = non_null.nunique()
    
    # 【修正点1】SMILES検出
    if _is_likely_smiles(non_null):
        return 'smiles'
    
    # 【修正点2】カテゴリカル判定: データサイズ依存の動的閾値
    dynamic_ratio_threshold = min_categorical_ratio * max(1, np.log10(n_non_null + 1))
    
    # 文字列/オブジェクト型の処理
    if non_null.dtype == 'object' or str(non_null.dtype) == 'category':
        if n_unique / n_non_null <= dynamic_ratio_threshold or n_unique <= max_categorical_unique:
            return 'categorical_low' if n_unique <= 10 else 'categorical_high'
        return 'text'
    
    # 数値型の詳細判定
    if pd.api.types.is_numeric_dtype(non_null):
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1, 0.0, 1.0, True, False}:
            return 'binary'
        
        if pd.api.types.is_integer_dtype(non_null):
            if n_unique < n_non_null * 0.1 and n_unique <= 20:
                return 'numeric_discrete'
        
        if n_non_null >= 10:
            vals = non_null.values.astype(float)
            if (vals > 0).all():
                skewness = stats.skew(vals)
                kurt = stats.kurtosis(vals)
                if skewness > skew_threshold and kurt > kurtosis_threshold:
                    log_vals = np.log1p(vals)
                    log_skew = stats.skew(log_vals)
                    if abs(log_skew) < skew_threshold:
                        return 'numeric_lognormal'
                    return 'numeric_exponential'
                
                log_vals = np.log1p(vals)
                _, p_val = stats.normaltest(log_vals)
                if p_val > 0.05 and abs(stats.skew(log_vals)) < 1.0:
                    return 'numeric_lognormal'
            return 'numeric_continuous'
        return 'numeric_continuous'
    
    if pd.api.types.is_datetime64_any_dtype(non_null) or _is_datetime_string(non_null):
        return 'datetime'
    
    return 'text'


def _is_likely_smiles(series: pd.Series, threshold: float = 0.8) -> bool:
    """Heuristic SMILES detection with RDKit fallback"""
    sample = series.dropna()
    if len(sample) > 20:
        sample = sample.sample(20, random_state=42)
    
    try:
        from rdkit import Chem
        valid_count = sum(1 for s in sample if Chem.MolFromSmiles(str(s)) is not None)
        return (valid_count / len(sample)) >= threshold if len(sample) > 0 else False
    except ImportError:
        import re
        pattern = re.compile(r'^[A-Za-z0-9@+\-\[\]\(\)\\/%#=]+$')
        match_count = sum(1 for s in sample if pattern.match(str(s)))
        return (match_count / len(sample)) >= 0.95 if len(sample) > 0 else False


def _is_datetime_string(series: pd.Series) -> bool:
    """Detect datetime strings with robust pattern matching"""
    if series.dtype != 'object': return False
    sample = series.dropna()
    if len(sample) > 10:
        sample = sample.sample(10, random_state=42)
    
    datetime_patterns = [
        r'^\d{4}[-/]\d{2}[-/]\d{2}',
        r'^\d{2}[-/]\d{2}[-/]\d{4}',
        r'^\d{2}:\d{2}:\d{2}',
        r'^\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}',
    ]
    import re
    compiled_patterns = [re.compile(p) for p in datetime_patterns]
    match_count = sum(1 for s in sample if any(p.match(str(s)) for p in compiled_patterns))
    return (match_count / len(sample)) >= 0.8 if len(sample) > 0 else False


def get_preprocessing_strategy(
    col_type: str,
    missing_ratio: float,
    user_preference: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Determine preprocessing strategy based on column type and missingness"""
    if user_preference:
        return {
            'imputer': user_preference.get('imputer', 'mean'),
            'scaler': user_preference.get('scaler', 'standard'),
            'encoder': user_preference.get('encoder', 'onehot')
        }
    
    strategy = {}
    if missing_ratio == 0:
        strategy['imputer'] = 'none'
    elif missing_ratio < 0.05:
        strategy['imputer'] = 'mean' if col_type.startswith('numeric') else 'most_frequent'
    elif missing_ratio < 0.3:
        strategy['imputer'] = 'median' if col_type.startswith('numeric') else 'most_frequent'
    else:
        strategy['imputer'] = 'constant'
        strategy['imputer_fill_value'] = '0' if col_type.startswith('numeric') else 'missing'
    
    if col_type == 'numeric_exponential':
        strategy['scaler'] = 'power_yeojohnson'
    elif col_type == 'numeric_lognormal':
        strategy['scaler'] = 'log'
    elif col_type == 'numeric_discrete' and missing_ratio < 0.1:
        strategy['scaler'] = 'robust'
    else:
        strategy['scaler'] = 'standard'
    
    if col_type == 'categorical_low':
        strategy['encoder'] = 'onehot'
    elif col_type == 'categorical_high':
        strategy['encoder'] = 'ordinal'
    else:
        strategy['encoder'] = 'none'
    
    return strategy
