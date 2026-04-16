"""
backend/data/eda_advanced.py

Advanced EDA Engine — OFAT検出、Conflictデータ検出
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def detect_quasi_ofat_patterns(df: pd.DataFrame, threshold: float = 0.05) -> List[Dict[str, Any]]:
    """
    一変数のみが変化し、他の変数がほぼ不変（quasi-OFAT）であるデータペアを検出する。
    """
    results = []
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty: return []

    # スケーリングして変化量を均一に評価
    scaled_df = (num_df - num_df.min()) / (num_df.max() - num_df.min() + 1e-9)
    cols = scaled_df.columns
    data = scaled_df.values
    n_samples = len(df)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diffs = np.abs(data[i] - data[j])
            # 小さな変化しかない列の数
            near_constant_mask = diffs < threshold
            n_changing = np.sum(~near_constant_mask)
            
            if n_changing == 1:
                changing_col_idx = np.where(~near_constant_mask)[0][0]
                results.append({
                    "idx1": i,
                    "idx2": j,
                    "variable": cols[changing_col_idx],
                    "change_amount": diffs[changing_col_idx]
                })
    
    return results

def detect_conflict_data(df: pd.DataFrame, target_col: str, 
                        feature_threshold: float = 0.02, 
                        target_threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    説明変数が酷似しているのに目的変数が大きく異なる（Conflict）データを検出する。
    """
    if target_col not in df.columns: return []
    
    num_df = df.select_dtypes(include=[np.number])
    if target_col not in num_df.columns: return []
    
    features = num_df.drop(columns=[target_col])
    # スケーリング
    scaled_features = (features - features.min()) / (features.max() - features.min() + 1e-9)
    targets = df[target_col].values
    
    conflicts = []
    n_samples = len(df)
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 特徴量の差（平均絶対誤差）
            f_diff = np.mean(np.abs(scaled_features.iloc[i] - scaled_features.iloc[j]))
            # 目的変数の差（相対または絶対）
            t_diff = np.abs(targets[i] - targets[j])
            
            if f_diff < feature_threshold and t_diff > target_threshold:
                conflicts.append({
                    "idx1": i,
                    "idx2": j,
                    "feature_diff": f_diff,
                    "target_diff": t_diff
                })
                
    return conflicts
