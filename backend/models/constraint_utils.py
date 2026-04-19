"""
モデルフレームワーク別 単調性制約適用ユーティリティ
LightGBM / XGBoost / CatBoost / Scikit-Learn に対応
"""
from typing import Dict, List, Any, Optional
import numpy as np


def apply_constraints_to_params(
    model_type: str,
    base_params: Dict[str, Any],
    constraints: Dict[str, Any],
    feature_columns: List[str],
) -> Dict[str, Any]:
    """
    制約情報をモデルパラメータに適用
    
    Parameters
    ----------
    model_type : str
        'lightgbm', 'xgboost', 'catboost', 'sklearn'
    base_params : dict
        基本学習パラメータ
    constraints : dict
        制約設定辞書 {column: {type, direction, ...}}
    feature_columns : list
        特徴量列名リスト（順序固定）
    """
    params = base_params.copy()
    
    # 方向リストの作成
    directions = []
    for col in feature_columns:
        info = constraints.get(col, {})
        direction = info.get('direction', 'none') if isinstance(info, dict) else 'none'
        if direction == 'positive':
            directions.append(1)
        elif direction == 'negative':
            directions.append(-1)
        else:
            directions.append(0)
    
    if all(d == 0 for d in directions):
        return params  # 制約なしならそのまま返す
    
    # フレームワーク別適用
    if model_type in ('lightgbm', 'lgbm'):
        params['monotone_constraints'] = directions
        # 対応するカーネル/アルゴリズムを指定（制約対応必須）
        params['monotone_constraints_method'] = 'basic'
        
    elif model_type in ('xgboost', 'xgb'):
        # XGBoostは dict 形式 {"f0": 1, "f1": -1, ...}
        mono_dict = {f"f{i}": d for i, d in enumerate(directions)}
        params['monotone_constraints'] = mono_dict
        
    elif model_type in ('catboost', 'cb'):
        # CatBoostはリスト形式（LightGBMと同様）
        params['monotone_constraints'] = directions
        
    elif model_type == 'sklearn':
        # scikit-learn は単調制約をネイティブサポートしない
        # 代替: 制約付き最適化ラッパーまたは警告
        import warnings
        warnings.warn(
            f"{model_type} は単調性制約をネイティブサポートしません。"
            "制約は無視されます。LightGBM/XGBoost/CatBoostの使用を推奨します。"
        )
    
    return params


def validate_constraints_for_model(
    model_type: str,
    constraints: Dict,
    feature_columns: List[str],
) -> List[str]:
    """制約設定の妥当性検証（警告メッセージを返す）"""
    warnings = []
    
    if model_type == 'sklearn' and any(
        info.get('direction') != 'none' 
        for info in constraints.values() 
        if isinstance(info, dict)
    ):
        warnings.append("⚠️ scikit-learnは単調制約未サポート。LightGBM/XGBoostを推奨します。")
    
    n_constrained = sum(
        1 for info in constraints.values() 
        if isinstance(info, dict) and info.get('direction') != 'none'
    )
    if n_constrained > len(feature_columns) * 0.8:
        warnings.append("⚠️ 80%以上の特徴量に制約が設定されています。過剰制約に注意。")
    
    return warnings
