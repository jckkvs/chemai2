"""
数値データとSMILES特徴量の統合レイヤー

SMILES列を含むデータと数値データを統合し、
すべて数値特徴量として機械学習パイプラインに渡せる形式に変換する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from backend.chem.smiles_feature_calculator import SMILESFeatureCalculator, FeatureCalculationResult
from backend.chem.descriptor_sets import DescriptorSet
from backend.chem.charge_config import ChargeConfig


@dataclass
class MergedDataResult:
    """統合結果のデータクラス"""
    features: pd.DataFrame  # 統合された特徴量（すべて数値）
    target: Optional[pd.Series] = None # 目的変数
    groups: Optional[pd.DataFrame] = None # グループ列
    df: Optional[pd.DataFrame] = None # 元のデータフレーム（計算済み記述子を含む） - TODO: 既存との互換性用
    metadata: Dict = field(default_factory=dict)  # 列の属性情報
    feature_set_info: Dict[str, List[str]] = field(default_factory=dict)  # {set_id: [column_names]}
    
    @property
    def all_numeric_columns(self) -> List[str]:
        """すべての数値特徴量列名を返す"""
        return self.features.columns.tolist()
    
    @property
    def n_samples(self) -> int:
        return len(self.features)
    
    @property
    def n_features(self) -> int:
        return self.features.shape[1]


class FeatureMerger:
    """
    数値データとSMILES特徴量を統合するクラス
    """
    
    def __init__(self, calculator: Optional[SMILESFeatureCalculator] = None):
        self.calculator = calculator or SMILESFeatureCalculator()
    
    def merge(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        smiles_column: Optional[str] = None,
        descriptor_set: Optional[DescriptorSet] = None,
        target_column: Optional[str] = None,
        group_columns: Optional[List[str]] = None,
        charge_config: Optional[ChargeConfig] = None,
        progress_callback: Optional[callable] = None,
    ) -> MergedDataResult:
        """
        DataFrameを統合数値特徴量に変換
        """
        # 列の自動検出
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 目的変数・グループ列の事前抽出
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column].copy()
        
        groups = None
        if group_columns:
            valid_groups = [c for c in group_columns if c in df.columns]
            if valid_groups:
                groups = df[valid_groups].copy()
        
        # SMILES特徴量の計算
        smiles_features = None
        feature_set_info: Dict[str, List[str]] = {}
        
        metadata = {"smiles_column": smiles_column, "n_smiles_features": 0}

        if smiles_column and smiles_column in df.columns and descriptor_set:
            smiles_list = df[smiles_column].dropna().astype(str).tolist()
            
            # 特徴量計算
            calc_result = self.calculator.calculate(
                smiles_list=smiles_list,
                descriptor_set=descriptor_set,
                charge_config=charge_config,
                progress_callback=progress_callback,
            )
            
            # 計算結果をDataFrameにマッピング
            smiles_features = calc_result.features
            
            # 特徴量セット情報の記録
            set_name = descriptor_set.name.replace(" ", "_").lower()
            feature_set_info[set_name] = smiles_features.columns.tolist()
            
            # メタデータに計算情報を追加
            metadata.update({
                "descriptor_set": descriptor_set.name,
                "engines_used": calc_result.metadata.get("engines_used", []),
                "n_smiles_features": len(smiles_features.columns),
                "failed_smiles_count": len(calc_result.failed_smiles),
            })

        # 数値特徴量の抽出
        numeric_df = df[[c for c in numeric_columns if c in df.columns]].copy()
        
        # 統合
        if smiles_features is not None and not smiles_features.empty:
            # 元データのインデックスに合わせて再構築
            # smiles_features は smiles_list (有効なSMILESのみ) の順。
            # 元の DataFrame とマージするために、インデックスを合わせる
            # ここでは単純に pd.concat を使うが、smiles_features は計算対象行のみなので注意。
            # 計算失敗行や欠損行は NaN または 前方埋めされる（Calculator内で処理済み）
            merged_features = pd.concat(
                [numeric_df.reset_index(drop=True), 
                 smiles_features.reset_index(drop=True)], 
                axis=1
            )
        else:
            merged_features = numeric_df.copy()
        
        # 数値型のみを保持
        numeric_cols_final = merged_features.select_dtypes(include=[np.number]).columns.tolist()
        merged_features = merged_features[numeric_cols_final]
        
        metadata.update({
            "original_numeric_cols": numeric_columns,
            "total_features": len(merged_features.columns),
            "feature_set_columns": feature_set_info,
        })
        
        return MergedDataResult(
            features=merged_features,
            target=target,
            groups=groups,
            metadata=metadata,
            feature_set_info=feature_set_info,
        )
    
    def merge_multiple_sets(
        self,
        df: pd.DataFrame,
        smiles_column: str,
        numeric_columns: List[str],
        descriptor_sets: List[DescriptorSet],
        target_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, MergedDataResult]:
        """
        複数の特徴量セットに対して個別に統合を実行
        """
        results = {}
        for desc_set in descriptor_sets:
            result = self.merge(
                df=df,
                numeric_columns=numeric_columns,
                smiles_column=smiles_column,
                descriptor_set=desc_set,
                target_column=target_column,
                **kwargs
            )
            results[desc_set.name] = result
        return results
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        DataFrameの列をタイプ別に分類
        """
        result = {
            "numeric": [],
            "smiles": [],
            "categorical": [],
            "other": [],
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                result["numeric"].append(col)
            elif df[col].astype(str).str.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#\$%\./\\]+$', na=False).any():
                non_null = df[col].dropna()
                if len(non_null) > 0 and non_null.astype(str).str.len().mean() < 500:
                    result["smiles"].append(col)
            elif pd.api.types.is_categorical_dtype(dtype) or (not pd.api.types.is_numeric_dtype(dtype) and df[col].nunique() < 20):
                result["categorical"].append(col)
            else:
                result["other"].append(col)
        
        return result
