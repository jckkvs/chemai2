"""
backend/data/data_merger.py

数値データと複数のSMILES特徴量セットを統合するモジュール。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.data.type_detector import TypeDetector, ColumnType

logger = logging.getLogger(__name__)

@dataclass
class MergedDataResult:
    """統合されたデータの実行結果。"""
    df: pd.DataFrame
    feature_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_col: Optional[str] = None
    smiles_col: Optional[str] = None
    
    def get_features_by_source(self, source: str) -> List[str]:
        """指定されたソース（'raw' or 'smiles_set'）の列名リストを返す。"""
        return [
            name for name, meta in self.feature_metadata.items()
            if meta.get("source") == source
        ]

    def get_features_by_set(self, set_id: str) -> List[str]:
        """指定されたセットIDに含まれる列名リストを返す。"""
        return [
            name for name, meta in self.feature_metadata.items()
            if meta.get("set_id") == set_id
        ]

class DataMerger:
    """
    数値データとSMILES記述子セットを安全にマージし、単一の数値DataFrameを作成する。
    """
    def __init__(self, raw_df: pd.DataFrame, target_col: str, smiles_col: str | None = None):
        self.raw_df = raw_df
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.detector = TypeDetector()

    def merge(self, smiles_feature_sets: List[Any]) -> MergedDataResult:
        """
        raw_df の数値列と、複数の SMILESFeatureSet を統合する。
        """
        # 1. raw_df から数値列（および目的変数）を抽出
        detection = self.detector.detect(self.raw_df)
        numeric_cols = detection.numeric_columns
        
        # 目的変数は必ず含める（数値でない場合も分類タスクとして必要）
        cols_to_keep = list(set(numeric_cols + [self.target_col]))
        
        # 統合用DFのベース
        merged_df = self.raw_df[cols_to_keep].copy()
        
        # メタデータの初期化
        feature_metadata = {}
        for col in numeric_cols:
            if col == self.target_col:
                continue
            feature_metadata[col] = {
                "source": "raw",
                "original_name": col,
                "category": "numeric"
            }

        # 2. SMILES特徴量セットをマージ
        for fset in smiles_feature_sets:
            if fset.dataframe is not None and not fset.dataframe.empty:
                # set_id を使って列名を一意に保ちつつマージ
                # smiles_feature_sets.py ですでに接頭辞が付与されている前提
                merged_df = pd.concat([merged_df, fset.dataframe], axis=1)
                
                # 特徴量ごとのメタデータを記録
                for col in fset.dataframe.columns:
                    # set1_RDKit_MW のような名前から分解を試みる（簡易的）
                    feature_metadata[col] = {
                        "source": "smiles_set",
                        "set_id": fset.id,
                        "original_name": col.replace(f"{fset.id}_", ""),
                        "category": "molecular_descriptor"
                    }

        return MergedDataResult(
            df=merged_df,
            feature_metadata=feature_metadata,
            target_col=self.target_col,
            smiles_col=self.smiles_col
        )
