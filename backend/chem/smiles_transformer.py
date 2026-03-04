"""
backend/chem/smiles_transformer.py

sklearn Pipeline に組み込める SMILES→記述子変換 Transformer。
学習時・推論時を通じて一貫した変換が保証される。
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class SmilesDescriptorTransformer(BaseEstimator, TransformerMixin):
    """
    SMILES列を記述子に変換するsklearn互換Transformer。

    Parameters
    ----------
    smiles_col : str
        SMILESが入力されている列名。
    selected_descriptors : list[str] | None
        使用する記述子名のリスト。Noneの場合は全計算結果を使用。
    """

    def __init__(
        self,
        smiles_col: str,
        selected_descriptors: list[str] | None = None,
    ) -> None:
        self.smiles_col = smiles_col
        self.selected_descriptors = selected_descriptors
        self._descriptor_cols: list[str] = []
        self._non_smiles_cols: list[str] = []

    def _compute_descriptors(self, smiles_list: list[str]) -> pd.DataFrame:
        """SMILESリストから記述子DataFrameを計算する。"""
        from backend.chem import RDKitAdapter, MordredAdapter
        from backend.chem.psmiles_adapter import PSmilesAdapter
        
        # ポリマーSMILES (PSMILES) の検出
        # リストの先頭50件を見て、1件でも '*' または '[*]' があればポリマーとみなす
        has_psmiles = any(PSmilesAdapter.is_psmiles(smi) for smi in smiles_list[:50] if isinstance(smi, str))
        
        adapters = []
        if has_psmiles:
            logger.info("PSMILESを検出しました。ポリマー用記述子抽出モード(PSmilesAdapter)に切り替えます。")
            adapters.append(PSmilesAdapter())
            # Mordred等はモノマー近似したMolでも動く可能性があるが、安定性のため今回はPSmiles優先
        else:
            # 通常の低分子SMILES
            adapters = [
                RDKitAdapter(compute_fp=True),
                MordredAdapter(selected_only=True),
            ]
            
        desc_dfs: list[pd.DataFrame] = []
        for adapter in adapters:
            if adapter.is_available():
                try:
                    # アダプタのAPI差を吸収
                    if isinstance(adapter, RDKitAdapter):
                        # RDKitAdapter等 (新API)
                        res_df = adapter.calculate(smiles_list, selected_descriptors=self.selected_descriptors)
                    else:
                        # PSmilesAdapter 等 古い/共通アダプタ (.compute() が ComputationResult/DescriptorResult を返す)
                        res = getattr(adapter, "compute")(smiles_list)
                        res_df = res.descriptors
                        
                    desc_dfs.append(res_df)
                except Exception as e:
                    logger.warning(f"{adapter.name}: 計算エラー - {e}")
                    
        if not desc_dfs:
            return pd.DataFrame()
            
        X_chem = pd.concat(desc_dfs, axis=1)
        
        # 選択されている場合はフィルタリング (PSMILES時はスキップ)
        if self.selected_descriptors and not has_psmiles:
            valid = [c for c in self.selected_descriptors if c in X_chem.columns]
            if valid:
                X_chem = X_chem[valid]
                
        return X_chem

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SmilesDescriptorTransformer":
        """学習フェーズで記述子カラム名を記憶する。"""
        if self.smiles_col not in X.columns:
            raise ValueError(f"SMILES列 '{self.smiles_col}' がDataFrameに存在しません。")
        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)
        self._descriptor_cols = X_chem.columns.tolist()
        self._non_smiles_cols = [c for c in X.columns if c != self.smiles_col]
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """SMILES列を記述子列に置換したDataFrameを返す。"""
        if self.smiles_col not in X.columns:
            raise ValueError(f"SMILES列 '{self.smiles_col}' がDataFrameに存在しません。")
        smiles_list = X[self.smiles_col].tolist()
        X_chem = self._compute_descriptors(smiles_list)

        # 記述子カラムをfitで記憶した順序・列に揃える（推論時の列不一致を防ぐ）
        for col in self._descriptor_cols:
            if col not in X_chem.columns:
                X_chem[col] = 0.0
        X_chem = X_chem[self._descriptor_cols].reset_index(drop=True)

        # 非SMILES列と結合
        X_rest = X[self._non_smiles_cols].reset_index(drop=True)
        return pd.concat([X_rest, X_chem], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._non_smiles_cols + self._descriptor_cols)
