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

from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.mordred_adapter import MordredAdapter
from backend.chem.psmiles_adapter import PSmilesAdapter

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
        
        # Streamlit セッションからの事前計算結果の再利用（存在する場合）
        try:
            import streamlit as st
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                precalc_df = st.session_state.get("precalc_smiles_df")
                orig_df = st.session_state.get("df")
                orig_smiles_col = st.session_state.get("smiles_col")
                
                if precalc_df is not None and not precalc_df.empty and orig_df is not None and orig_smiles_col:
                    if not hasattr(st.session_state, "_smiles_precalc_dict"):
                        mapping = {}
                        for idx, smi in zip(orig_df.index, orig_df[orig_smiles_col]):
                            if pd.notna(smi) and idx in precalc_df.index:
                                mapping[str(smi)] = precalc_df.loc[idx]
                        st.session_state["_smiles_precalc_dict"] = mapping
                    
                    smi_dict = st.session_state["_smiles_precalc_dict"]
                    
                    hit_rows = []
                    all_hit = True
                    for smi in smiles_list:
                        smi_str = str(smi)
                        if smi_str in smi_dict:
                            hit_rows.append(smi_dict[smi_str])
                        else:
                            all_hit = False
                            break
                            
                    if all_hit and hit_rows:
                        logger.info("事前計算された記述子キャッシュを再利用します。")
                        cached_df = pd.DataFrame(hit_rows)
                        cached_df.index = range(len(cached_df))
                        
                        if self.selected_descriptors and not has_psmiles:
                            valid_cols = [c for c in self.selected_descriptors if c in cached_df.columns]
                            if valid_cols:
                                return cached_df[valid_cols]
                        
                        return cached_df
        except Exception as e:
            logger.debug(f"キャッシュ再利用スキップ: {e}")

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
                    # 全てのアダプタはBaseChemAdapterを継承し、compute()を実装している
                    res = adapter.compute(smiles_list)
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

        # 非SMILES列（記述子以外の生データ列）から、新しく計算する記述子と重複する名前を除去
        # これにより EDA フェーズで事前計算された列があっても二重定義にならずに済む
        # ※SMILES列自体はドロップせず、後続の ColumnTransformer 等で処理(除外)させるのが安全
        X_rest = X.reset_index(drop=True)
        X_rest = X_rest.drop(columns=[c for c in self._descriptor_cols if c in X_rest.columns], errors="ignore")
        
        return pd.concat([X_rest, X_chem], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._non_smiles_cols + self._descriptor_cols)

def progressive_precalculate(smiles_list: list[str], target_col_name: str = ""):
    """
    ユーザーの要求に応じ、優先順位をつけて事前計算を行い、進捗を yield するジェネレータ。
    Yields:
        (progress_float, status_message, current_df)
    """
    if not smiles_list:
        yield 1.0, "SMILES列が空です", pd.DataFrame()
        return

    # PSMILES check
    has_psmiles = any(PSmilesAdapter.is_psmiles(smi) for smi in smiles_list[:50] if isinstance(smi, str))
    
    if has_psmiles:
        yield 0.1, "PSMILES形式を検出しました。Polymer用モデルをロード中...", pd.DataFrame()
        adapter = PSmilesAdapter()
        try:
            res = adapter.compute(smiles_list)
            df_res = res.descriptors
            yield 1.0, "計算完了（PSMILES）", df_res
        except Exception as e:
            logger.error(f"PSMILES事前計算エラー: {e}")
            yield 1.0, f"エラー: {e}", pd.DataFrame()
        return

    # 通常のSMILES：3ステップで計算を完了させる
    from backend.chem.recommender import get_target_recommendation_by_name

    # --- ステップ1: 目的変数に対する推奨記述子 ---
    yield 0.3, f"目的変数「{target_col_name or '不明'}」に関連する推奨記述子を計算中...", pd.DataFrame()
    rec = get_target_recommendation_by_name(target_col_name)
    rec_names = [d.name for d in rec.descriptors] if rec else []

    rdkit_adapter = RDKitAdapter(compute_fp=False)
    df_result = pd.DataFrame(index=range(len(smiles_list)))

    if rec_names and rdkit_adapter.is_available():
        try:
            df_rd_rec = rdkit_adapter.compute(smiles_list, selected_descriptors=rec_names).descriptors
            df_result = pd.concat([df_result, df_rd_rec], axis=1)
        except Exception:
            pass
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    # --- ステップ2: 数え上げ系記述子 (is_count=True) ---
    yield 0.6, "数え上げ系記述子（原子数、環数等）を計算中...", df_result
    if rdkit_adapter.is_available():
        try:
            mdata = rdkit_adapter.get_descriptors_metadata()
            count_names = [m.name for m in mdata if m.is_count and m.name not in df_result.columns]
            if count_names:
                df_counts = rdkit_adapter.compute(smiles_list, selected_descriptors=count_names).descriptors
                df_result = pd.concat([df_result, df_counts], axis=1)
        except Exception:
            pass
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    # --- ステップ3: 意味のある主要記述子 (厳選12個) ---
    CURATED_DESCRIPTORS = [
        "MolWt", "LogP", "TPSA", "HBA", "HBD",
        "RotBonds", "RingCount", "AromaticRingCount",
        "FractionCSP3", "HeavyAtoms", "MolMR", "HallKierAlpha",
    ]
    yield 0.9, "主要な物理化学記述子（分子量・LogP・TPSA等）を計算中...", df_result

    curated = [c for c in CURATED_DESCRIPTORS if c not in df_result.columns]
    if curated and rdkit_adapter.is_available():
        try:
            df_curated = rdkit_adapter.compute(smiles_list, selected_descriptors=curated).descriptors
            df_result = pd.concat([df_result, df_curated], axis=1)
        except Exception:
            pass

    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
    yield 1.0, f"完了 — {len(df_result.columns)}個の主要記述子を抽出しました", df_result
