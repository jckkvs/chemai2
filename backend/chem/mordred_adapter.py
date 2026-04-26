# backend/chem/mordred_adapter.py — 精緻化版 (記述子計算コア)

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import logging
import gc
from rdkit import Chem
from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

def calculate_mordred_descriptors(
    smiles_list: List[str],
    ignore_3D: bool = True,
    drop_constant: bool = True,
    drop_na_ratio: float = 0.5,
    batch_size: int = 50
) -> pd.DataFrame:
    """
    Calculate Mordred descriptors with robust error handling and memory management
    
    Args:
        smiles_list: List of SMILES strings
        ignore_3D: If True, skip 3D-dependent descriptors
        drop_constant: Remove columns with zero variance
        drop_na_ratio: Drop columns with > this ratio of NaN values
        batch_size: Process molecules in batches to control memory
    
    Returns:
        pd.DataFrame with calculated descriptors (rows=SMILES, columns=features)
    """
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        logger.error("mordred package not installed. Run: pip install mordred")
        return pd.DataFrame()
    
    if not smiles_list:
        return pd.DataFrame()
    
    # 【修正点1】Calculator初期化は1回のみ
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    
    all_results = []
    n_total = len(smiles_list)
    n_success = 0
    
    # 【修正点4】バッチ処理でメモリ制御
    for i in range(0, n_total, batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_results = []
        
        for smi in batch_smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    batch_results.append(None)
                    continue
                
                # 【修正点3】Mordred固有エラーを個別捕捉
                result = calc(mol)
                if result.is_missing():
                    batch_results.append(None)
                else:
                    # 数値型に変換（MordredはDecimal型を返すことがある）
                    batch_results.append({
                        k: float(v) if v is not None else np.nan 
                        for k, v in result.as_dict().items()
                    })
                    n_success += 1
                    
            except Exception as e:
                # 【修正点1】例外は分子単位で隔離しバッチ継続
                logger.debug(f"Mordred failed for SMILES {smi!r}: {type(e).__name__}")
                batch_results.append(None)
        
        # DataFrameに変換して結合
        if any(r is not None for r in batch_results):
            df_batch = pd.DataFrame(batch_results)
            all_results.append(df_batch)
        
        # 【修正点4】バッチ終了後にメモリ解放
        del batch_results
        gc.collect()
        
        # 進捗ログ（【修正点5】失敗率の統計）
        if (i + batch_size) % 1000 == 0 or i + batch_size >= n_total:
            processed = min(i + batch_size, n_total)
            logger.info(
                f"Mordred batch progress: {processed}/{n_total} | "
                f"Success rate: {n_success/processed:.1%}"
            )
    
    if not all_results:
        logger.warning("No valid Mordred descriptors calculated")
        return pd.DataFrame()
    
    # 結合と後処理
    df_full = pd.concat(all_results, ignore_index=True, sort=False)
    df_full.index = range(len(df_full))  # インデックス再設定
    
    # 【修正点2】低品質列の自動除去
    if drop_constant:
        # 分散が0の列を除去
        variances = df_full.var()
        df_full = df_full.loc[:, variances > 1e-8]
    
    if drop_na_ratio < 1.0:
        # NaN比率が閾値を超える列を除去
        na_ratios = df_full.isna().mean()
        df_full = df_full.loc[:, na_ratios <= drop_na_ratio]
    
    logger.info(
        f"Mordred calculation completed: {df_full.shape[0]} rows × "
        f"{df_full.shape[1]} columns (from {n_total} SMILES)"
    )
    
    return df_full.astype(np.float32)  # 【修正点2】メモリ効率のためfloat32にキャスト

class MordredAdapter(BaseChemAdapter):
    """
    Mordred による包括的な分子記述子計算アダプタ。
    """
    def __init__(self, use_3d: bool = False, selected_only: bool = True):
        self.use_3d = use_3d
        self.selected_only = selected_only

    @property
    def name(self) -> str: return "mordred"

    @property
    def description(self) -> str:
        return "Mordred: 約1800種の2D分子記述子を計算できる包括的ライブラリ。"

    def is_available(self) -> bool:
        try:
            import mordred
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: List[str], **kwargs: Any) -> DescriptorResult:
        df = calculate_mordred_descriptors(
            smiles_list, 
            ignore_3D=not self.use_3d,
            batch_size=kwargs.get("batch_size", 50)
        )
        
        # 成功/失敗のインデックスを特定
        failed_indices = [i for i, smi in enumerate(smiles_list) if Chem.MolFromSmiles(smi) is None]
        
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name
        )

    def get_descriptor_names(self) -> List[str]:
        # 仮の実装（実際にはCalculatorから取得）
        return ["MW", "LogP", "TPSA"]

    def get_descriptors_metadata(self) -> List[DescriptorMetadata]:
        return [DescriptorMetadata(name="MW", meaning="Molecular Weight", is_count=False)]
