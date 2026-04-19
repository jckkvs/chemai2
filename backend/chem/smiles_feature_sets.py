"""
backend/chem/smiles_feature_sets.py

SMILES特徴量セット（SMILES Feature Set）の管理クラス。
複数の特徴量セットを並行して保持し、計算結果をキャッシュする。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.chem import ADAPTER_REGISTRY

logger = logging.getLogger(__name__)

@dataclass
class SMILESFeatureSet:
    """
    SMILESから生成される特徴量の集合体。
    1つの「セット」は複数のエンジン（RDKit, Mordred等）の組み合わせで構成される。
    """
    id: str
    name: str = "新規セット"
    enabled_engines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 計算結果キャッシュ
    dataframe: Optional[pd.DataFrame] = None
    success_rate: float = 0.0
    calculation_summary: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled_engines": self.enabled_engines,
            "success_rate": self.success_rate,
            "calculation_summary": self.calculation_summary,
            "has_data": self.dataframe is not None
        }

class SMILESFeatureSetManager:
    """
    複数の SMILESFeatureSet を管理する。
    """
    def __init__(self):
        self.sets: Dict[str, SMILESFeatureSet] = {}
        # デフォルトセットを作成
        self.create_set("set1", "基本記述子セット")
        self.sets["set1"].enabled_engines = {"RDKit": {}}

    def create_set(self, set_id: str, name: str) -> SMILESFeatureSet:
        if set_id in self.sets:
            logger.warning(f"Set ID '{set_id}' already exists. Overwriting.")
        new_set = SMILESFeatureSet(id=set_id, name=name)
        self.sets[set_id] = new_set
        return new_set

    def remove_set(self, set_id: str):
        if set_id in self.sets:
            del self.sets[set_id]

    def get_set(self, set_id: str) -> Optional[SMILESFeatureSet]:
        return self.sets.get(set_id)

    async def calculate_set(self, set_id: str, smiles_list: List[str]) -> Optional[pd.DataFrame]:
        """
        指定されたセットの全エンジンを実行し、結果をマージしてキャッシュする。
        """
        fset = self.get_set(set_id)
        if not fset:
            return None
            
        if not smiles_list:
            return None

        combined_dfs = []
        summary = {}
        total_failed = set()

        for engine_name, kwargs in fset.enabled_engines.items():
            adapter_cls = ADAPTER_REGISTRY.get(engine_name)
            if not adapter_cls:
                logger.error(f"Engine '{engine_name}' not found in registry.")
                continue
            
            try:
                adapter = adapter_cls(**kwargs)
                if not adapter.is_available():
                    logger.warning(f"Engine '{engine_name}' is not available.")
                    continue
                
                # 計算実行
                result = adapter.compute(smiles_list)
                df = result.descriptors
                
                # 列名に 接頭辞 (setid_) を追加して衝突を回避
                # ただしユーザーが分かりやすいように setid_engine_name という形式にする
                df.columns = [f"{set_id}_{engine_name}_{col}" for col in df.columns]
                
                combined_dfs.append(df)
                summary[engine_name] = df.shape[1]
                total_failed.update(result.failed_indices)
                
            except Exception as e:
                logger.error(f"Calculation failed for engine '{engine_name}': {e}")
                continue

        if not combined_dfs:
            fset.dataframe = None
            fset.success_rate = 0.0
            fset.calculation_summary = {}
            return None

        # 全エンジンの結果を横結合
        fset.dataframe = pd.concat(combined_dfs, axis=1)
        fset.calculation_summary = summary
        fset.success_rate = (len(smiles_list) - len(total_failed)) / len(smiles_list) if smiles_list else 0.0
        
        return fset.dataframe

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.sets.values()]
