"""
backend/chem/group_contrib_adapter.py

原子団寄与法（Bicerano, Van Krevelen法等）によるポリマーや有機材料物性の推定値の記述子化アダプタ（スタブ実装）。
"""
import logging
from typing import Any
import pandas as pd
from backend.chem.base import BaseChemAdapter, DescriptorResult

logger = logging.getLogger(__name__)

_GROUP_CONTRIB_DESCRIPTORS = {
    "EntanglementMW": "絡み合い分子量",
    "Tg_estimated": "ガラス転移温度の推測値",
    "CohesiveEnergy": "凝集エネルギー（Bicerano）",
    "CohesiveEnergyDensity": "凝集エネルギー密度（CED）",
    "BackboneRigidity": "主鎖の剛直性パラメータ",
    "BackboneFlexibility": "主鎖の柔軟性パラメーター",
    "VanDerWaalsVolume": "Van der Waals体積（Group Contribution）",
    "FreeVolume": "分子的自由体積",
    "ChainEntanglement": "分子鎖絡み合いパラメーター",
    "HLB": "親水性疎水性バランス (Griffin/Davies)",
    "CrosslinkDensity": "架橋点密度",
}

class GroupContribAdapter(BaseChemAdapter):
    @property
    def name(self) -> str:
        return "group_contrib"

    @property
    def description(self) -> str:
        return "原子団寄与法(Bicerano/Van Krevelen法)による高分子物性の記述子群（モック実装）"

    def is_available(self) -> bool:
        # 原子団寄与法のパーサー・計算エンジン（Bicerano等）のロード可否判定。現在未統合。
        return False

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()

        raise NotImplementedError("GroupContribAdapter: 実際の推定モデルは未統合です。")

    def get_descriptor_names(self) -> list[str]:
        return list(_GROUP_CONTRIB_DESCRIPTORS.keys())
