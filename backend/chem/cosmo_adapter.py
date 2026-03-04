"""
backend/chem/cosmo_adapter.py

COSMO-RS理論に基づく流体および熱力学的性質の予測による分子記述子の計算アダプタ（スタブ実装）。
"""
import logging
from typing import Any
import pandas as pd
from backend.chem.base import BaseChemAdapter, DescriptorResult

logger = logging.getLogger(__name__)

_COSMO_DESCRIPTORS = {
    "Density": "密度（COSMO-RS体積等からの推定）",
    "CohesiveEnergy": "凝集エネルギー（蒸発熱・分子間相互作用）",
    "CohesiveEnergyDensity": "凝集エネルギー密度 (CED)",
    "SolvationFreeEnergy": "水和自由エネルギー",
    "AqueousSolubility": "水溶性 (LogS)",
    "HSP_Dispersion": "ハンセン溶解度パラメータ(HSP) 分散力項",
    "HSP_Polar": "ハンセン溶解度パラメータ(HSP) 極性項",
    "HSP_Hbond": "ハンセン溶解度パラメータ(HSP) 水素結合項",
    "VanDerWaalsVolume": "Van der Waals体積(COSMO-RS)",
    "MolVolume": "モル体積",
}

class CosmoAdapter(BaseChemAdapter):
    @property
    def name(self) -> str:
        return "cosmo_rs"

    @property
    def description(self) -> str:
        return "COSMO-RSによる熱力学的・溶液論的記述子の計算（モック実装）"

    def is_available(self) -> bool:
        # COSMO-RS（COSMOtherm等）の利用可否を判定する。現在は未統合のためFalse
        return False

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()

        raise NotImplementedError("CosmoAdapter: 実際のCOSMO-RS計算は未統合です。")

    def get_descriptor_names(self) -> list[str]:
        return list(_COSMO_DESCRIPTORS.keys())
