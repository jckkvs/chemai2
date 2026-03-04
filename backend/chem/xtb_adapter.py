"""
backend/chem/xtb_adapter.py

XTB (GFN2-xTB) による量子化学計算に基づく記述子生成アダプター（スタブ実装）。
"""
import logging
from typing import Any
import pandas as pd
from backend.chem.base import BaseChemAdapter, DescriptorResult

logger = logging.getLogger(__name__)

# XTBで計算できると想定する記述子
_XTB_DESCRIPTORS = {
    "HomoLumoGap": "HOMO-LUMOエネルギーギャップ（光吸収・反応性）",
    "HomoEnergy": "HOMOエネルギー",
    "LumoEnergy": "LUMOエネルギー",
    "IonizationPotential": "イオン化ポテンシャル",
    "ElectronAffinity": "電子親和力",
    "DipoleMoment": "双極子モーメント（極性の指標）",
    "Polarizability": "分極率（電場への応答性）",
    "MaxAbsorptionWavelength": "最大吸収波長 (sTDA)",
    "OscillatorStrength": "振動子強度",
    "Electrophilicity": "親電子性インデックス",
    "MinBDE": "最小結合解離エネルギー（熱安定性）",
    "PartialCharge_Acidic": "酸性プロトン/塩基性原子の最大部分電荷",
    "PartialCharges_Max": "最大部分電荷（分極の指標）",
}

class XTBAdapter(BaseChemAdapter):
    """
    XTB による量子化学計算記述子アダプタ（モック実装）。
    将来的には実際の xTB / xtb-python への連携を実装する。
    """
    @property
    def name(self) -> str:
        return "xtb"

    @property
    def description(self) -> str:
        return "XTB (GFN2-xTB) による量子化学的電子状態・エネルギー記述子の計算（モック実装）"

    def is_available(self) -> bool:
        # 現在未統合。将来的には xtb-python などのインポート可否で判定する
        return False

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()  # 未インストールの場合は例外を送出する

        # 本来の実装ロジックを書く位置。現在は到達しない。
        raise NotImplementedError("XTBAdapter: 実際のXTB計算は未統合です。")

    def get_descriptor_names(self) -> list[str]:
        return list(_XTB_DESCRIPTORS.keys())
