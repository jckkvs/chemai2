"""
backend/chem/unipka_adapter.py

Uni-pKa等の専用ライブラリを用いたpKa（酸解離定数）の特化計算アダプタ（スタブ実装）。
"""
import logging
from typing import Any
import pandas as pd
from backend.chem.base import BaseChemAdapter, DescriptorResult

logger = logging.getLogger(__name__)

_UNIPKA_DESCRIPTORS = {
    "pKa_pred": "Uni-pKa等のディープラーニングモデルによる予測pKa値",
}

class UniPkaAdapter(BaseChemAdapter):
    @property
    def name(self) -> str:
        return "unipka"

    @property
    def description(self) -> str:
        return "Uni-pKaによる高精度pKa予測（モック実装）"

    def is_available(self) -> bool:
        # Uni-pKa モデルのロード可否を判定する。現在は未統合。
        return False

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()

        raise NotImplementedError("UniPkaAdapter: 実際のモデル予測は未統合です。")

    def get_descriptor_names(self) -> list[str]:
        return list(_UNIPKA_DESCRIPTORS.keys())
