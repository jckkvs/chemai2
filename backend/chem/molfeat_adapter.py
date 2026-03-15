# -*- coding: utf-8 -*-
"""
backend/chem/molfeat_adapter.py

Molfeat アダプタ。
Datamol/Valence Discovery が開発した統合的な分子特徴量フレームワーク。
多数のフィンガープリント/記述子を統一APIで提供。

参考: molfeat (https://github.com/datamol-io/molfeat)
  pip install molfeat
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# 利用可能な計算機タイプ
_CALCULATOR_TYPES = {
    "ecfp": {"kind": "ecfp", "n_bits": 2048},
    "fcfp": {"kind": "fcfp", "n_bits": 2048},
    "maccs": {"kind": "maccs"},
    "topological": {"kind": "topological", "n_bits": 2048},
    "avalon": {"kind": "avalon", "n_bits": 512},
    "atompair": {"kind": "atompair", "n_bits": 2048},
    "rdkit": {"kind": "rdkit", "n_bits": 2048},
    "desc2d": {"kind": "desc2D"},
    "desc3d": {"kind": "desc3D"},
    "cats": {"kind": "cats"},
    "scaffoldkeys": {"kind": "scaffoldkeys"},
    "skeys": {"kind": "skeys"},
    "electroshape": {"kind": "electroshape"},
    "usr": {"kind": "usr"},
    "usrcat": {"kind": "usrcat"},
    "pharm2d": {"kind": "Pharmacophore2D"},
}


class MolfeatAdapter(BaseChemAdapter):
    """
    Molfeat アダプタ。

    統合的な分子特徴量フレームワーク。FP/2D記述子/3D記述子/薬理学的特徴等を
    統一インターフェースで計算。
    """

    def __init__(self, calculator_type: str = "ecfp", **calc_kwargs):
        """
        Args:
            calculator_type: 計算機タイプ（ecfp, maccs, desc2d, desc3d等）
            **calc_kwargs: 計算機に渡す追加パラメータ
        """
        self._calculator_type = calculator_type
        self._calc_kwargs = calc_kwargs

    @property
    def name(self) -> str:
        return "molfeat"

    @property
    def description(self) -> str:
        return (
            "Molfeat: Datamol統合分子特徴量フレームワーク。"
            "FP/2D記述子/3D記述子/薬理学的特徴等を統一APIで計算。"
            "pip install molfeat"
        )

    def is_available(self) -> bool:
        try:
            import molfeat  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        from molfeat.calc import FPCalculator, RDKitDescriptors2D
        from molfeat.trans import MoleculeTransformer

        n = len(smiles_list)
        failed_indices = []

        try:
            calc_config = _CALCULATOR_TYPES.get(self._calculator_type, {"kind": self._calculator_type})
            config = {**calc_config, **self._calc_kwargs}
            kind = config.pop("kind", self._calculator_type)

            if kind in ("desc2D", "desc3D"):
                calc = RDKitDescriptors2D()
            else:
                calc = FPCalculator(kind, **config)

            transformer = MoleculeTransformer(calc)
            features = transformer.transform(smiles_list)

            if isinstance(features, np.ndarray):
                n_cols = features.shape[1]
                col_names = [f"molfeat_{kind}_{j}" for j in range(n_cols)]
                descriptors = pd.DataFrame(features, columns=col_names)

                # NaN行を失敗として記録
                for i in range(n):
                    if np.all(np.isnan(features[i])):
                        failed_indices.append(i)
            else:
                descriptors = pd.DataFrame(index=range(n))
                failed_indices = list(range(n))

        except Exception as e:
            logger.error(f"Molfeat 計算エラー: {e}")
            descriptors = pd.DataFrame(index=range(n))
            failed_indices = list(range(n))

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(
                name=f"molfeat_{self._calculator_type}_0",
                meaning=f"Molfeat {self._calculator_type} feature 0",
                is_count=False,
            ),
        ]
