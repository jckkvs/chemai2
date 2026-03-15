# -*- coding: utf-8 -*-
"""
backend/chem/padel_adapter.py

PaDEL-Descriptor アダプタ。
1800+ 分子記述子と 10 種フィンガープリントを計算する。
Java ベースの PaDEL-Descriptor を Python 経由で利用。

参考: padelpy (https://github.com/ecrl/padelpy)
  pip install padelpy
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)


class PaDELAdapter(BaseChemAdapter):
    """
    PaDEL-Descriptor アダプタ。

    1D/2D/3D 記述子と10種のフィンガープリント（MACCS, PubChem, SubstructureFP等）を計算。
    Java が必要（padelpy が内蔵のJARファイルを使用）。
    """

    def __init__(self, compute_fingerprints: bool = False, timeout: int = 120):
        """
        Args:
            compute_fingerprints: True でフィンガープリントも計算
            timeout: 1分子あたりのタイムアウト（秒）
        """
        self._compute_fp = compute_fingerprints
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "padel"

    @property
    def description(self) -> str:
        return (
            "PaDEL-Descriptor: 1800+ 分子記述子 + 10種フィンガープリント。"
            "pip install padelpy (Java Runtime 必要)"
        )

    def is_available(self) -> bool:
        try:
            import padelpy  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        from padelpy import from_smiles

        rows = []
        failed_indices = []

        for i, smi in enumerate(smiles_list):
            try:
                desc = from_smiles(smi, fingerprints=self._compute_fp, timeout=self._timeout)
                # desc は OrderedDict {name: value}
                row = {}
                for k, v in desc.items():
                    try:
                        row[f"PaDEL_{k}"] = float(v) if v not in ("", "Infinity", "-Infinity") else float("nan")
                    except (ValueError, TypeError):
                        row[f"PaDEL_{k}"] = float("nan")
                rows.append(row)
            except Exception as e:
                logger.warning(f"PaDEL: SMILES '{smi}' でエラー: {e}")
                failed_indices.append(i)
                rows.append({})

        descriptors = pd.DataFrame(rows)

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(name="PaDEL_MW", meaning="分子量", is_count=False),
            DescriptorMetadata(name="PaDEL_nAtom", meaning="原子数", is_count=True),
            DescriptorMetadata(name="PaDEL_TPSA", meaning="極性表面積", is_count=False),
        ]
