"""
backend/chem/rdkit_adapter.py

RDKit を使った化合物特徴量化アダプタ。
物理化学的性質 + Morgan/RDKit フィンガープリント + 位相的記述子を計算する。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorResult
from backend.utils.optional_import import safe_import

_rdkit = safe_import("rdkit", "rdkit")
logger = logging.getLogger(__name__)


# 計算する物理化学記述子の定義 {カラム名: (関数名, 説明)}
_PHYSICOCHEMICAL_DESCRIPTORS = {
    "MolWt": ("MolWt", "分子量"),
    "LogP": ("MolLogP", "LogP（脂溶性）"),
    "HBA": ("NumHAcceptors", "水素結合受容体数"),
    "HBD": ("NumHDonors", "水素結合供与体数"),
    "TPSA": ("TPSA", "位相的極性表面積"),
    "RotBonds": ("NumRotatableBonds", "回転可能結合数"),
    "RingCount": ("RingCount", "環数"),
    "AromaticRingCount": ("NumAromaticRings", "芳香環数"),
    "FractionCSP3": ("FractionCSP3", "sp3炭素割合"),
    "HeavyAtoms": ("HeavyAtomCount", "重原子数"),
    "MolMR": ("MolMR", "モル屈折"),
    "HallKierAlpha": ("HallKierAlpha", "Hall-Kierアルファ"),
}


class RDKitAdapter(BaseChemAdapter):
    """
    RDKit による化合物記述子計算アダプタ。

    計算内容:
    - 物理化学的記述子（MolWt, LogP, HBA, HBD, TPSA, RotBonds等）
    - Morgan フィンガープリント（ECFP4: radius=2）
    - RDKit フィンガープリント（2048 bit）
    - MACCS Keys

    Args:
        compute_fp: フィンガープリントも計算するか
        morgan_radius: Morgan FPの半径（デフォルト2 = ECFP4）
        morgan_bits: Morgan FPのビット数
        rdkit_fp_bits: RDKit FPのビット数
        include_maccs: MACCS keysを含めるか
    """

    def __init__(
        self,
        compute_fp: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        rdkit_fp_bits: int = 2048,
        include_maccs: bool = False,
    ) -> None:
        self.compute_fp = compute_fp
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.rdkit_fp_bits = rdkit_fp_bits
        self.include_maccs = include_maccs

    @property
    def name(self) -> str:
        return "rdkit"

    @property
    def description(self) -> str:
        return "RDKit による物理化学記述子・フィンガープリント計算"

    def is_available(self) -> bool:
        return bool(_rdkit)

    def compute(
        self,
        smiles_list: list[str],
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES リストから RDKit 記述子を計算する。

        Args:
            smiles_list: 入力SMILESのリスト
            **kwargs: 未使用（互換性のために受け取る）

        Returns:
            DescriptorResult インスタンス
        """
        self._require_available()

        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys  # type: ignore

        rows: list[dict[str, float]] = []
        failed: list[int] = []

        for idx, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"無効なSMILES: {smi!r}")

                row: dict[str, float] = {}

                # 物理化学記述子
                for col, (fn_name, _) in _PHYSICOCHEMICAL_DESCRIPTORS.items():
                    fn = getattr(Descriptors, fn_name, None)
                    if fn is None:
                        fn = getattr(rdMolDescriptors, fn_name, None)
                    if fn is not None:
                        row[col] = float(fn(mol))

                # Morgan フィンガープリント (ECFP4)
                if self.compute_fp:
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.morgan_radius, nBits=self.morgan_bits
                    )
                    for j, bit in enumerate(morgan_fp):
                        row[f"Morgan_r{self.morgan_radius}_{j}"] = float(bit)

                    # RDKit トポロジカルフィンガープリント
                    rdkit_fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_fp_bits)
                    for j, bit in enumerate(rdkit_fp):
                        row[f"RDKitFP_{j}"] = float(bit)

                    # MACCS Keys (166 bit)
                    if self.include_maccs:
                        maccs = MACCSkeys.GenMACCSKeys(mol)
                        for j, bit in enumerate(maccs):
                            row[f"MACCS_{j}"] = float(bit)

                rows.append(row)

            except Exception as e:
                logger.warning(f"RDKit: index={idx}, SMILES={smi!r}: {e}")
                failed.append(idx)
                rows.append({})

        df = pd.DataFrame(rows).fillna(0.0)
        logger.info(
            f"RDKit計算完了: {len(smiles_list)}件 / "
            f"失敗={len(failed)} / 記述子={df.shape[1]}"
        )
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
            metadata={
                "morgan_radius": self.morgan_radius,
                "morgan_bits": self.morgan_bits if self.compute_fp else 0,
            },
        )

    def get_descriptor_names(self) -> list[str]:
        """利用可能な記述子名のリストを返す。"""
        names = list(_PHYSICOCHEMICAL_DESCRIPTORS.keys())
        if self.compute_fp:
            names += [f"Morgan_r{self.morgan_radius}_{j}" for j in range(self.morgan_bits)]
            names += [f"RDKitFP_{j}" for j in range(self.rdkit_fp_bits)]
            if self.include_maccs:
                names += [f"MACCS_{j}" for j in range(167)]
        return names
