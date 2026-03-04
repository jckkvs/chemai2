"""
backend/chem/mordred_adapter.py

Mordred を使った化合物記述子計算アダプタ。
RDKit上に構築され、約1800種の2D/3D記述子を計算できる。
QSAR/QSPR研究で広く使用されている包括的な記述子セット。

GitHub: https://github.com/mordred-descriptor/mordred
論文: Moriwaki et al., J Cheminform. 10:4 (2018)
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


class MordredAdapter(BaseChemAdapter):
    """
    Mordred による包括的な分子記述子計算アダプタ。

    Mordredは~1800種の2D/3D記述子を計算できるライブラリで、
    QSAR/QSPR（定量的構造活性/物性相関）研究で広く使われる。

    RDKitの上に構築されているため、RDKitが必要。
    インストール: pip install mordred

    Args:
        use_3d: 3D記述子も計算するか（デフォルト: False、高速化のため2Dのみ）
        ignore_3d:  3D座標がない場合の2Dフォールバックを許可するか
        max_descriptors: 上位N個のみ使用（None=全部）
    """

    # 有機化学/ポリマー物性予測に使いやすい主要な2D記述子のサブセット
    # (Mordredは約1800種あるため、デフォルトではこの厳選セットを返す)
    SELECTED_DESCRIPTORS: list[str] = [
        # 分子全体の形状・サイズ
        "MW",           "nHeavyAtom",   "nAtom",        "nBonds",
        "nBondsO",      "nBondsS",
        # 環構造
        "nRing",        "nHRing",       "nARing",       "nBRing",
        "nFARing",      "nFHRing",      "nFRing",       "nSpiro",
        "nBridgehead",
        # 原子タイプ
        "nC",           "nN",           "nO",           "nS",
        "nF",           "nCl",          "nBr",          "nI",
        "nHet",         "nHetero",
        # 水素結合
        "nHBAcc",       "nHBDon",       "nHBAcc_Lipin", "nHBDon_Lipin",
        # 電荷・極性
        "TPSA",
        # 疎水性
        "LogP",         "SLogP",
        # 位相的形状記述子（複雑性）
        "BertzCT",      "TopoPSA",
        # Wiener / Randic系トポロジー
        "WPath",        "WPol",         "Lop",
        # 電子状態 (2D)
        "PEOE_VSA1",    "PEOE_VSA2",    "PEOE_VSA3",    "PEOE_VSA4",
        "PEOE_VSA5",    "PEOE_VSA6",
        "SMR_VSA1",     "SMR_VSA2",     "SMR_VSA3",
        "SlogP_VSA1",   "SlogP_VSA2",   "SlogP_VSA3",
        # Kappa形状指数
        "Kier1",        "Kier2",        "Kier3",        "KierFlex",
        # 情報理論的記述子
        "IC0",          "IC1",          "IC2",          "TIC0",
        # 分子電子グラフ
        "EState_VSA1",  "EState_VSA2",  "EState_VSA3",
        "MaxEStateIndex","MinEStateIndex","MaxAbsEStateIndex",
        # BCUT記述子
        "BCUTc-1h",     "BCUTc-1l",     "BCUTdv-1h",    "BCUTdv-1l",
        # ABCガスケ/ガウジエ
        "AXp-0dv",      "AXp-1dv",      "AXp-2dv",
        # 接触面積
        "LabuteASA",
        # フラグメントベース
        "SIC0",         "SIC1",         "CIC0",         "CIC1",
        # その他
        "Ipc",          "BalabanJ",     "nRotB",        "RotRatio",
        "FragCpx",
    ]

    def __init__(
        self,
        use_3d: bool = False,
        selected_only: bool = True,
    ):
        self.use_3d = use_3d
        self.selected_only = selected_only
        self._descriptor_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "mordred"

    @property
    def description(self) -> str:
        return (
            "Mordred: 約1800種の2D分子記述子を計算できる包括的ライブラリ。"
            "QSAR/QSPR研究で広く使用。RDKit上に構築。"
            "参考: Moriwaki et al., J Cheminform. 10:4 (2018)"
        )

    def is_available(self) -> bool:
        try:
            import mordred  # noqa: F401
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_calculator(self):
        """Mordredのcalculatorを構築する（遅延初期化）"""
        from mordred import Calculator, descriptors as mordred_desc
        calc = Calculator(mordred_desc, ignore_3D=not self.use_3d)
        return calc

    def _get_all_mordred_names(self) -> list[str]:
        """Mordredで計算できる全記述子名を返す"""
        if self._descriptor_names is None:
            calc = self._build_calculator()
            self._descriptor_names = [str(d) for d in calc.descriptors]
        return self._descriptor_names

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        """
        SMILES文字列のリストからMordred記述子を計算する。

        Args:
            smiles_list: 入力SMILESのリスト
            **kwargs: 追加オプション（ignore_errors など）

        Returns:
            DescriptorResult インスタンス
        """
        self._require_available()

        from rdkit import Chem
        from mordred import Calculator, descriptors as mordred_desc

        calc = Calculator(mordred_desc, ignore_3D=not self.use_3d)

        mols = []
        failed_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                logger.warning(f"MordredAdapter: SMILES parse failed at index {i}: {smi!r}")
                failed_indices.append(i)
                mols.append(None)
            else:
                mols.append(mol)

        try:
            # Mordredは失敗分子をNaNとして処理できる
            df_all = calc.pandas([m for m in mols if m is not None])
        except Exception as e:
            logger.error(f"MordredAdapter: 計算中にエラーが発生しました: {e}")
            # フォールバック: 全てNaN
            desc_names = self.get_descriptor_names()
            df_all = pd.DataFrame(
                index=range(len(smiles_list) - len(failed_indices)),
                columns=desc_names,
                dtype=float
            )

        # Mordredの数値以外（エラーオブジェクト等）をNaNに変換
        df_all = df_all.apply(pd.to_numeric, errors="coerce")

        # NaN・無限大を0で補完（機械学習に渡すため）
        df_all = df_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 必要に応じてサブセットのみに絞る
        if self.selected_only:
            available_cols = [c for c in self.SELECTED_DESCRIPTORS if c in df_all.columns]
            df_all = df_all[available_cols] if available_cols else df_all

        # 失敗した行を0行で挿入（全体と行数を一致させる）
        df_result = pd.DataFrame(
            index=range(len(smiles_list)),
            columns=df_all.columns,
            dtype=float
        ).fillna(0.0)

        valid_idx = [i for i in range(len(smiles_list)) if i not in failed_indices]
        if len(df_all) == len(valid_idx):
            df_all.index = valid_idx
            # 明示的にfloat型にキャストしてFutureWarningを防止
            df_all_float = df_all.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            df_result.loc[valid_idx] = df_all_float.values

        return DescriptorResult(
            descriptors=df_result,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={
                "use_3d": self.use_3d,
                "n_descriptors": df_result.shape[1],
                "selected_only": self.selected_only,
            }
        )

    def get_descriptor_names(self) -> list[str]:
        """計算可能な記述子名のリストを返す。"""
        if self.selected_only:
            # インストールチェックしてから実際に存在するものだけ返す
            if self.is_available():
                try:
                    all_names = self._get_all_mordred_names()
                    return [n for n in self.SELECTED_DESCRIPTORS if n in all_names]
                except Exception:
                    pass
            return self.SELECTED_DESCRIPTORS
        else:
            if self.is_available():
                return self._get_all_mordred_names()
            return self.SELECTED_DESCRIPTORS
