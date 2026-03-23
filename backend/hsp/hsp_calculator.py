# -*- coding: utf-8 -*-
"""
backend/hsp/hsp_calculator.py

Hansen Solubility Parameters (HSP) 計算モジュール。
HSPiPyラッパー + REDスコア計算。

Implements: HSP球体最適化・RED値・可視化
引用: Hansen, C.M. "Hansen Solubility Parameters: A User's Handbook", 2nd Ed., CRC Press, 2007

API:
    load_solvent_data(filepath) → 溶媒データ読み込み
    calculate_hsp_sphere() → HSP球体最適化
    calculate_red_value() → RED値計算
    plot_3d() / plot_2d() → 可視化

前提:
    pip install hspipy  # オプション依存
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_HSPIPY_AVAILABLE = False
try:
    import hspipy  # noqa: F401
    _HSPIPY_AVAILABLE = True
except ImportError:
    pass


def is_hspipy_available() -> bool:
    """HSPiPyが利用可能か。"""
    return _HSPIPY_AVAILABLE


class HSPCalculator:
    """HSPiPy を使用した Hansen Solubility Parameters 計算。

    Implements: HSP球体最適化（Hansen 2007, Ch.2）
    引用: Hansen, C.M. "Hansen Solubility Parameters", 2nd Ed., 2007

    RED < 1 → 溶解  /  RED > 1 → 不溶解
    Ra = √(4·ΔδD² + ΔδP² + ΔδH²)   ... 式(1)
    RED = Ra / R₀                      ... 式(2)
    """

    def __init__(self) -> None:
        self._hsp_obj: Any = None
        self._result: Any = None

    def load_solvent_data(self, filepath: str) -> None:
        """溶媒データ読み込み (CSV/HSD/HSDX)。"""
        if not _HSPIPY_AVAILABLE:
            raise ImportError(
                "HSPiPy が未インストールです。pip install hspipy を実行してください。"
            )
        from hspipy import HSP
        self._hsp_obj = HSP()
        self._hsp_obj.read(filepath)
        logger.info("HSP 溶媒データ読み込み: %s", filepath)

    def calculate_hsp_sphere(
        self,
        inside_limit: float = 1.0,
        n_spheres: int = 1,
    ) -> dict:
        """
        HSP 球体最適化計算。

        Implements: Hansen 2007, §2.3 球体最適化アルゴリズム

        Args:
            inside_limit: 球体内判定閾値 (デフォルト 1.0)
            n_spheres: 球体数 (1 or 2)

        Returns:
            dict: hsp=[δD,δP,δH], radius, accuracy, datafit, n_solvents_in/out, n_wrong_in/out
        """
        if self._hsp_obj is None:
            raise ValueError("溶媒データ未読み込み。load_solvent_data()を先に実行してください。")

        self._result = self._hsp_obj.get(
            inside_limit=inside_limit,
            n_spheres=n_spheres,
        )

        return {
            "hsp": self._result.hsp.tolist(),
            "radius": float(self._result.radius),
            "accuracy": float(self._result.accuracy),
            "datafit": float(self._result.datafit),
            "n_solvents_in": int(self._result.n_solvents_in),
            "n_solvents_out": int(self._result.n_solvents_out),
            "n_wrong_in": int(self._result.n_wrong_in),
            "n_wrong_out": int(self._result.n_wrong_out),
        }

    @staticmethod
    def calculate_red_value(
        solute_hsp: tuple[float, float, float],
        solvent_hsp: tuple[float, float, float],
        radius: float,
    ) -> float:
        """
        RED 値 (Relative Energy Difference) 計算。

        Implements: Hansen 2007, §2.2 式(1)(2)
        引用: Ra = √(4·ΔδD² + ΔδP² + ΔδH²)
              RED = Ra / R₀

        RED < 1 → 溶解 / RED > 1 → 不溶解

        Args:
            solute_hsp: 溶質の (δD, δP, δH)
            solvent_hsp: 溶媒の (δD, δP, δH)
            radius: 溶解性球体の半径 R₀
        """
        d_diff = solute_hsp[0] - solvent_hsp[0]
        p_diff = solute_hsp[1] - solvent_hsp[1]
        h_diff = solute_hsp[2] - solvent_hsp[2]

        # Hansenの距離式: 分散力は4倍の重み付け
        ra = float(np.sqrt(4 * d_diff**2 + p_diff**2 + h_diff**2))

        if radius <= 0:
            return float("inf")
        return ra / radius

    def plot_3d(self) -> Any:
        """3D プロット (HSPiPy)。matplotlibのfigを返す。"""
        if self._hsp_obj is None:
            raise ValueError("データ未読み込み")
        return self._hsp_obj.plot_3d()

    def plot_2d(self) -> Any:
        """2D プロット (HSPiPy)。matplotlibのfigを返す。"""
        if self._hsp_obj is None:
            raise ValueError("データ未読み込み")
        return self._hsp_obj.plot_2d()
