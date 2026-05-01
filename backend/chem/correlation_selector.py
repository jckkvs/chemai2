"""
backend/chem/correlation_selector.py

相関係数ベースの特徴量選択エンジン。
計算済みの特徴量と目的変数の相関を計算し、
相関係数の高い順に特徴量を選択する。

既存の AdaptiveFeatureSelector（コスト・タスクベース）を補完する位置付け。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """相関係数の計算方法。"""
    PEARSON = auto()
    SPEARMAN = auto()
    KENDALL = auto()


@dataclass
class FeatureRanking:
    """1つの特徴量の相関ランキング結果。"""
    feature_name: str
    correlation: float        # 相関係数（-1 〜 1）
    abs_correlation: float    # 絶対値
    p_value: float            # 有意確率
    n_samples: int            # 有効サンプル数

    def __lt__(self, other: FeatureRanking) -> bool:
        return self.abs_correlation < other.abs_correlation


@dataclass
class CorrelationResult:
    """相関分析の結果全体。"""
    target_column: str
    method: CorrelationMethod
    rankings: list[FeatureRanking] = field(default_factory=list)
    n_samples: int = 0

    def get_top_n(self, n: int) -> list[FeatureRanking]:
        """相関係数の絶対値が高い上位n件を返す。"""
        sorted_rankings = sorted(self.rankings, key=lambda r: r.abs_correlation, reverse=True)
        return sorted_rankings[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """結果をDataFrameとして返す（UI表示・デバッグ用）。"""
        return pd.DataFrame([
            {
                "feature_name": r.feature_name,
                "correlation": r.correlation,
                "abs_correlation": r.abs_correlation,
                "p_value": r.p_value,
                "n_samples": r.n_samples,
                "significant": r.p_value < 0.05,
            }
            for r in sorted(self.rankings, key=lambda r: r.abs_correlation, reverse=True)
        ])


class CorrelationBasedSelector:
    """
    目的変数との相関係数に基づいて特徴量を選択するエンジン。

    使い方::

        selector = CorrelationBasedSelector()
        result = selector.compute_correlations(df, target_column="logS")
        top_features = selector.select_top_features(result, n_features=10)
        # → ["feature_1", "feature_4", ...]
    """

    def __init__(
        self,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        min_correlation: float = 0.0,
        max_features: int | None = None,
    ) -> None:
        self.method = method
        self.min_correlation = min_correlation
        self.max_features = max_features

    def compute_correlations(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> CorrelationResult:
        """
        全特徴量と目的変数の相関係数を計算する。

        Args:
            df: 特徴量と目的変数を含むDataFrame。
            target_column: 目的変数の列名。

        Returns:
            CorrelationResult。

        Raises:
            ValueError: 目的変数が存在しない場合。
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        target_series = df[target_column]
        if not pd.api.types.is_numeric_dtype(target_series):
            raise ValueError(f"Target column '{target_column}' must be numeric")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target_column]

        rankings: list[FeatureRanking] = []
        n_samples = len(df)

        for col in feature_cols:
            feature_series = df[col]
            # 有効なペアを取得（両方ともNaNでない）
            valid_mask = feature_series.notna() & target_series.notna()
            valid_feature = feature_series[valid_mask]
            valid_target = target_series[valid_mask]

            if len(valid_feature) < 3:
                logger.debug(f"Skipping '{col}': insufficient valid pairs ({len(valid_feature)})")
                continue

            try:
                corr, p_val = self._compute_pairwise(
                    valid_feature, valid_target, len(valid_feature)
                )
                ranking = FeatureRanking(
                    feature_name=col,
                    correlation=corr,
                    abs_correlation=abs(corr),
                    p_value=p_val,
                    n_samples=len(valid_feature),
                )
                rankings.append(ranking)
            except Exception as e:
                logger.debug(f"Failed to compute correlation for '{col}': {e}")
                continue

        rankings.sort(key=lambda r: r.abs_correlation, reverse=True)

        return CorrelationResult(
            target_column=target_column,
            method=self.method,
            rankings=rankings,
            n_samples=n_samples,
        )

    def _compute_pairwise(
        self,
        x: pd.Series,
        y: pd.Series,
        n: int,
    ) -> tuple[float, float]:
        """2つの系列の相関係数とp値を計算する。"""
        if self.method == CorrelationMethod.PEARSON:
            return stats.pearsonr(x, y)
        elif self.method == CorrelationMethod.SPEARMAN:
            return stats.spearmanr(x, y)
        elif self.method == CorrelationMethod.KENDALL:
            return stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")

    def select_top_features(
        self,
        result: CorrelationResult,
        n_features: int | None = None,
    ) -> list[str]:
        """
        相関係数の絶対値が高い順に特徴量を選択する。

        Args:
            result: compute_correlations() の結果。
            n_features: 選択する特徴量の数。Noneの場合は max_features を使用。

        Returns:
            選択された特徴量名のリスト。
        """
        n = n_features if n_features is not None else (self.max_features or len(result.rankings))
        top_rankings = result.get_top_n(n)

        selected = []
        for r in top_rankings:
            if self.min_correlation > 0 and r.abs_correlation < self.min_correlation:
                break
            selected.append(r.feature_name)
        return selected

    def select_by_threshold(
        self,
        result: CorrelationResult,
        threshold: float = 0.3,
    ) -> list[str]:
        """
        相関係数の絶対値が閾値以上の特徴量を選択する。

        Args:
            result: compute_correlations() の結果。
            threshold: 最小相関係数（絶対値）。

        Returns:
            選択された特徴量名のリスト。
        """
        selected = []
        for r in result.rankings:
            if r.abs_correlation >= threshold:
                selected.append(r.feature_name)
        return selected
