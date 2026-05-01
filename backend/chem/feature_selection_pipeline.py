"""
backend/chem/feature_selection_pipeline.py

特徴量選択の自動化パイプライン。
目的変数に基づいて、相関分析・LLM推奨・タスクベース推奨を
自動で実行し、最適な特徴量セットを決定する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """パイプライン設定。"""
    task_type: str = "general"
    target_column: str = ""
    n_molecules: int = 100
    max_time_per_mol_s: float = 120.0
    xtb_available: bool = True
    top_n_features: int | None = None
    min_correlation: float = 0.0
    use_correlation: bool = True
    use_llm: bool = False
    llm_provider: Any | None = None
    force_include: list[str] | None = None
    force_exclude: list[str] | None = None


@dataclass
class PipelineResult:
    """パイプライン実行結果。"""
    selected_features: list[str]
    correlation_rankings: list[dict] = field(default_factory=list)
    llm_recommendations: list[str] = field(default_factory=list)
    llm_reasoning: str = ""
    llm_confidence: float = 0.0
    estimated_time_per_mol_s: float = 0.0
    estimated_total_minutes: float = 0.0
    notes: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str = ""


class FeatureSelectionPipeline:
    """
    特徴量選択を自動化するパイプライン。

    使い方::

        config = PipelineConfig(
            task_type="solubility",
            target_column="logS",
            use_correlation=True,
            use_llm=True,
            llm_provider=my_llm_provider,
        )
        pipeline = FeatureSelectionPipeline(config)
        result = pipeline.run(df=df_with_features)
        print(result.selected_features)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._selector = None
        self._llm_advisor = None

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """
        パイプラインを実行する。

        Args:
            df: 特徴量と目的変数を含むDataFrame。

        Returns:
            PipelineResult。
        """
        notes: list[str] = []
        correlation_rankings: list[dict] = []
        llm_recommendations: list[str] = []
        llm_reasoning = ""
        llm_confidence = 0.0

        # 1. 相関分析
        if self.config.use_correlation and self.config.target_column:
            try:
                from backend.chem.correlation_selector import (
                    CorrelationBasedSelector, CorrelationMethod,
                )

                corr_selector = CorrelationBasedSelector(
                    method=CorrelationMethod.PEARSON,
                    min_correlation=self.config.min_correlation,
                )
                corr_result = corr_selector.compute_correlations(
                    df, target_column=self.config.target_column,
                )
                correlation_rankings = [
                    {
                        "feature_name": r.feature_name,
                        "correlation": r.correlation,
                        "abs_correlation": r.abs_correlation,
                        "p_value": r.p_value,
                    }
                    for r in corr_result.rankings
                ]
                notes.append(
                    f"相関分析完了: {len(corr_result.rankings)}件の特徴量"
                )
            except Exception as e:
                notes.append(f"相関分析エラー: {e}")
                logger.error("Correlation analysis failed", exc_info=True)

        # 2. LLM推奨
        if self.config.use_llm:
            if self.config.llm_provider:
                try:
                    from backend.chem.llm_feature_advisor import (
                        LLMFeatureAdvisor,
                    )

                    advisor = LLMFeatureAdvisor(
                        provider=self.config.llm_provider,
                    )
                    rec = advisor.recommend(
                        df=df,
                        target_column=self.config.target_column,
                        task_type=self.config.task_type,
                        n_features=self.config.top_n_features or 10,
                    )
                    llm_recommendations = rec.feature_names
                    llm_reasoning = rec.reasoning
                    llm_confidence = rec.confidence
                    notes.append(
                        f"LLM推奨完了: {len(rec.feature_names)}件"
                        + (f" (確信度: {rec.confidence:.1%})" if rec.confidence else "")
                    )
                except Exception as e:
                    notes.append(f"LLM推奨エラー: {e}")
                    logger.error("LLM recommendation failed", exc_info=True)
            else:
                notes.append("LLM使用指定だがプロバイダーが未設定")

        # 3. 統合選択 (AdaptiveFeatureSelector)
        try:
            from backend.chem.adaptive_feature_selector import (
                AdaptiveFeatureSelector,
            )

            selector = AdaptiveFeatureSelector()
            llm_advisor_instance = None
            if self.config.use_llm and self.config.llm_provider:
                from backend.chem.llm_feature_advisor import (
                    LLMFeatureAdvisor,
                )
                llm_advisor_instance = LLMFeatureAdvisor(
                    provider=self.config.llm_provider,
                )

            result = selector.select_with_correlation(
                df=df,
                target_column=self.config.target_column,
                task_type=self.config.task_type,
                n_molecules=self.config.n_molecules,
                max_time_per_mol_s=self.config.max_time_per_mol_s,
                xtb_available=self.config.xtb_available,
                top_n_features=self.config.top_n_features,
                force_include=self.config.force_include,
                force_exclude=self.config.force_exclude,
                llm_advisor=llm_advisor_instance,
            )

            return PipelineResult(
                selected_features=result.selected_features,
                correlation_rankings=correlation_rankings,
                llm_recommendations=llm_recommendations,
                llm_reasoning=llm_reasoning,
                llm_confidence=llm_confidence,
                estimated_time_per_mol_s=result.estimated_time_per_mol_s,
                estimated_total_minutes=result.estimated_total_minutes,
                notes=notes + result.notes,
                success=True,
            )

        except Exception as e:
            logger.error("Pipeline failed", exc_info=True)
            return PipelineResult(
                selected_features=[],
                correlation_rankings=correlation_rankings,
                llm_recommendations=llm_recommendations,
                llm_reasoning=llm_reasoning,
                llm_confidence=llm_confidence,
                notes=notes,
                success=False,
                error_message=str(e),
            )
