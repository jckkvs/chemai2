"""
backend/chem/llm_feature_advisor.py

LLMを活用した特徴量推奨エンジン。
目的変数名・タスク種別・データ特性から、
LLMが最適な特徴量を推奨する。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMRecommendation:
    """LLMによる特徴量推奨結果。"""
    feature_names: list[str]
    confidence: float  # 0.0 ~ 1.0
    reasoning: str = ""
    raw_response: str = ""


class LLMFeatureAdvisor:
    """
    LLMを使って目的変数に適した特徴量を推奨する。

    使い方::

        provider = SomeLLMProvider()
        advisor = LLMFeatureAdvisor(provider=provider)
        rec = advisor.recommend(
            df=df,
            target_column="logS",
            task_type="solubility",
        )
        print(rec.feature_names)
        # → ["rdkit_2d", "morgan_fp", "xtb_sp", ...]
    """

    def __init__(
        self,
        provider: Any | None = None,
        available_features: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.available_features = available_features

    def recommend(
        self,
        df: Any | None = None,
        target_column: str = "",
        task_type: str = "general",
        n_features: int = 10,
    ) -> LLMRecommendation:
        """
        LLMに問い合わせて特徴量を推奨させる。

        Args:
            df: データフレーム（オプション、統計情報を渡すため）。
            target_column: 目的変数名。
            task_type: タスク種別。
            n_features: 推奨する特徴量の数。

        Returns:
            LLMRecommendation。
        """
        if self.provider is None:
            logger.warning("LLM provider is not set. Returning empty recommendation.")
            return LLMRecommendation(feature_names=[], confidence=0.0)

        prompt = self._build_prompt(
            target_column=target_column,
            task_type=task_type,
            n_features=n_features,
            available_features=self.available_features,
        )

        try:
            response = self.provider.query(prompt)
            feature_names = self._parse_feature_names(response)
            confidence = self._estimate_confidence(response, feature_names)
            reasoning = self._extract_reasoning(response)

            return LLMRecommendation(
                feature_names=feature_names,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return LLMRecommendation(feature_names=[], confidence=0.0)

    def _build_prompt(
        self,
        target_column: str,
        task_type: str,
        n_features: int,
        available_features: list[str] | None,
    ) -> str:
        """LLMへのプロンプトを構築する。"""
        prompt = f"""You are a chemistry machine learning expert.

Task: Recommend the best molecular descriptors/features for predicting "{target_column}" (task type: {task_type}).

Please select the top {n_features} most relevant features from the following list.
Focus on features that have strong theoretical correlation with the target.

"""
        if available_features:
            prompt += "Available features:\n"
            for f in available_features:
                prompt += f"- {f}\n"
        else:
            prompt += (
                "Available features: rdkit_2d, morgan_fp, maccs_keys, "
                "xtb_sp, xtb_opt, xtb_ml_derived, 3d_geometry, "
                "vibrational, fukui_approx, conformer_ensemble\n"
            )

        prompt += """
Respond in the following format:
FEATURES: feature1, feature2, feature3, ...
REASONING: <brief explanation of why these features are relevant>
"""
        return prompt

    def _parse_feature_names(self, text: str) -> list[str]:
        """
        LLMの応答テキストから特徴量名を抽出する。
        形式: "FEATURES: rdkit_2d, morgan_fp, xtb_sp"
        """
        feature_names = []

        # "FEATURES:" セクションを探す
        match = re.search(
            r"FEATURES:\s*(.+?)(?:\nREASONING:|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            features_text = match.group(1).strip()
        else:
            # "I recommend: ..." パターンも試す
            match = re.search(
                r"(?:recommend|features?):\s*(.+?)(?:\n|$)",
                text,
                re.IGNORECASE,
            )
            if match:
                features_text = match.group(1).strip()
            else:
                features_text = text

        # カンマ区切りでパース
        raw_names = re.split(r"[,;\n]+", features_text)
        for name in raw_names:
            cleaned = re.sub(r"[^a-z0-9_]", "", name.lower().strip())
            if cleaned and len(cleaned) > 2:
                feature_names.append(cleaned)

        return feature_names

    def _extract_reasoning(self, text: str) -> str:
        """応答から推論理由を抽出する。"""
        match = re.search(
            r"REASONING:\s*(.+?)(?:\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _estimate_confidence(
        self,
        response: str,
        feature_names: list[str],
    ) -> float:
        """推奨の確信度を推定する（簡易的）。"""
        if not feature_names:
            return 0.0
        # 応答が構造化されているほど高い確信度
        has_structure = "FEATURES:" in response and "REASONING:" in response
        confidence = 0.5
        if has_structure:
            confidence += 0.3
        # 多くの特徴量を挙げているほど確信度が高い
        if len(feature_names) >= 3:
            confidence += 0.2
        return min(confidence, 1.0)
