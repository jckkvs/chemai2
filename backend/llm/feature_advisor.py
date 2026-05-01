# backend/llm/feature_advisor.py
"""
LLM-driven feature selection and monotonic constraint recommendation.

Uses LLM to:
  1. Analyze data features and recommend important ones.
  2. Suggest monotonic constraints (direction and strength).
  3. Adjust constraint strength based on sample size vs feature count.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from backend.llm.provider import LLMProvider, LLMRequest, LLMResponse
from backend.llm import get_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class FeatureRecommendation:
    """Recommendation for a single feature."""
    name: str
    importance: float  # 0.0 - 1.0
    monotonic_direction: str  # "increasing", "decreasing", "none"
    monotonic_strength: str  # "strong", "medium", "weak"
    reason: str = ""


@dataclass
class FeatureSelectionResult:
    """Result of LLM-driven feature selection."""
    selected_features: List[str]
    feature_details: List[FeatureRecommendation]
    monotonic_constraints: Dict[str, Dict[str, any]]  # feature -> {direction, strength}
    constraint_strength_global: str  # "strong", "medium", "weak"
    notes: str = ""


class LLMFeatureAdvisor:
    """
    LLM-powered feature selection and constraint advisor.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider("stub")
        if self.provider is None:
            from backend.llm.provider import StubLLMProvider
            self.provider = StubLLMProvider()

    def recommend_features(
        self,
        feature_names: List[str],
        feature_types: Dict[str, str],
        n_samples: int,
        target_col: str,
        task_type: str = "regression",  # "regression" or "classification"
        user_goal: str = "prediction",  # "prediction", "interpretation", "experimental_planning"
        prediction_target: str = "similar_samples",  # "similar_samples", "far_samples"
    ) -> FeatureSelectionResult:
        """
        Use LLM to recommend features and monotonic constraints.
        """
        system_prompt = """あなたは機械学習の特徴選択と単調性制約の専門家です。
化学・材料科学データに対して、以下を行ってください：

1. 各特徴量について、予測への重要性を評価する（0.0-1.0）
2. 物理化学的知見に基づき、単調性制約を推奨する
   - 温度→反応速度：通常は単調増加（高温ほど反応が進む）
   - 圧力→収率：多くの場合単調増加
   - 分子量→物性：単調関係が多い
3. 制約の強さを推奨する（strong/medium/weak）
   - 強い科学的根拠がある場合：strong
   - 経験的知見：medium
   - 不確実：weak

出力形式（JSON）：
{
  "selected_features": ["temp", "pressure"],
  "features": [
    {"name": "temp", "importance": 0.9, "monotonic_direction": "increasing",
     "monotonic_strength": "strong", "reason": "温度上昇で反応速度増加"},
    ...
  ],
  "constraint_strength_global": "medium",
  "notes": "サンプル数が少ないため、medium強度を推奨"
}"""

        user_prompt = f"""データ情報：
- サンプル数: {n_samples}
- 特徴量: {feature_names}
- 特徴量の型: {feature_types}
- 目的変数: {target_col} ({task_type})
- 解析目的: {user_goal}
- 予測対象: {prediction_target}

上記の情報に基づいて、特徴選択と単調性制約を推奨してください。
サンプル数に対して特徴量が多すぎる場合は、重要な特徴に絞ってください。
{n_samples}サンプルに対して、適切な特徴数は通常 max(5, n_samples/10) 程度です。"""

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.2,
        )

        try:
            response = self.provider.generate(request)
            return self._parse_feature_recommendation(response.content, feature_names)
        except Exception as e:
            logger.error(f"LLM feature recommendation failed: {e}")
            return self._fallback_recommendation(feature_names, feature_types)

    def _parse_feature_recommendation(
        self, text: str, feature_names: List[str]
    ) -> FeatureSelectionResult:
        """Parse LLM response into structured result."""
        import json

        # Try to extract JSON from response
        try:
            # Find JSON block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")

            features = []
            for f in data.get("features", []):
                features.append(FeatureRecommendation(
                    name=f["name"],
                    importance=f.get("importance", 0.5),
                    monotonic_direction=f.get("monotonic_direction", "none"),
                    monotonic_strength=f.get("monotonic_strength", "medium"),
                    reason=f.get("reason", "")
                ))

            # Build constraints dict
            constraints = {}
            for f in features:
                if f.monotonic_direction != "none":
                    constraints[f.name] = {
                        "direction": f.monotonic_direction,
                        "strength": self._strength_to_value(f.monotonic_strength),
                    }

            return FeatureSelectionResult(
                selected_features=data.get("selected_features", feature_names[:10]),
                feature_details=features,
                monotonic_constraints=constraints,
                constraint_strength_global=data.get("constraint_strength_global", "medium"),
                notes=data.get("notes", "")
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_recommendation(feature_names, {})

    def _strength_to_value(self, strength: str) -> float:
        """Convert strength string to numeric value."""
        mapping = {"strong": 1.0, "medium": 0.5, "weak": 0.1}
        return mapping.get(strength.lower(), 0.5)

    def _fallback_recommendation(
        self, feature_names: List[str], feature_types: Dict[str, str]
    ) -> FeatureSelectionResult:
        """Fallback when LLM fails."""
        selected = feature_names[:min(10, len(feature_names))]
        features = [
            FeatureRecommendation(
                name=f,
                importance=0.5,
                monotonic_direction="none",
                monotonic_strength="medium",
            )
            for f in selected
        ]
        return FeatureSelectionResult(
            selected_features=selected,
            feature_details=features,
            monotonic_constraints={},
            constraint_strength_global="medium",
            notes="LLM failed, using fallback recommendation"
        )

    def recommend_constraint_strength(
        self,
        n_samples: int,
        n_features: int,
        feature_name: str,
        physical_meaning: str = "",
    ) -> str:
        """
        Recommend constraint strength based on sample size and feature count.

        Rules:
        - n_samples < 50: "weak" (avoid overfitting)
        - 50 <= n_samples < 200: "medium"
        - n_samples >= 200: "strong" (enough data)
        - If n_features > n_samples / 5: reduce strength by one level
        """
        if n_samples < 50:
            strength = "weak"
        elif n_samples < 200:
            strength = "medium"
        else:
            strength = "strong"

        # Adjust based on feature count
        if n_features > n_samples / 5:
            if strength == "strong":
                strength = "medium"
            elif strength == "medium":
                strength = "weak"

        return strength


def apply_feature_selection_to_pipeline(
    result: FeatureSelectionResult,
    pipeline_config: dict,
) -> dict:
    """
    Apply LLM feature selection recommendations to pipeline config.

    Updates:
    - monotonic_constraints in model config
    - feature selection (if supported)
    """
    config = pipeline_config.copy()

    # Add monotonic constraints if any
    if result.monotonic_constraints:
        if "model_params" not in config:
            config["model_params"] = {}
        config["model_params"]["monotonic_constraints"] = result.monotonic_constraints
        config["model_params"]["constraint_strength"] = result.constraint_strength_global

    return config


if __name__ == "__main__":
    # Simple test
    advisor = LLMFeatureAdvisor()
    result = advisor.recommend_features(
        feature_names=["temp", "pressure", "solvent_type", "catalyst_conc"],
        feature_types={"temp": "numeric", "pressure": "numeric",
                     "solvent_type": "categorical", "catalyst_conc": "numeric"},
        n_samples=100,
        target_col="yield",
        task_type="regression",
        user_goal="prediction",
        prediction_target="similar_samples",
    )
    print(f"Selected features: {result.selected_features}")
    print(f"Global strength: {result.constraint_strength_global}")
    print(f"Constraints: {result.monotonic_constraints}")
