"""
加重平均の種類自動判断 - LLMがデータと文脈から適切な平均手法を判定

20260429.txtの要件：
  「加重平均も重量平均なのかmol平均なのか、特殊な平均なのかもLLMが自動で判断する」

対応する平均手法：
  - 重量平均 (Weight-based): 分子量・密度等で重み付け
  - mol平均 (Mol-count-based): 分子数で重み付け
  - 等加重平均 (Simple): すべて同じ重み
  - 体積平均 (Volume-based): 分子体積で重み付け
  - 特殊平均 (Special): データの性質に応じた特殊な重み付け
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from backend.llm.provider import LLMProvider, LLMRequest
from backend.llm import get_llm_provider

logger = logging.getLogger(__name__)


# ── データクラス ────────────────────────────────────────

@dataclass
class AverageTypeResult:
    """加重平均の判定結果"""
    average_type: Literal["weight", "mol", "simple", "volume", "special"]
    display_name: str  # 日本語表示名
    confidence: float  # 0.0-1.0
    reasoning: str  # LLMの判断根拠
    weight_column: Optional[str] = None  # 重みとして使用する列名
    special_formula: Optional[str] = None  # 特殊平均の場合の計算式
    warnings: List[str] = field(default_factory=list)


@dataclass
class AverageContext:
    """平均判定のためのコンテキスト"""
    target_property: str  # 対象となる物性値（屈折率、粘度など）
    available_columns: List[str]  # 利用可能な列名
    sample_data: Dict[str, List[Any]] = field(default_factory=dict)  # サンプルデータ
    mixture_type: Optional[Literal["binary", "ternary", "multi"]] = None
    has_smiles: bool = False
    smiles_col: Optional[str] = None
    has_molecular_weight: bool = False
    has_density: bool = False
    has_volume: bool = False
    has_mol_fraction: bool = False
    user_hint: Optional[str] = None  # ユーザーのヒント


# ── 物理化学的知見データベース ─────────────────────────

# 物性値 → 推奨される平均手法のマッピング
_PROPERTY_AVERAGE_MAP: Dict[str, Dict[str, Any]] = {
    "屈折率": {
        "default": "volume",
        "reason": "屈折率はロレンツ-ローレンツの式に基づき体積平均が物理的に正しい",
        "alternatives": ["weight", "simple"],
    },
    "粘度": {
        "default": "special",
        "reason": "混合物の粘度は対数混合則（log-additivity）を用いるのが一般的",
        "special_formula": "log_viscosity = Σ(xi * log(μi))",
        "alternatives": ["weight"],
    },
    "密度": {
        "default": "weight",
        "reason": "密度は質量基準での配合比率が自然",
        "alternatives": ["volume", "simple"],
    },
    "誘電率": {
        "default": "volume",
        "reason": "誘電率は体積分率での平均が物理的根拠に基づく",
        "alternatives": ["weight"],
    },
    "溶解度": {
        "default": "mol",
        "reason": "溶解度はモル分率での記述が化学的に正しい",
        "alternatives": ["weight"],
    },
    "沸点": {
        "default": "special",
        "reason": "混合物の沸点はラウールの法則等の特殊則を使用",
        "special_formula": "P_total = Σ(xi * Pi_sat) (Raoult's law)",
        "alternatives": ["mol"],
    },
    "融点": {
        "default": "mol",
        "reason": "融点降下はモル分率で記述される",
        "alternatives": ["weight"],
    },
    "熱伝導率": {
        "default": "volume",
        "reason": "熱伝導率は体積分率平均が一般的",
        "alternatives": ["weight"],
    },
    "引張強度": {
        "default": "weight",
        "reason": "機械的特性は質量あたりで評価されることが多い",
        "alternatives": ["simple"],
    },
    "ガラス転移温度": {
        "default": "special",
        "reason": "TgはFox方程式やGordon-Taylor式などの特殊則を使用",
        "special_formula": "1/Tg = Σ(wi/Tgi) (Fox equation)",
        "alternatives": ["weight"],
    },
}


# ── メインクラス ──────────────────────────────────────────

class AverageTypeDetector:
    """
    加重平均の種類をLLM＋物理化学的知見から自動判定
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider("stub")
        if self.provider is None:
            from backend.llm.provider import StubLLMProvider
            self.provider = StubLLMProvider()

    def detect(
        self,
        context: AverageContext,
        use_llm: bool = True,
    ) -> AverageTypeResult:
        """
        コンテキストから適切な平均手法を判定

        Args:
            context: 判定に必要なコンテキスト
            use_llm: LLMを使用するか（Falseでルールベースのみ）

        Returns:
            AverageTypeResult
        """
        # ステップ1: 物理化学的知見からの推奨
        rule_based_result = self._rule_based_detect(context)

        if not use_llm:
            return rule_based_result

        # ステップ2: LLMによる判定
        try:
            llm_result = self._llm_detect(context, rule_based_result)
            return llm_result
        except Exception as e:
            logger.warning(f"LLM average type detection failed: {e}, using rule-based result")
            return rule_based_result

    def _rule_based_detect(self, context: AverageContext) -> AverageTypeResult:
        """物理化学的知見に基づく判定（フォールバック）"""
        target = context.target_property
        # マッチする物性を探す
        matched_key = None
        for key in _PROPERTY_AVERAGE_MAP:
            if key in target or key in target.lower():
                matched_key = key
                break

        if matched_key:
            info = _PROPERTY_AVERAGE_MAP[matched_key]
            avg_type = info["default"]
            return AverageTypeResult(
                average_type=avg_type,
                display_name=self._type_to_display(avg_type),
                confidence=0.8,
                reasoning=f"物理化学的知見に基づく推奨：{info['reason']}",
                special_formula=info.get("special_formula"),
                warnings=[],
            )

        # マッチしない場合：データの性質から推奨
        if context.has_molecular_weight or any("mw" in c.lower() or "molwt" in c.lower() for c in context.available_columns):
            return AverageTypeResult(
                average_type="weight",
                display_name="重量平均",
                confidence=0.6,
                reasoning="分子量情報があるため、重量平均を推奨します。",
                warnings=["物性値に応じて適切な平均手法が異なる場合があります。"],
            )

        if context.has_mol_fraction or any("mol" in c.lower() or "fraction" in c.lower() for c in context.available_columns):
            return AverageTypeResult(
                average_type="mol",
                display_name="mol平均",
                confidence=0.6,
                reasoning="モル分率情報があるため、mol平均を推奨します。",
                warnings=[],
            )

        # デフォルト
        return AverageTypeResult(
            average_type="simple",
            display_name="等加重平均",
            confidence=0.4,
            reasoning="十分な情報がないため、等加重平均をデフォルトとしています。",
            warnings=["混合系の場合は、適切な加重平均手法を選択してください。"],
        )

    def _llm_detect(
        self, context: AverageContext, rule_based: AverageTypeResult,
    ) -> AverageTypeResult:
        """LLMによる判定"""
        prompt = self._build_prompt(context, rule_based)
        system_prompt = """あなたは物理化学・材料科学の専門家です。
混合物の物性値に対する平均手法の選択について、科学的に正しい判断をしてください。

特に以下の観点から判断してください：
1. 物理化学的性質：その物性値が.additivityに対してどう振る舞うか
2. 混合メカニズム：質量・モル数・体積のどれが支配的か
3. 実験条件：配合比率は質量ベースか体積ベースか

日本語で回答してください。"""

        request = LLMRequest(
            user_prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.2,
        )

        response = self.provider.generate(request)
        return self._parse_llm_response(response.content, context, rule_based)

    def _build_prompt(self, context: AverageContext, rule_based: AverageTypeResult) -> str:
        """LLMへのプロンプトを構築"""
        lines = [
            "# 加重平均手法の選択相談",
            "",
            "## 対象物性",
            f"物性値: {context.target_property}",
            "",
            "## データ情報",
            f"- 利用可能な列: {', '.join(context.available_columns[:15])}",
            f"- 混合タイプ: {context.mixture_type or '不明'}",
            f"- SMILESあり: {'はい' if context.has_smiles else 'いいえ'}",
            f"- 分子量情報あり: {'はい' if context.has_molecular_weight else 'いいえ'}",
            f"- 密度情報あり: {'はい' if context.has_density else 'いいえ'}",
            f"- 体積情報あり: {'はい' if context.has_volume else 'いいえ'}",
            f"- モル分率あり: {'はい' if context.has_mol_fraction else 'いいえ'}",
        ]

        if context.sample_data:
            lines.append("")
            lines.append("## サンプルデータ")
            for col, values in list(context.sample_data.items())[:5]:
                lines.append(f"  {col}: {values[:3]}")

        lines.append("")
        lines.append("## ルールベース推奨（参考）")
        lines.append(f"推奨手法: {rule_based.display_name} ({rule_based.average_type})")
        lines.append(f"理由: {rule_based.reasoning}")

        if context.user_hint:
            lines.append("")
            lines.append(f"## ユーザーヒント: {context.user_hint}")

        lines.append("")
        lines.append("## 選択肢")
        lines.append("1. **重量平均 (weight)**: 分子量・質量で重み付け")
        lines.append("2. **mol平均 (mol)**: 分子数・モル分率で重み付け")
        lines.append("3. **等加重平均 (simple)**: すべて同じ重み")
        lines.append("4. **体積平均 (volume)**: 分子体積・体積分率で重み付け")
        lines.append("5. **特殊平均 (special)**: その物性に特化した特殊な平均式")

        lines.append("")
        lines.append("## 指示")
        lines.append("上記の情報を元に、最も適切な平均手法を1つ選択し、")
        lines.append("以下のJSON形式で回答してください：")
        lines.append("")
        lines.append("""```json
{
    "average_type": "weight|mol|simple|volume|special",
    "confidence": 0.0-1.0,
    "reasoning": "判断の根拠を日本語で",
    "weight_column": "重みとして使用する列名（該当する場合）",
    "special_formula": "特殊平均の場合の計算式（該当する場合）",
    "warnings": ["注意すべき点があればリストで"]
}
```""")

        return "\n".join(lines)

    def _parse_llm_response(
        self,
        content: str,
        context: AverageContext,
        rule_based: AverageTypeResult,
    ) -> AverageTypeResult:
        """LLMの回答をパース"""
        import json
        import re

        # JSONブロックを抽出
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSONっぽい部分を探す
            json_match = re.search(r'\{[^{}]*"average_type"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # パース失敗 → ルールベースにフォールバック
                logger.warning(f"Failed to parse LLM response: {content[:200]}")
                return rule_based

        try:
            data = json.loads(json_str)
            avg_type = data.get("average_type", rule_based.average_type)
            if avg_type not in ["weight", "mol", "simple", "volume", "special"]:
                avg_type = rule_based.average_type

            return AverageTypeResult(
                average_type=avg_type,
                display_name=self._type_to_display(avg_type),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", rule_based.reasoning),
                weight_column=data.get("weight_column"),
                special_formula=data.get("special_formula"),
                warnings=data.get("warnings", []),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse error: {e}, content: {content[:200]}")
            return rule_based

    @staticmethod
    def _type_to_display(avg_type: str) -> str:
        """平均手法の種類を表示名に変換"""
        mapping = {
            "weight": "重量平均",
            "mol": "mol平均",
            "simple": "等加重平均",
            "volume": "体積平均",
            "special": "特殊平均",
        }
        return mapping.get(avg_type, avg_type)

    def get_available_types(self) -> List[Dict[str, str]]:
        """利用可能な平均手法のリストを返す（UI用）"""
        return [
            {"value": "weight", "display": "重量平均", "desc": "分子量・質量で重み付け"},
            {"value": "mol", "display": "mol平均", "desc": "分子数・モル分率で重み付け"},
            {"value": "simple", "display": "等加重平均", "desc": "すべて同じ重み"},
            {"value": "volume", "display": "体積平均", "desc": "分子体積・体積分率で重み付け"},
            {"value": "special", "display": "特殊平均", "desc": "物性に特化した特殊な平均式"},
        ]


# ── 便利関数 ──────────────────────────────────────────────

def calculate_weighted_average(
    values: List[float],
    weights: List[float],
    average_type: str,
    special_formula: Optional[str] = None,
) -> float:
    """
    指定された平均手法で加重平均を計算

    Args:
        values: 平均する値のリスト
        weights: 重みのリスト
        average_type: 平均手法
        special_formula: 特殊平均の計算式

    Returns:
        平均値
    """
    import numpy as np

    values = np.array(values)
    weights = np.array(weights)

    if average_type == "simple":
        return float(np.mean(values))

    if average_type == "special" and special_formula:
        # 特殊平均の処理（ログ混合則など）
        if "log" in special_formula.lower():
            # 対数混合則: log(y) = Σ wi * log(xi)
            log_values = np.log(np.maximum(values, 1e-10))
            log_result = np.sum(weights * log_values) / np.sum(weights)
            return float(np.exp(log_result))
        # その他の特殊則は拡張可能

    # 標準的な加重平均
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total_weight)


def detect_average_type_for_target(
    target_property: str,
    available_columns: List[str],
    sample_df: Optional[Any] = None,
    provider: Optional[LLMProvider] = None,
) -> AverageTypeResult:
    """
    便利関数：目標変数名から平均手法を自動検出

    Args:
        target_property: 目標物性値名
        available_columns: 利用可能な列名
        sample_df: サンプルデータ（オプション）
        provider: LLMプロバイダー

    Returns:
        AverageTypeResult
    """
    # サンプルデータの準備
    sample_data = {}
    if sample_df is not None:
        for col in available_columns[:5]:
            if col in sample_df.columns:
                sample_data[col] = sample_df[col].dropna().head(3).tolist()

    # コンテキスト構築
    ctx = AverageContext(
        target_property=target_property,
        available_columns=available_columns,
        sample_data=sample_data,
        has_smiles=any("smiles" in c.lower() for c in available_columns),
        has_molecular_weight=any(
            kw in " ".join(available_columns).lower()
            for kw in ["mw", "molwt", "molecular_weight", "分子量"]
        ),
        has_density=any("density" in c.lower() for c in available_columns),
        has_volume=any("volume" in c.lower() for c in available_columns),
        has_mol_fraction=any(
            kw in " ".join(available_columns).lower()
            for kw in ["mol_frac", "mole_fraction", "モル分率"]
        ),
    )

    detector = AverageTypeDetector(provider=provider)
    return detector.detect(ctx)
