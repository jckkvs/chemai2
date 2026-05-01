"""
backend/llm/analysis_advisor.py

LLMを活用したデータ分析アドバイザー。
データ要約から解析手法、特徴量、CV手法、DOE手法まで自動推奨。

要件:
  - データ要約（LLMが読み、解析目的を判定）
  - 解析手法推奨（予測/逆解析/実験計画）
  - 特徴量推奨（物理化学的知見に基づく）
  - CV手法推奨（データ特性に基づく）
  - DOE手法推奨（因子数・水準数に基づく）
  - 単調性制約推奨（物理化学的仮定に基づく）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── データクラス ──────────────────────────────────────────────

@dataclass
class DataSummary:
    """データ要約情報。"""
    n_rows: int
    n_cols: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    has_smiles: bool
    smiles_col: Optional[str] = None
    target_col: Optional[str] = None
    target_type: Optional[str] = None  # 'regression' / 'classification'
    n_samples: int = 0
    missing_ratio: float = 0.0
    suggested_task: str = ""  # 'prediction' / 'inverse' / 'doe'
    suggested_approach: str = ""


@dataclass
class DescriptorRecommendation:
    """特徴量推奨結果。"""
    target_type: str
    recommended_sets: List[str]
    individual_descriptors: List[str]
    reasoning: str


@dataclass
class CVRecommendation:
    """CV手法推奨結果。"""
    method: str
    reason: str
    params: Dict[str, Any]


@dataclass
class DoeRecommendation:
    """DOE手法推奨結果。"""
    method: str
    reason: str
    default_criterion: str


@dataclass
class MonotonicityRecommendation:
    """単調性制約推奨結果。"""
    is_monotonic_assumed: bool
    strong_constraints: List[str]
    weak_constraints: List[str]
    reasoning: str


# ── 物理化学的知見データベース ─────────────────────────────────

_PHYSICAL_CHEM_TARGETS = {
    "屈折率": {
        "type": "regression",
        "descriptors": ["MolWt", "LogP", "TPSA", "Refractivity", "Polarizability", "RI"],
        "monotonic": {"MolWt": "weak", "LogP": "weak", "Polarizability": "strong"},
    },
    "誘電率": {
        "type": "regression",
        "descriptors": ["LogP", "DipoleMoment", "Polarizability", "HOMO", "LUMO", "HeteroatomCount"],
        "monotonic": {"DipoleMoment": "strong", "Polarizability": "strong"},
    },
    "溶解度": {
        "type": "regression",
        "descriptors": ["LogP", "LogS", "TPSA", "HBD", "HBA", "MolWt", "RingCount"],
        "monotonic": {"LogP": "strong", "LogS": "strong"},
    },
    "バンドギャップ": {
        "type": "regression",
        "descriptors": ["HOMO", "LUMO", "BandGap", "Polarizability", "DipoleMoment"],
        "monotonic": {"HOMO": "weak", "LUMO": "weak", "BandGap": "strong"},
    },
    "引張強度": {
        "type": "regression",
        "descriptors": ["MolWt", "TPSA", "LogP", "RotatableBondCount", "RingCount", "FractionCSP3"],
        "monotonic": {"MolWt": "weak", "RingCount": "weak"},
    },
    "熱膨張係数": {
        "type": "regression",
        "descriptors": ["MolWt", "LogP", "TPSA", "NumAromaticRings"],
        "monotonic": {"NumAromaticRings": "weak"},
    },
    "毒性": {
        "type": "regression",
        "descriptors": ["LogP", "TPSA", "HBD", "HBA", "Mutagenicity", "Carcinogenicity"],
        "monotonic": {"LogP": "weak", "Mutagenicity": "strong"},
    },
    "水素透過性": {
        "type": "regression",
        "descriptors": ["LogP", "TPSA", "MolWt", "RotatableBondCount"],
        "monotonic": {"LogP": "strong", "TPSA": "strong"},
    },
}


# ── メインクラス ────────────────────────────────────────────────

class AnalysisAdvisor:
    """LLM活用データ分析アドバイザー。"""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self._summary: Optional[DataSummary] = None

    def summarize_data(self, df: pd.DataFrame, target_col: str = "") -> DataSummary:
        """データを要約し、解析目的を推奨。"""
        n_rows, n_cols = df.shape
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(exclude=['number']).columns)

        # SMILES列の検出
        smiles_col = None
        for col in df.columns:
            if col.upper() in {"SMILES", "SMILE", "SMLS"}:
                smiles_col = col
                break
            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if any(str(v).startswith("C") and "C" in str(v) for v in sample):
                    smiles_col = col
                    break

        # 欠損率
        missing = df.isnull().sum().sum()
        total = df.size
        missing_ratio = missing / total if total > 0 else 0.0

        summary = DataSummary(
            n_rows=n_rows,
            n_cols=n_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            has_smiles=smiles_col is not None,
            smiles_col=smiles_col,
            target_col=target_col or (numeric_cols[0] if numeric_cols else None),
            n_samples=n_rows,
            missing_ratio=missing_ratio,
        )

        # 目標変数タイプ判定
        if summary.target_col:
            col_data = df[summary.target_col]
            n_unique = col_data.nunique()
            if col_data.dtype in ['object', 'category'] or n_unique <= 10:
                summary.target_type = 'classification'
            else:
                summary.target_type = 'regression'

        # 解析手法推奨
        if summary.has_smiles and smiles_col:
            summary.suggested_task = 'prediction'
            if n_rows < 50:
                summary.suggested_approach = 'SMILES特徴量化 + 少量データ用モデル（LinearTree/RGF等）'
            else:
                summary.suggested_approach = 'SMILES特徴量化 + 全記述子計算 + AutoML'
        elif n_cols > summary.n_rows:
            summary.suggested_task = 'prediction'
            summary.suggested_approach = '説明変数多数 → Random Projection + 区分線形モデル'
        elif n_rows < 100:
            summary.suggested_task = 'prediction'
            summary.suggested_approach = '少量データ → CV慎重 + 単調性制約モデル推奨'
        else:
            summary.suggested_task = 'prediction'
            summary.suggested_approach = '標準的な予測解析 → AutoML + SHAP解釈'

        self._summary = summary
        return summary

    def recommend_descriptors(self, target_name: str) -> DescriptorRecommendation:
        """目標変数名から特徴量を推奨（物理化学的知見）"""
        # 目標変数名から推奨を検索
        target_lower = target_name.lower()
        matched_key = None
        for key in _PHYSICAL_CHEM_TARGETS:
            if key in target_name or key in target_lower:
                matched_key = key
                break

        if matched_key:
            info = _PHYSICAL_CHEM_TARGETS[matched_key]
            return DescriptorRecommendation(
                target_type=info["type"],
                recommended_sets=["rdkit_physical", "rdkit_electronic"],
                individual_descriptors=info["descriptors"],
                reasoning=f'{matched_key}の物理化学的性質に基づき、{", ".join(info["descriptors"])}等の記述子が有効です。',
            )

        # マッチしない場合のデフォルト
        return DescriptorRecommendation(
            target_type='regression',
            recommended_sets=["rdkit_physical", "rdkit_electronic", "mordred_selected"],
            individual_descriptors=["MolWt", "LogP", "TPSA", "HOMO", "LUMO", "DipoleMoment"],
            reasoning='目標変数が特定できませんでした。汎用的な物理化学記述子を推奨します。',
        )

    def recommend_cv(self, summary: DataSummary) -> CVRecommendation:
        """データ特性からCV手法を推奨。"""
        n = summary.n_samples
        has_smiles = summary.has_smiles

        if has_smiles:
            return CVRecommendation(
                method="GroupKFold",
                reason="SMILESデータでは化学構造の類似性に基づくGroupKFoldが適しています。",
                params={"n_splits": min(5, n // 3), "shuffle": True, "random_state": 42},
            )
        elif n < 50:
            return CVRecommendation(
                method="LeaveOneOut",
                reason="極めて少量のデータです。LeaveOneOut CVを推奨します。",
                params={"n_splits": n},
            )
        elif n < 200:
            return CVRecommendation(
                method="KFold",
                reason="少量データです。標準的なKFoldを推奨します。",
                params={"n_splits": 5, "shuffle": True, "random_state": 42},
            )
        else:
            return CVRecommendation(
                method="StratifiedKFold" if summary.target_type == "classification" else "KFold",
                reason="十分なデータ数です。StratifiedKFold/KFoldを推奨します。",
                params={"n_splits": 5, "shuffle": True, "random_state": 42},
            )

    def recommend_doe(self, n_factors: int, n_levels: int) -> DoeRecommendation:
        """因子数・水準数からDOE手法を推奨。"""
        total = n_levels ** n_factors

        if total <= 1000:
            return DoeRecommendation(
                method="FullFactorial",
                reason=f"全組み合わせ数({total})が少ないため、完全要因実験が可能です。",
                default_criterion="D",
            )
        elif n_factors <= 3:
            return DoeRecommendation(
                method="Maximin",
                reason="因子数が少ないため、Maximin（デフォルト）を推奨します。",
                default_criterion="MAXIMIN",
            )
        elif n_factors <= 6:
            return DoeRecommendation(
                method="LatinHypercube",
                reason="中程度の因子数です。LHSまたはSobolを推奨します。",
                default_criterion="SOBOL",
            )
        else:
            return DoeRecommendation(
                method="Sobol",
                reason="因子数が多いため、高次元に強いSobolを推奨します。",
                default_criterion="SOBOL",
            )

    def recommend_monotonicity(self, target_name: str) -> MonotonicityRecommendation:
        """物理化学的仮定に基づき単調性制約を推奨。"""
        target_lower = target_name.lower()
        strong = []
        weak = []

        for key, info in _PHYSICAL_CHEM_TARGETS.items():
            if key in target_name or key in target_lower:
                constraints = info.get("monotonic", {})
                for desc, strength in constraints.items():
                    if strength == "strong":
                        strong.append(desc)
                    else:
                        weak.append(desc)
                break

        if not strong and not weak:
            # デフォルト: 物理化学では多くの関係は単調
            weak = ["LogP", "MolWt", "TPSA"]

        return MonotonicityRecommendation(
            is_monotonic_assumed=True,
            strong_constraints=strong,
            weak_constraints=weak,
            reasoning="物理化学においては、多くの物性は構造パラメータに対して単調な関係にあると仮定できます。",
        )

    def build_llm_prompt_for_data(self, summary: DataSummary) -> str:
        """データ要約からLLMへのプロンプトを生成。"""
        prompt = f"""# データ分析アドバイス要求

## データ概要
- サンプル数: {summary.n_samples}
- 特徴量数: {summary.n_cols}
- 数値列: {len(summary.numeric_cols)}個 ({', '.join(summary.numeric_cols[:5])}...)
- カテゴリ列: {len(summary.categorical_cols)}個
- SMILESデータ: {'あり' if summary.has_smiles else 'なし'} ({summary.smiles_col or ''})
- 目標変数: {summary.target_col or '未設定'}
- 目標タイプ: {summary.target_type or '不明'}
- 欠損率: {summary.missing_ratio:.1%}

## 依頼内容
上記のデータに対して、以下の観点からアドバイスしてください：

1. **推奨される解析手法**: 予測（回帰/分類）、逆解析、実験計画のどれが適していますか？
2. **推奨される特徴量**: 物理化学的知見に基づき、どのような記述子・特徴量が有効ですか？
3. **CV手法**: データ数・特性に基づき、どのような交差検証手法が適していますか？
4. **実験計画法**: もし実験計画をする場合、どのような手法が適していますか？
5. **単調性制約**: 物理化学的仮定から、どのような単調性制約が妥当ですか？

日本語で、簡潔明瞭に回答してください。
"""
        return prompt

    async def get_llm_advice(self, summary: DataSummary) -> str:
        """LLMからアドバイスを取得。"""
        if self.llm is None:
            return "LLMが設定されていません。設定タブからLLMを構成してください。"

        prompt = self.build_llm_prompt_for_data(summary)
        try:
            if hasattr(self.llm, 'generate_async'):
                response = await self.llm.generate_async(prompt)
            elif hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
            else:
                return "LLMプロバイダーが対応していません。"
            return str(response)
        except Exception as e:
            logger.error(f"LLM advice error: {e}", exc_info=True)
            return f"LLMアドバイス取得エラー: {str(e)}"
