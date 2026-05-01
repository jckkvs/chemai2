# backend/llm/report_generator.py
"""
Automated reporting with LLM-generated explanations and hypotheses.

Collects analysis results and uses LLM to generate:
  1. Model performance interpretation
  2. Feature importance explanation
  3. Hypothesis generation for future experiments
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from backend.llm.provider import LLMProvider, LLMRequest, LLMResponse
from backend.llm import get_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Collection of all analysis results for reporting."""
    # Data info
    n_samples: int = 0
    n_features: int = 0
    target_col: str = ""
    task_type: str = "regression"  # "regression" or "classification"

    # Model performance
    best_model: str = ""
    best_score: float = 0.0  # R2 or Accuracy
    model_comparison: List[Dict[str, Any]] = field(default_factory=list)

    # Feature importance
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    monotonic_constraints: Dict[str, str] = field(default_factory=dict)

    # EDA findings
    eda_summary: str = ""
    outliers_detected: int = 0
    pseudo_ofat_opportunities: int = 0

    # CV strategy used
    cv_strategy: str = ""
    cv_scores: List[float] = field(default_factory=list)

    # Error analysis
    top_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_patterns: str = ""


@dataclass
class GeneratedReport:
    """LLM-generated report."""
    title: str = ""
    summary: str = ""
    model_interpretation: str = ""
    feature_explanations: str = ""
    hypotheses: List[str] = field(default_factory=list)
    experiment_suggestions: List[str] = field(default_factory=list)
    full_report: str = ""


class LLMReportGenerator:
    """
    LLM-powered automated report generation.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider("stub")
        if self.provider is None:
            from backend.llm.provider import StubLLMProvider
            self.provider = StubLLMProvider()

    def generate_report(
        self,
        results: AnalysisResults,
        include_hypotheses: bool = True,
        include_experiments: bool = True,
    ) -> GeneratedReport:
        """
        Generate a comprehensive report based on analysis results.

        Args:
            results: AnalysisResults containing all analysis outcomes
            include_hypotheses: Whether to generate hypotheses
            include_experiments: Whether to suggest new experiments

        Returns:
            GeneratedReport with all sections
        """
        system_prompt = """あなたは化学・材料科学分野の機械学習解析レポート作成の専門家です。
以下の情報を元に、詳細で科学的な解析レポートを作成してください。

レポート構成：
1. 要約（エグゼクティブサマリー）
2. モデル性能の解釈
3. 特徴量重要度の解釈と物理化学的意味
4. 仮説生成（なぜその結果になったのか？）
5. 次の実験提案（pseudo-OFAT等）

重要：
- 科学的根拠に基づいた解釈をする
- 単調性制約があれば、それに基づく解釈を含める
- 疑似的OFAT（One-Factor-At-a-Time）機会があれば、それを次の実験提案に含める
- 日本語で出力する"""

        user_prompt = self._format_results(results)

        # Add specific instructions
        if include_hypotheses:
            user_prompt += "\n\n仮説を3-5個生成してください。"
        if include_experiments:
            user_prompt += "\n\n次の実験提案を3-5個生成してください（pseudo-OFATを含む）。"

        user_prompt += "\n\n上記の情報に基づいて、構造化されたレポートを作成してください。"

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.3,
        )

        try:
            response = self.provider.generate(request)
            return self._parse_report(response.content, results)
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._fallback_report(results)

    def _format_results(self, results: AnalysisResults) -> str:
        """Format AnalysisResults into prompt text."""
        lines = [
            "=== 解析結果サマリー ===",
            f"データ: {results.n_samples}サンプル、{results.n_features}特徴量",
            f"目的変数: {results.target_col} ({results.task_type})",
            "",
            "=== モデル性能 ===",
            f"最適モデル: {results.best_model}",
            f"性能スコア: {results.best_score:.4f}",
        ]

        if results.model_comparison:
            lines.append("\nモデル比較:")
            for m in results.model_comparison[:5]:
                lines.append(f"  - {m.get('name', '')}: {m.get('score', 0.0):.4f}")

        if results.feature_importance:
            lines.append("\n=== 特徴量重要度 ===")
            for f in sorted(results.feature_importance,
                              key=lambda x: x.get('importance', 0.0), reverse=True)[:10]:
                lines.append(f"  - {f.get('name', '')}: {f.get('importance', 0.0):.4f}")

        if results.monotonic_constraints:
            lines.append("\n=== 単調性制約 ===")
            for feat, direction in results.monotonic_constraints.items():
                lines.append(f"  - {feat}: {direction}")

        if results.eda_summary:
            lines.append(f"\n=== EDA発見 ===\n{results.eda_summary}")

        if results.pseudo_ofat_opportunities > 0:
            lines.append(
                f"\n疑似的OFAT機会: {results.pseudo_ofat_opportunities}件発見されました。"
                f"これらを次の実験提案に活用してください。"
            )

        if results.cv_strategy:
            lines.append(f"\n=== 交差検証 ===")
            lines.append(f"戦略: {results.cv_strategy}")
            if results.cv_scores:
                lines.append(f"スコア: {[f'{s:.4f}' for s in results.cv_scores]}")

        if results.error_patterns:
            lines.append(f"\n=== エラー分析 ===\n{results.error_patterns}")

        return "\n".join(lines)

    def _parse_report(
        self, text: str, results: AnalysisResults
    ) -> GeneratedReport:
        """Parse LLM response into GeneratedReport."""
        # Simple parsing - split by sections
        report = GeneratedReport()

        # Extract title (first line or first ## section)
        lines = text.split("\n")
        for line in lines:
            if line.startswith("# ") or line.startswith("## "):
                report.title = line.lstrip("# ").strip()
                break
        if not report.title and lines:
            report.title = lines[0].strip()

        report.full_report = text

        # Try to extract sections
        if "要約" in text or "概要" in text:
            report.summary = self._extract_section(text, ["要約", "概要", "Summary"])

        if "モデル" in text or "性能" in text:
            report.model_interpretation = self._extract_section(
                text, ["モデル性能", "Model Performance", "性能解釈"]
            )

        if "特徴" in text or "重要度" in text:
            report.feature_explanations = self._extract_section(
                text, ["特徴量", "Feature", "重要度"]
            )

        if "仮説" in text or "Hypothesis" in text:
            hypotheses_text = self._extract_section(
                text, ["仮説", "Hypothesis", "推論"]
            )
            # Extract bullet points
            report.hypotheses = [
                line.strip("- ").strip()
                for line in hypotheses_text.split("\n")
                if line.strip().startswith("-")
            ]

        if "実験" in text or "Experiment" in text:
            exp_text = self._extract_section(
                text, ["実験", "Experiment", "提案"]
            )
            report.experiment_suggestions = [
                line.strip("- ").strip()
                for line in exp_text.split("\n")
                if line.strip().startswith("-")
            ]

        return report

    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """Extract a section from report text based on keywords."""
        lines = text.split("\n")
        start_idx = -1
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                start_idx = i
                break

        if start_idx < 0:
            return ""

        # Find next section (next ## or #)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith("# "):
                end_idx = i
                break

        return "\n".join(lines[start_idx:end_idx]).strip()

    def _fallback_report(self, results: AnalysisResults) -> GeneratedReport:
        """Generate a simple fallback report when LLM fails."""
        report = GeneratedReport()
        report.title = f"解析レポート: {results.target_col}"

        summary_lines = [
            f"データ: {results.n_samples}サンプル、{results.n_features}特徴量",
            f"最適モデル: {results.best_model} (スコア: {results.best_score:.4f})",
        ]
        if results.pseudo_ofat_opportunities > 0:
            summary_lines.append(
                f"疑似的OFAT機会が{results.pseudo_ofat_opportunities}件あります。"
            )
        report.summary = "\n".join(summary_lines)

        report.model_interpretation = (
            f"{results.best_model} が最も良い性能を示しました "
            f"(スコア: {results.best_score:.4f})。"
        )

        if results.feature_importance:
            top_features = sorted(
                results.feature_importance,
                key=lambda x: x.get('importance', 0.0),
                reverse=True
            )[:3]
            report.feature_explanations = "重要な特徴量:\n" + "\n".join(
                f"  - {f.get('name', '')}: {f.get('importance', 0.0):.4f}"
                for f in top_features
            )

        report.full_report = "\n\n".join([
            report.title,
            report.summary,
            report.model_interpretation,
            report.feature_explanations,
        ])

        return report


def generate_quick_report(
    results: AnalysisResults,
    provider: Optional[LLMProvider] = None,
) -> str:
    """
    Convenience function to generate a quick report.

    Args:
        results: AnalysisResults
        provider: Optional LLM provider

    Returns:
        Full report text
    """
    generator = LLMReportGenerator(provider=provider)
    report = generator.generate_report(results)
    return report.full_report


if __name__ == "__main__":
    # Simple test
    results = AnalysisResults(
        n_samples=100,
        n_features=5,
        target_col="yield",
        task_type="regression",
        best_model="RandomForestRegressor",
        best_score=0.85,
        feature_importance=[
            {"name": "temp", "importance": 0.4},
            {"name": "pressure", "importance": 0.3},
            {"name": "solvent_type", "importance": 0.2},
        ],
        monotonic_constraints={"temp": "increasing", "pressure": "increasing"},
        cv_strategy="KFold",
        cv_scores=[0.82, 0.85, 0.84, 0.86, 0.83],
        eda_summary="温度と収率に正の相関が見られます。",
        pseudo_ofat_opportunities=3,
    )

    # Test with stub provider
    generator = LLMReportGenerator()
    report = generator.generate_report(results)

    print(f"Title: {report.title}")
    print(f"\nSummary:\n{report.summary}")
    print(f"\nHypotheses: {len(report.hypotheses)} generated")
    print(f"\nExperiment suggestions: {len(report.experiment_suggestions)} generated")
