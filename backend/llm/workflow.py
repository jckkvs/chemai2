# backend/llm/workflow.py
"""
LLM-guided analysis workflow.

Implements:
  1. Data assessment: LLM reads data summary and understands analysis context.
  2. User interview: LLM asks questions to clarify analysis goal.
  3. CV strategy selection: LLM recommends appropriate cross-validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from backend.llm.provider import LLMProvider, LLMRequest, LLMResponse
from backend.llm import get_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class DataSummary:
    """Summary of the input data for LLM assessment."""
    n_samples: int
    n_features: int
    feature_names: List[str]
    feature_types: Dict[str, str]  # column name -> type (numeric, categorical, smiles, etc.)
    target_col: str
    target_type: str  # "numeric", "binary", "multiclass"
    has_groups: bool = False
    group_col: str = ""
    missing_ratio: float = 0.0
    sample_preview: str = ""  # first few rows as string


@dataclass
class InterviewResult:
    """Result of the LLM-guided interview."""
    goal: str  # "prediction", "interpretation", "experimental_planning"
    prediction_target: str = ""  # "similar_samples", "far_samples", "new_conditions"
    optimization_goal: str = ""  # "maximize", "minimize", "classify"
    monotonic_features: List[Tuple[str, str]] = field(default_factory=list)  # (feature, "increasing"/"decreasing")
    monotonic_strength: str = "medium"  # "strong", "medium", "weak"
    cv_strategy: str = ""  # recommended CV strategy
    notes: str = ""


class LLMWorkflow:
    """
    LLM-guided analysis workflow manager.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider("huggingface")
        if self.provider is None:
            self.provider = get_llm_provider("stub")
        self.interview_history: List[Dict] = []

    def assess_data(
        self,
        data_summary: DataSummary,
        additional_context: str = "",
    ) -> str:
        """
        Step 1: Data assessment by LLM.

        LLM reads the data summary and provides initial assessment.
        """
        system_prompt = """あなたは化学・材料科学分野の機械学習の専門家です。
データの概要を読み取り、以下を行ってください：
1. データの特徴を要約する
2. 考えられる解析目的を提案する（予測、解釈、実験計画）
3. 次にユーザーに聞くべき質問をリストアップする"""

        user_prompt = self._format_data_summary(data_summary)
        if additional_context:
            user_prompt += f"\n\n追加情報: {additional_context}"

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.2,
        )

        try:
            response = self.provider.generate(request)
            return response.content
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            return f"Assessment failed: {e}"

    def conduct_interview(
        self,
        data_summary: DataSummary,
        assessment: str,
        user_answers: Optional[Dict[str, str]] = None,
    ) -> InterviewResult:
        """
        Step 2: Conduct interview with user (via LLM-generated questions).

        If user_answers is provided, use them to generate recommendations.
        Otherwise, generate questions for the user.
        """
        system_prompt = """あなたは機械学習解析のコンサルタントです。
ユーザーの解析目的を詳細に聞き出し、適切な手法を提案してください。
特に以下の点について明らかにしてください：
- 解析目的（予測、解釈、実験計画）
- 予測したいサンプルは訓練データに近いか遠いか
- 最適化（逆解析）が目的かどうか
- 単調性制約が必要な特徴量とその方向
- 単調性制約の強さ（strong/medium/weak）"""

        user_prompt = f"""データ概要:
{self._format_data_summary(data_summary)}

LLM評価:
{assessment}

"""
        if user_answers:
            user_prompt += f"\nユーザーの回答:\n{user_answers}"
            user_prompt += "\n\n上記の回答に基づいて、解析計画を提案してください。"
        else:
            user_prompt += "\n\nユーザーに聞くべき質問を生成してください。"

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.2,
        )

        try:
            response = self.provider.generate(request)
            return self._parse_interview_result(response.content)
        except Exception as e:
            logger.error(f"Interview failed: {e}")
            return InterviewResult(goal="prediction")

    def recommend_cv_strategy(
        self,
        interview_result: InterviewResult,
        data_summary: DataSummary,
    ) -> str:
        """
        Step 3: Recommend CV strategy based on interview result.
        """
        system_prompt = """あなたは機械学習の交差検証（CV）専門家です。
解析目的と予測対象に基づいて、最適なCV手法を推奨してください。

選択肢：
- KFold: 予測対象が訓練データに近い場合
- GroupKFold: グループ（同一化合物等）がある場合
- LeaveOneGroupOut: 予測対象が訓練データから遠い場合
- TimeSeriesSplit: 時系列データの場合
- StratifiedKFold: 分類でクラスバランスが重要な場合"""

        user_prompt = f"""解析目的: {interview_result.goal}
予測対象: {interview_result.prediction_target}
最適化: {interview_result.optimization_goal}
データ情報: {data_summary.n_samples}サンプル、{data_summary.n_features}特徴量
グループ有無: {data_summary.has_groups}

推奨すべきCV手法を選択し、理由を説明してください。"""

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.1,
        )

        try:
            response = self.provider.generate(request)
            return response.content
        except Exception as e:
            logger.error(f"CV recommendation failed: {e}")
            return "KFold (default)"

    def _format_data_summary(self, summary: DataSummary) -> str:
        """Format data summary for LLM prompt."""
        lines = [
            f"サンプル数: {summary.n_samples}",
            f"特徴量数: {summary.n_features}",
            f"目的変数: {summary.target_col} ({summary.target_type})",
            f"欠損率: {summary.missing_ratio:.1%}",
        ]

        if summary.feature_types:
            lines.append("\n特徴量の型:")
            for name, dtype in list(summary.feature_types.items())[:20]:  # limit to 20
                lines.append(f"  - {name}: {dtype}")

        if summary.has_groups:
            lines.append(f"\nグループ列: {summary.group_col}")

        if summary.sample_preview:
            lines.append(f"\nプレビュー:\n{summary.sample_preview[:500]}")

        return "\n".join(lines)

    def _parse_interview_result(self, text: str) -> InterviewResult:
        """Parse LLM response into InterviewResult."""
        # Simplified parsing - in practice, use structured output
        result = InterviewResult()

        text_lower = text.lower()

        # Goal
        if "予測" in text or "prediction" in text_lower:
            result.goal = "prediction"
        elif "解釈" in text or "interpretation" in text_lower:
            result.goal = "interpretation"
        elif "実験計画" in text or "experimental" in text_lower:
            result.goal = "experimental_planning"

        # Prediction target
        if "近い" in text or "similar" in text_lower:
            result.prediction_target = "similar_samples"
        elif "遠い" in text or "far" in text_lower:
            result.prediction_target = "far_samples"

        # Optimization
        if "最大化" in text or "maximize" in text_lower:
            result.optimization_goal = "maximize"
        elif "最小化" in text or "minimize" in text_lower:
            result.optimization_goal = "minimize"

        # Monotonic strength
        if "strong" in text_lower or "強" in text:
            result.monotonic_strength = "strong"
        elif "weak" in text_lower or "弱" in text:
            result.monotonic_strength = "weak"

        return result


def run_full_workflow(
    data_summary: DataSummary,
    provider: Optional[LLMProvider] = None,
) -> Tuple[InterviewResult, str]:
    """
    Run the full LLM-guided workflow: assessment → interview → CV recommendation.
    """
    workflow = LLMWorkflow(provider)

    # Step 1: Data assessment
    assessment = workflow.assess_data(data_summary)

    # Step 2: Interview
    interview_result = workflow.conduct_interview(data_summary, assessment)

    # Step 3: CV recommendation
    cv_strategy = workflow.recommend_cv_strategy(interview_result, data_summary)
    interview_result.cv_strategy = cv_strategy

    return interview_result, assessment
