"""
LLM対話式ヒアリング・セッション管理

20260429.txtの要件：
  - データが整形されている場合、LLMがデータ概要を読み、解析目的をユーザーにヒアリング
  - 「どういうサンプルを予測したいか」「実験誤差は？」「制御変数vs成り行き変数」などを対話で聞き出す
  - LLMが動的に質問を生成し、ユーザーの回答を踏まえて次の質問を導き出す

設計：
  - InterviewSession: セッション状態を管理（フェーズ進行、Q&A履歴、LLMプロンプト構築）
  - フェーズ: data_summary → goal_clarification → sample_target → error_understanding → variable_nature → feature_selection → cv_strategy → confirmation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.llm.provider import LLMProvider, LLMRequest
from backend.llm import get_llm_provider
from backend.data.auto_analyzer import AutoAnalyzer, AnalysisPlan

logger = logging.getLogger(__name__)


# ── フェーズ定義 ────────────────────────────────────────

class InterviewPhase(Enum):
    """ヒアリングのフェーズ"""

    INIT = "init"                          # 初期化
    DATA_SUMMARY = "data_summary"            # データ要約・LLMが読み込む
    GOAL_CLARIFICATION = "goal_clarification"  # 解析目的（予測/解釈/実験計画）
    SAMPLE_TARGET = "sample_target"            # 予測したいサンプルの性質
    ERROR_UNDERSTANDING = "error_understanding"  # 実験誤差・再現性
    VARIABLE_NATURE = "variable_nature"        # 制御変数vs成り行き変数
    FEATURE_SELECTION = "feature_selection"      # 特徴量選択方針
    CV_STRATEGY = "cv_strategy"                # 交差検証方針
    MONOTONICITY = "monotonicity"              # 単調性制約の確認
    CONFIRMATION = "confirmation"              # 最終確認
    COMPLETED = "completed"                     # 完了


# ── データクラス ──────────────────────────────────────────

@dataclass
class QAPair:
    """質問と回答のペア"""
    question: str
    answer: Optional[str] = None
    phase: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterviewContext:
    """セッション全体のコンテキスト"""
    n_samples: int = 0
    n_features: int = 0
    columns: List[str] = field(default_factory=list)
    target_col: Optional[str] = None
    task_type: Optional[str] = None
    has_smiles: bool = False
    smiles_col: Optional[str] = None
    data_summary: str = ""           # LLMが生成したデータ要約
    user_goal: Optional[str] = None  # ユーザーの解析目的
    prediction_target: Optional[str] = None  # どういうサンプルを予測したいか
    experimental_error: Optional[str] = None  # 実験誤差の程度
    controlled_vars: List[str] = field(default_factory=list)  # 制御できる変数
    circumstance_vars: List[str] = field(default_factory=list)  # 成り行き変数
    monotonic_constraints: Dict[str, str] = field(default_factory=dict)
    cv_strategy: Optional[str] = None
    feature_selection_notes: Optional[str] = None


@dataclass
class InterviewResult:
    """ヒアリング完了時の結果"""
    context: InterviewContext
    qa_history: List[QAPair]
    analysis_plan: Optional[AnalysisPlan] = None
    ready_to_proceed: bool = False
    next_action: Optional[str] = None  # "start_prediction" / "start_inverse" / "start_doe"


# ── メインクラス ──────────────────────────────────────────

class InterviewSession:
    """
    LLM対話式ヒアリング・セッション

    使用フロー:
        session = InterviewSession()
        session.start(df)  # データを渡して開始

        # ユーザーに質問を提示
        question = session.get_current_question()

        # ユーザーが回答を入力
        session.submit_answer(answer)

        # 次の質問を取得（自動進行）
        next_question = session.get_current_question()

        # 完了までループ
        while not session.is_completed():
            ...
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        auto_analyzer: Optional[AutoAnalyzer] = None,
    ):
        self.provider = provider or get_llm_provider("stub")
        if self.provider is None:
            from backend.llm.provider import StubLLMProvider
            self.provider = StubLLMProvider()

        self.analyzer = auto_analyzer or AutoAnalyzer()
        self.context = InterviewContext()
        self.qa_history: List[QAPair] = []
        self.current_phase = InterviewPhase.INIT
        self._df: Optional[Any] = None  # pandas DataFrame

    def start(self, df, target_hint: Optional[str] = None) -> str:
        """
        セッションを開始し、最初の質問を返す。

        Args:
            df: 解析対象のDataFrame
            target_hint: 目的変数のヒント

        Returns:
            最初の質問（データ要約＋解析目的の確認）
        """
        self._df = df
        self.context.n_samples = len(df)
        self.context.n_features = len(df.columns)
        self.context.columns = list(df.columns)

        # 基本分析
        plan = self.analyzer.create_analysis_plan(df, target_hint=target_hint)
        self.context.target_col = plan.target_column
        self.context.task_type = plan.task_type.value if plan.task_type else None

        # SMILES列検出
        for col in df.columns:
            if col.upper() in {"SMILES", "SMILE", "SMLS"}:
                self.context.smiles_col = col
                self.context.has_smiles = True
                break
            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if any(str(v).startswith("C") for v in sample):
                    self.context.smiles_col = col
                    self.context.has_smiles = True
                    break

        # データ要約をLLMで生成
        self.current_phase = InterviewPhase.DATA_SUMMARY
        self._generate_data_summary(df)
        self.current_phase = InterviewPhase.GOAL_CLARIFICATION

        return self.get_current_question()

    def _generate_data_summary(self, df):
        """LLMでデータ要約を生成"""
        try:
            prompt = self._build_data_summary_prompt(df)
            request = LLMRequest(
                user_prompt=prompt,
                system_prompt="あなたは化学・材料データの専門家です。データの要約を日本語で簡潔に作成してください。",
                max_tokens=1024,
                temperature=0.2,
            )
            response = self.provider.generate(request)
            self.context.data_summary = response.content.strip()
        except Exception as e:
            logger.warning(f"LLM data summary failed: {e}, using fallback")
            self.context.data_summary = self._fallback_data_summary(df)

    def _build_data_summary_prompt(self, df) -> str:
        """データ要約用プロンプトを構築"""
        cols_info = []
        for col in df.columns[:20]:  # 最初の20列まで
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            cols_info.append(f"  - {col} ({dtype}, ユニーク:{nunique}, 例:{sample})")

        return f"""以下のデータについて要約してください：

## データ基本情報
- サンプル数: {len(df)}
- 列数: {len(df.columns)}
- 欠損値: {df.isnull().sum().sum()}件

## 列情報
{chr(10).join(cols_info)}

## 指示
1. データの概要（何についてのデータか、構造の特徴）を2-3文で
2. 特徴量の種類（数値・カテゴリ・SMILES等）の分類
3. 解析の際の注意点（高相関、欠損、スケール等）

日本語で、機械学習初心者にも分かるように説明してください。
"""

    def _fallback_data_summary(self, df) -> str:
        """LLM失敗時のフォールバック要約"""
        numeric = len(df.select_dtypes(include=['number']).columns)
        categorical = len(df.select_dtypes(exclude=['number']).columns)
        return (
            f"データは{len(df)}サンプル、{len(df.columns)}列から構成されています。"
            f"数値列:{numeric}個、カテゴリ列:{categorical}個です。"
        )

    def get_current_question(self) -> str:
        """現在のフェーズに応じた質問を取得"""
        if self.current_phase == InterviewPhase.GOAL_CLARIFICATION:
            return self._q_goal_clarification()
        elif self.current_phase == InterviewPhase.SAMPLE_TARGET:
            return self._q_sample_target()
        elif self.current_phase == InterviewPhase.ERROR_UNDERSTANDING:
            return self._q_error_understanding()
        elif self.current_phase == InterviewPhase.VARIABLE_NATURE:
            return self._q_variable_nature()
        elif self.current_phase == InterviewPhase.FEATURE_SELECTION:
            return self._q_feature_selection()
        elif self.current_phase == InterviewPhase.CV_STRATEGY:
            return self._q_cv_strategy()
        elif self.current_phase == InterviewPhase.MONOTONICITY:
            return self._q_monotonicity()
        elif self.current_phase == InterviewPhase.CONFIRMATION:
            return self._q_confirmation()
        elif self.current_phase == InterviewPhase.COMPLETED:
            return "ヒアリングが完了しました。解析を開始できます。"
        return "セッションを開始してください。"

    def _q_goal_clarification(self) -> str:
        """解析目的を聞く（予測/解釈/実験計画）"""
        base = f"""データの要約：
{self.context.data_summary}

以下のうち、あなたの解析目的に最も近いものを教えてください：

1. **予測** - 新しいサンプルの値を予測したい
2. **解釈** - データから知見・仮説を得たい
3. **実験計画** - 次にどの実験をすべきか知りたい
4. **予測＋解釈** - 予測と同時に、なぜそうなるかを知りたい

※予測を選んだ場合、オプションとして**逆解析（最適条件の探索）**も自動的に含まれます。

どの目的で進めますか？"""
        return base

    def _q_sample_target(self) -> str:
        """予測したいサンプルの性質を聞く"""
        return """予測したいサンプルについて教えてください：

**過去のデータセットに近いサンプルを予測したい場合**：
  → 既存データの範囲内での予測になります（内挿）

**過去のデータセットから遠いサンプルを予測したい場合**：
  → 未知の領域への予測になります（外挿）

**両方のケースがある**：
  → 状況に応じて使い分けたい

どちらに近いでしょうか？また、予測したいサンプルは具体的にどういうものですか？"""

    def _q_error_understanding(self) -> str:
        """実験誤差について聞く"""
        return """同じ条件で実験したときの**実験誤差（再現性）**について教えてください：

1. **非常に小さい** - 同じ条件で実験しても数%以内の誤差
2. **中程度** - 同じ条件でも5-10%程度のばらつきがある
3. **大きい** - 同じ条件でもかなりのばらつきがある
4. **不明** - よく分からない

※この情報は、モデルの信頼性評価や実験計画の設計に使います。
※科学者は「完全に制御できる」と思っていても、実際には成り行き変数（制御できない変数）の影響で誤差が生じることがよくあります。ご注意ください。"""

    def _q_variable_nature(self) -> str:
        """変数の性質（制御可vs成り行き）を聞く"""
        numeric_cols = [c for c in self.context.columns
                       if c in self._df.columns and self._df[c].dtype in ['int64', 'float64']]
        numeric_cols = numeric_cols[:10]  # 最初の10個

        return f"""使用する変数の性質について教えてください：

**完全に制御できる変数**：実験者が意図的に設定できる変数
  （例：温度、時間、濃度、触媒量など）

**成り行き変数（制御できない変数）**：実験条件として設定できない変数
  （例：湿度、室温、試薬ロット、経年変化など）
  ※科学者でも「実質的には制御できる」と誤って回答することがあります。
  厳密に「実験者が意図的に値を決定できるか」でご判断ください。

以下の変数のうち、完全に制御できるものを教えてください：
{chr(10).join(f"  - {c}" for c in numeric_cols)}

※制御できる変数と成り行き変数の両方がある場合は、両方リストで教えてください。"""

    def _q_feature_selection(self) -> str:
        """特徴量選択の方針を聞く"""
        base = "特徴量の選択について：\n\n"

        if self.context.has_smiles:
            base += """このデータにはSMILES（分子構造）が含まれています。
**LLMが自動で最適な記述子（特徴量）を選択**することをお勧めします。

選択肢：
1. **LLMにお任せ** - 物理化学的知見に基づきLLMが最適な記述子を選ぶ
2. **相関係数で選択** - 目的変数との相関が高いものを選ぶ
3. **物理化学的特徴量セット** - 専門知識に基づく事前定義セットを使用
4. **全特徴量を使用** - 計算可能な全記述子を使用（次元が高くなる可能性あり）

どの方法で進めますか？"""
        else:
            base += """選択肢：
1. **LLMにお任せ** - データの性質に基づきLLMが最適な特徴量を選ぶ
2. **相関係数で選択** - 目的変数との相関が高いものを選ぶ
3. **全特徴量を使用** - そのまま全て使用
4. **手動で選択** - 自分で選ぶ

どの方法で進めますか？"""

        # サンプル数に応じた警告
        if self.context.n_samples > 0 and len(self.context.columns) > self.context.n_samples:
            base += f"\n\n⚠️ 警告：特徴量数({len(self.context.columns)})がサンプル数({self.context.n_samples})を上回っています。"
            base += "\n次元削減や特徴量選択が必要になる可能性が高いです。"

        return base

    def _q_cv_strategy(self) -> str:
        """交差検証の方針を聞く・提案"""
        # サンプル数に基づく推奨
        n = self.context.n_samples
        if n < 50:
            rec = "Leave-One-Out (各サンプルを1回ずつ検証に使用)"
        elif n < 200:
            rec = "5-Fold CV (データを5分割して検証)"
        elif self.context.has_smiles:
            rec = "Group K-Fold (化学構造の類似性を考慮)"
        else:
            rec = "5-Fold CV (標準的な分割)"

        return f"""交差検証（モデルの性能評価方法）について：

**推奨**：{rec}

予測したいサンプルの性質に応じて、適切な交差検証方法を選ぶ必要があります：

1. **KFold** - データをランダムに分割（標準的）
2. **GroupKFold** - SMILES等の類似性を考慮した分割（化学データ向け）
3. **Leave-One-Out** - 少量データ用（サンプル数が少ない場合）
4. **StratifiedKFold** - 分類タスク用（クラス比率を維持）

どの方法を使用しますか？また、{self.context.n_samples}サンプルに対して適切な分割数（3〜10）はいくつですか？"""

    def _q_monotonicity(self) -> str:
        """単調性制約について確認"""
        return """物理化学的な仮定に基づく**単調性制約**について：

多くの物性は、説明変数に対して単調な関係にあります（例：温度が上がる→反応速度が上がる）。

1. **絶対単調にする** - すべての変数で単調性を強く仮定（物理化学的制約を重視）
2. **強めの単調性制約** - 主要な変数で単調性を仮定
3. **弱めの単調性制約** - ゆるく単調性を仮定（データの傾向を優先）
4. **単調性制約なし** - 制約をかけない（純粋なデータ駆動）

※単調性制約をかけると、解釈性が高まり、外挿時の予測が安定します。
※学習データの±xシグマ範囲内でのみ制約を適用することも可能です。

どの方針で進めますか？"""

    def _q_confirmation(self) -> str:
        """最終確認"""
        goal_map = {
            "予測": "予測（回帰/分類）",
            "解釈": "データ解釈・知見抽出",
            "実験計画": "実験計画",
            "予測＋解釈": "予測＋解釈",
        }
        goal = goal_map.get(self.context.user_goal or "", self.context.user_goal or "未設定")

        return f"""ヒアリングの結果を確認してください：

## 解析設定
- **解析目的**: {goal}
- **予測対象**: {self.context.prediction_target or '未設定'}
- **実験誤差**: {self.context.experimental_error or '未設定'}
- **制御変数**: {', '.join(self.context.controlled_vars) or '未設定'}
- **成り行き変数**: {', '.join(self.context.circumstance_vars) or '未設定'}
- **特徴量選択**: {self.context.feature_selection_notes or '未設定'}
- **交差検証**: {self.context.cv_strategy or '未設定'}
- **単調性制約**: {self.context.monotonic_constraints or '未設定'}

この内容で解析を開始しますか？
「はい」で開始、「修正」または項目名で該当項目をやり直しできます。"""

    def submit_answer(self, answer: str) -> str:
        """
        ユーザーの回答を記録し、次の質問を返す。

        Returns:
            次の質問、または確認メッセージ
        """
        # 現在のフェーズの質問を記録
        qa = QAPair(
            question=self.get_current_question(),
            answer=answer,
            phase=self.current_phase.value,
        )
        self.qa_history.append(qa)

        # 回答をコンテキストに反映
        self._process_answer(answer)

        # フェーズ進行
        self._advance_phase()

        # 次の質問を返す
        return self.get_current_question()

    def _process_answer(self, answer: str):
        """回答をコンテキストに反映"""
        phase = self.current_phase
        answer_lower = answer.lower()

        if phase == InterviewPhase.GOAL_CLARIFICATION:
            self.context.user_goal = answer.strip()
            # 予測が含まれていれば次はsample_targetへ
            if "予測" in answer or "1" == answer.strip():
                self._next_phase = InterviewPhase.SAMPLE_TARGET
            elif "実験計画" in answer or "3" == answer.strip():
                self._next_phase = InterviewPhase.CV_STRATEGY
            else:
                self._next_phase = InterviewPhase.SAMPLE_TARGET

        elif phase == InterviewPhase.SAMPLE_TARGET:
            self.context.prediction_target = answer.strip()
            self._next_phase = InterviewPhase.ERROR_UNDERSTANDING

        elif phase == InterviewPhase.ERROR_UNDERSTANDING:
            self.context.experimental_error = answer.strip()
            self._next_phase = InterviewPhase.VARIABLE_NATURE

        elif phase == InterviewPhase.VARIABLE_NATURE:
            # 「制御できる：A, B」のような形式をパース
            controlled = []
            circumstance = []
            lines = answer.strip().split("\n")
            current_section = None
            for line in lines:
                if "制御" in line or "コントロール" in line.lower():
                    current_section = "controlled"
                    # 同一行に変数リストがある場合
                    vars_in_line = [c.strip("- ").strip() for c in line.split(",")]
                    controlled.extend([v for v in vars_in_line if v and v not in ["制御できる", "完全に制御できる変数"]])
                elif "成り行き" in line or " circumstance" in line.lower():
                    current_section = "circumstance"
                else:
                    # リスト形式の回答をパース
                    clean = line.strip("- ").strip()
                    if clean and current_section == "controlled":
                        controlled.append(clean)
                    elif clean and current_section == "circumstance":
                        circumstance.append(clean)

            # 数字選択のみの場合のデフォルト処理
            if not controlled and not circumstance:
                numeric_cols = [c for c in self.context.columns
                               if c in self._df.columns and self._df[c].dtype in ['int64', 'float64']]
                numeric_cols = numeric_cols[:10]
                # 回答から列名を抽出
                for col in numeric_cols:
                    if col in answer:
                        controlled.append(col)

            self.context.controlled_vars = controlled
            self.context.circumstance_vars = circumstance
            self._next_phase = InterviewPhase.FEATURE_SELECTION

        elif phase == InterviewPhase.FEATURE_SELECTION:
            self.context.feature_selection_notes = answer.strip()
            self._next_phase = InterviewPhase.CV_STRATEGY

        elif phase == InterviewPhase.CV_STRATEGY:
            self.context.cv_strategy = answer.strip()
            self._next_phase = InterviewPhase.MONOTONICITY

        elif phase == InterviewPhase.MONOTONICITY:
            self.context.monotonic_constraints = {"setting": answer.strip()}
            self._next_phase = InterviewPhase.CONFIRMATION

        elif phase == InterviewPhase.CONFIRMATION:
            if "はい" in answer or "yes" in answer_lower or "ok" in answer_lower:
                self._next_phase = InterviewPhase.COMPLETED
            else:
                # 修正項目を特定
                self._handle_modification(answer)

    def _handle_modification(self, answer: str):
        """修正要求を処理して該当フェーズに戻る"""
        answer_lower = answer.lower()
        if "目的" in answer or "goal" in answer_lower:
            self.current_phase = InterviewPhase.GOAL_CLARIFICATION
        elif "予測" in answer or "sample" in answer_lower or "ターゲット" in answer:
            self.current_phase = InterviewPhase.SAMPLE_TARGET
        elif "誤差" in answer or "error" in answer_lower or "実験" in answer:
            self.current_phase = InterviewPhase.ERROR_UNDERSTANDING
        elif "変数" in answer or "variable" in answer_lower or "制御" in answer:
            self.current_phase = InterviewPhase.VARIABLE_NATURE
        elif "特徴" in answer or "feature" in answer_lower:
            self.current_phase = InterviewPhase.FEATURE_SELECTION
        elif "交差" in answer or "cv" in answer_lower or "検証" in answer:
            self.current_phase = InterviewPhase.CV_STRATEGY
        elif "単調" in answer or "mono" in answer_lower:
            self.current_phase = InterviewPhase.MONOTONICITY
        else:
            # デフォルトは確認フェーズに留まる
            self._next_phase = InterviewPhase.CONFIRMATION

    def _advance_phase(self):
        """フェーズを進行"""
        if hasattr(self, '_next_phase'):
            self.current_phase = self._next_phase
            del self._next_phase
            return

        # デフォルトの進行順
        phase_order = [
            InterviewPhase.GOAL_CLARIFICATION,
            InterviewPhase.SAMPLE_TARGET,
            InterviewPhase.ERROR_UNDERSTANDING,
            InterviewPhase.VARIABLE_NATURE,
            InterviewPhase.FEATURE_SELECTION,
            InterviewPhase.CV_STRATEGY,
            InterviewPhase.MONOTONICITY,
            InterviewPhase.CONFIRMATION,
            InterviewPhase.COMPLETED,
        ]
        try:
            idx = phase_order.index(self.current_phase)
            if idx + 1 < len(phase_order):
                self.current_phase = phase_order[idx + 1]
        except ValueError:
            pass

    def is_completed(self) -> bool:
        return self.current_phase == InterviewPhase.COMPLETED

    def get_result(self) -> InterviewResult:
        """ヒアリング結果を取得"""
        # AnalysisPlanも生成
        plan = None
        if self._df is not None:
            try:
                plan = self.analyzer.create_analysis_plan(
                    self._df,
                    user_goal=self.context.user_goal,
                )
            except Exception as e:
                logger.warning(f"Failed to create analysis plan: {e}")

        # 次のアクションを決定
        goal = self.context.user_goal or ""
        if "予測" in goal:
            next_action = "start_prediction"
        elif "実験計画" in goal:
            next_action = "start_doe"
        elif "解釈" in goal:
            next_action = "start_interpretation"
        else:
            next_action = "start_prediction"

        return InterviewResult(
            context=self.context,
            qa_history=self.qa_history,
            analysis_plan=plan,
            ready_to_proceed=True,
            next_action=next_action,
        )

    def get_qa_history_text(self) -> str:
        """Q&A履歴をテキスト形式で取得（LLMへのコンテキストとして使用）"""
        lines = []
        for qa in self.qa_history:
            lines.append(f"Q({qa.phase}): {qa.question}")
            if qa.answer:
                lines.append(f"A: {qa.answer}")
            lines.append("")
        return "\n".join(lines)
