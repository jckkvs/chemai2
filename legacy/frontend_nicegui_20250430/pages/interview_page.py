"""
frontend_nicegui/pages/interview_page.py
LLM対話式ヒアリングページ - 機械学習初心者向けの対話型解析導線

20260429.txtの要件：
  - LLMがデータ概要を読み、解析目的をユーザーにヒアリング
  - 「どういうサンプルを予測したいか」「実験誤差は？」「制御変数vs成り行き変数」を対話で聞き出す
  - すべての設計でLLMが自動で判断するが、ユーザーも調整できる
"""

from nicegui import ui, app
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class InterviewPage:
    """
    LLM対話式ヒアリングページ

    ワークフロー：
      1. データ要約（LLMが自動生成）
      2. 解析目的のヒアリング（予測/解釈/実験計画）
      3. 予測対象サンプルの性質をヒアリング
      4. 実験誤差・変数の性質をヒアリング
      5. 特徴量選択・CV手法・単調性制約の確認
      6. 最終確認 → 解析開始
    """

    def __init__(self, navigate_to_automl=None, navigate_to_doe=None, navigate_to_llm=None):
        self.session = None  # InterviewSession（バックエンド）
        self.current_question = ""
        self.current_phase = ""
        self.qa_history: list[dict] = []
        self.navigate_to_automl = navigate_to_automl
        self.navigate_to_doe = navigate_to_doe
        self.navigate_to_llm = navigate_to_llm  # LLM Assistantへの遷移用

        # Analysis plan from LLM
        self.analysis_plan = None

        # UI参照
        self._question_label = None
        self._answer_input = None
        self._phase_label = None
        self._history_container = None
        self._next_btn = None
        self._progress_bar = None
        self._data_summary_card = None
        self._result_card = None
        self._plan_card = None  # Analysis plan display
        self._report_btn = None  # Report generation button

    def render(self):
        """ページをレンダリング"""
        with ui.column().classes('w-full max-w-4xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4 bg-gradient-to-r from-blue-50 to-indigo-50'):
                with ui.row().classes('w-full items-center gap-3'):
                    ui.icon('psychology', size='32px').classes('text-blue-600')
                    with ui.column().classes('gap-1'):
                        with ui.row().classes('items-center gap-2'):
                            ui.label('LLM対話式ヒアリング').classes('text-2xl font-bold text-gray-800')
                            ui.icon('help_outline').classes('text-blue-400 cursor-pointer').tooltip(
                                'このページでは、AI（LLM）が対話形式で解析方針を一緒に考えます。\n'
                                '機械学習の知識がなくても、AIが適切な方針を提案します。'
                            )
                        ui.label('機械学習初心者向けガイド - LLMが最適な解析方針を一緒に考えます').classes('text-sm text-gray-600')

            # 進捗バー
            with ui.card().classes('w-full mb-4'):
                self._phase_label = ui.label('準備中...').classes('text-sm text-gray-600 mb-2')
                self._progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')

            # データ要約カード
            self._data_summary_card = ui.card().classes('w-full mb-4')
            self._data_summary_card.visible = False
            with self._data_summary_card:
                ui.label('データ要約（LLM生成）').classes('font-bold text-lg mb-2')
                self._summary_label = ui.label('').classes('text-sm text-gray-700 whitespace-pre-wrap')

            # 質問カード
            with ui.card().classes('w-full mb-4 min-h-64'):
                ui.label('質問').classes('font-bold text-lg mb-2')
                self._question_label = ui.label('「Data Upload」タブでデータをアップロードしてください。').classes('text-base text-gray-800 whitespace-pre-wrap mb-4 min-h-32')

                ui.separator()

                with ui.row().classes('w-full items-end gap-2'):
                    with ui.column().classes('w-full'):
                        with ui.row().classes('items-center gap-1'):
                            ui.label('回答').classes('text-sm text-gray-600')
                            ui.icon('tips').tooltip(
                                '💡 ヒント：回答は詳しく書くほどLLMが適切な方針を立てやすくなります。\n'
                                '例：「温度を50度から100度まで10度刻みで変化させたい」\n'
                                '「分子量が大きいほど屈折率が高くなると予想される」'
                            )
                        self._answer_input = ui.textarea(
                            label='回答を入力してください',
                            placeholder='ここに回答を入力...\n例：「予測したいです。新しい化合物の物性を知りたい」',
                        ).classes('w-full').props('autogrow')
                    self._next_btn = ui.button(
                        '送信 →',
                        icon='send',
                        on_click=self._on_submit_answer,
                    ).props('color=primary')

            # ナビゲーションボタン
            with ui.row().classes('w-full justify-between mt-2'):
                self._back_btn = ui.button(
                    '← 戻る',
                    on_click=self._on_back,
                    icon='arrow_back',
                ).props('flat')
                self._back_btn.visible = False

                ui.space()

                self._skip_btn = ui.button(
                    'スキップ',
                    on_click=self._on_skip,
                    icon='skip_next',
                ).props('flat color=grey')
                self._skip_btn.visible = False

            # Q&A履歴
            self._history_container = ui.expansion(
                'Q&A履歴',
                icon='history',
            ).classes('w-full mt-4')
            with self._history_container:
                self._history_list = ui.column().classes('w-full gap-2')

            # 解析プラン表示カード
            self._plan_card = ui.card().classes('w-full mt-4 bg-blue-50')
            self._plan_card.visible = False
            with self._plan_card:
                ui.label('📋 LLM提案：解析プラン').classes('text-lg font-bold text-blue-800 mb-2')
                self._plan_label = ui.label('').classes('text-sm text-gray-700 whitespace-pre-wrap')

            # 結果カード（完了時）
            self._result_card = ui.card().classes('w-full mt-4 bg-green-50')
            self._result_card.visible = False
            with self._result_card:
                ui.label('✅ ヒアリング完了').classes('text-xl font-bold text-green-800 mb-2')
                self._result_label = ui.label('').classes('text-sm text-gray-700 whitespace-pre-wrap')
                with ui.row().classes('w-full gap-2 mt-4'):
                    self._start_automl_btn = ui.button(
                        '予測解析を開始',
                        icon='model_training',
                        on_click=self._on_start_automl,
                    ).props('color=primary')
                    self._start_doe_btn = ui.button(
                        '実験計画を開始',
                        icon='science',
                        on_click=self._on_start_doe,
                    ).props('color=teal')
                    self._report_btn = ui.button(
                        '📝 レポート生成へ',
                        icon='description',
                        on_click=self._on_go_to_report,
                    ).props('flat color=blue')
                    self._report_btn.visible = False

    async def start_interview(self, df: pd.DataFrame, target_hint: Optional[str] = None):
        """
        ヒアリングを開始（解析プラン提案付き）

        Args:
            df: 解析対象のDataFrame
            target_hint: 目的変数のヒント
        """
        try:
            from backend.llm.interview_session import InterviewSession, InterviewPhase

            self.session = InterviewSession()
            first_question = self.session.start(df, target_hint=target_hint)

            self.current_question = first_question
            self._update_phase_display()
            self._update_question_display()

            # データ要約を表示
            if self.session.context.data_summary:
                self._data_summary_card.visible = True
                self._summary_label.text = self.session.context.data_summary

            # 解析プランの提案を非同期で取得
            from nicegui import ui
            ui.notify("LLMが解析プランを提案中...", type="info")
            import asyncio
            try:
                self.analysis_plan = await asyncio.to_thread(
                    self._generate_analysis_plan, df
                )
                self._update_plan_display()
            except Exception as plan_err:
                logger.warning(f"Analysis plan generation failed: {plan_err}")

        except Exception as e:
            logger.error(f"Failed to start interview: {e}", exc_info=True)
            self._question_label.text = f"エラーが発生しました: {e}"

    def _update_phase_display(self):
        """フェーズ表示を更新"""
        if not self.session:
            return

        phase = self.session.current_phase
        phase_names = {
            InterviewPhase.DATA_SUMMARY: ("データ要約", 0.1),
            InterviewPhase.GOAL_CLARIFICATION: ("解析目的の確認", 0.2),
            InterviewPhase.SAMPLE_TARGET: ("予測対象の確認", 0.3),
            InterviewPhase.ERROR_UNDERSTANDING: ("実験誤差の確認", 0.4),
            InterviewPhase.VARIABLE_NATURE: ("変数の性質", 0.5),
            InterviewPhase.FEATURE_SELECTION: ("特徴量選択", 0.65),
            InterviewPhase.CV_STRATEGY: ("交差検証手法", 0.8),
            InterviewPhase.MONOTONICITY: ("単調性制約", 0.9),
            InterviewPhase.CONFIRMATION: ("最終確認", 0.95),
            InterviewPhase.COMPLETED: ("完了", 1.0),
        }

        name, progress = phase_names.get(phase, ("不明", 0.0))
        self.current_phase = name
        self._phase_label.text = f"フェーズ: {name}"
        self._progress_bar.value = progress

    def _update_question_display(self):
        """質問表示を更新"""
        self._question_label.text = self.current_question
        self._answer_input.value = ""
        self._answer_input.focus()

        # ボタンの表示制御
        if self.session and self.session.current_phase == InterviewPhase.CONFIRMATION:
            self._next_btn.text = "確定 ✅"
        elif self.session and self.session.current_phase == InterviewPhase.COMPLETED:
            self._next_btn.text = "完了 🎉"
            self._next_btn.visible = False
        else:
            self._next_btn.text = "送信 →"
            self._next_btn.visible = True

        # 戻るボタンの表示
        if self.session and len(self.session.qa_history) > 0:
            self._back_btn.visible = True
        else:
            self._back_btn.visible = False

    async def _on_submit_answer(self):
        """回答送信時の処理"""
        answer = self._answer_input.value
        if not answer or not answer.strip():
            ui.notify("回答を入力してください", type="warning")
            return

        if not self.session:
            ui.notify("セッションが開始されていません", type="error")
            return

        try:
            # 回答を送信し、次の質問を取得
            self._next_btn.disable()
            self._answer_input.disable()

            # 非同期でLLM呼び出し（裏で実行）
            import asyncio
            next_question = await asyncio.to_thread(
                self.session.submit_answer, answer.strip()
            )

            # 履歴に追加
            self.qa_history.append({
                "phase": self.current_phase,
                "question": self.current_question,
                "answer": answer.strip(),
            })
            self._update_history_display()

            self.current_question = next_question
            self._update_phase_display()
            self._update_question_display()

            # 完了チェック
            if self.session.is_completed():
                self._show_completion()

        except Exception as e:
            logger.error(f"Error submitting answer: {e}", exc_info=True)
            ui.notify(f"エラー: {e}", type="error")
        finally:
            self._next_btn.enble()
            self._answer_input.enble()

    def _update_history_display(self):
        """Q&A履歴を更新"""
        self._history_list.clear()
        for qa in self.qa_history:
            with self._history_list:
                with ui.card().classes('w-full bg-gray-50'):
                    ui.label(f"Q ({qa['phase']}):").classes('text-xs text-gray-500 font-bold')
                    ui.label(qa['question']).classes('text-sm text-gray-700')
                    ui.label(f"A: {qa['answer']}").classes('text-sm text-blue-700')

    def _on_back(self):
        """戻るボタン押下時"""
        if not self.session or len(self.session.qa_history) == 0:
            return

        # 履歴の最後を削除して再表示
        self.session.qa_history.pop()
        if self.qa_history:
            self.qa_history.pop()

        # セッションをリセットして最初からやり直す
        if hasattr(self, '_df_backup') and self._df_backup is not None:
            self.start_interview(self._df_backup)

    def _on_skip(self):
        """スキップボタン押下時"""
        self._answer_input.value = "スキップ"
        self._on_submit_answer()


    def _generate_analysis_plan(self, df: pd.DataFrame) -> Optional[dict]:
        """LLMにデータ特性に基づく解析プランを提案させる"""
        try:
            from backend.llm.analysis_advisor import AnalysisAdvisor
            advisor = AnalysisAdvisor()
            plan = advisor.suggest_plan(df, self.session.context if self.session else None)
            return plan
        except ImportError:
            # Fallback: simple rule-based plan
            return {
                "task_type": "regression" if df.select_dtypes(include='number').shape[1] > 0 else "classification",
                "recommended_models": ["RandomForest", "XGBoost", "SVR"],
                "cv_strategy": "5-fold CV",
                "feature_engineering": "Standard scaling + select top 20 features",
                "notes": "LLM advisor not available, using default plan."
            }
        except Exception as e:
            logger.warning(f"Failed to generate analysis plan: {e}")
            return None

    def _update_plan_display(self):
        """解析プラン表示を更新"""
        if not self.analysis_plan:
            return
        self._plan_card.visible = True
        lines = ["LLMからの解析プラン提案："]
        for key, value in self.analysis_plan.items():
            if isinstance(value, list):
                lines.append(f"{key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"{key}: {value}")
        self._plan_label.text = "\n".join(lines)

    def _show_completion(self):
        """完了時の表示"""
        if not self.session:
            return

        result = self.session.get_result()
        self._result_card.visible = True

        # 結果のサマリーを表示
        ctx = result.context
        lines = [
            f"解析目的: {ctx.user_goal or '未設定'}",
            f"予測対象: {ctx.prediction_target or '未設定'}",
            f"実験誤差: {ctx.experimental_error or '未設定'}",
            f"制御変数: {', '.join(ctx.controlled_vars) or '未設定'}",
            f"CV手法: {ctx.cv_strategy or '未設定'}",
        ]
        self._result_label.text = "\n".join(lines)

        # 解析プランも表示
        if self.analysis_plan:
            lines.append("\n=== LLM提案プラン ===")
            for key, value in self.analysis_plan.items():
                if isinstance(value, list):
                    lines.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    lines.append(f"{key}: {value}")
            self._result_label.text = "\n".join(lines)

        # 次のアクションに応じてボタンを制御
        if result.next_action == "start_doe":
            self._start_automl_btn.visible = False
            self._start_doe_btn.visible = True
        else:
            self._start_automl_btn.visible = True
            self._start_doe_btn.visible = False

        # レポート生成ボタンを表示
        if self._report_btn:
            self._report_btn.visible = True

    def _on_start_automl(self):
        """AutoML開始"""
        ui.notify("AutoMLページへ移動します", type="info")
        if self.navigate_to_automl:
            self.navigate_to_automl()

    def _on_start_doe(self):
        """DOE開始"""
        ui.notify("実験計画ページへ移動します", type="info")
        if self.navigate_to_doe:
            self.navigate_to_doe()


    def _on_go_to_report(self):
        """レポート生成へ遷移"""
        ui.notify("LLM Assistantのレポート生成へ移動します", type="info")
        if self.navigate_to_llm:
            self.navigate_to_llm()
        else:
            ui.notify("レポート生成機能はLLM Assistantタブで利用できます", type="warning")

    def set_data(self, df: pd.DataFrame, target_hint: Optional[str] = None):
        """外部からデータをセットして開始"""
        self._df_backup = df
        self.start_interview(df, target_hint=target_hint)
