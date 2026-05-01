"""
frontend_nicegui/pages/decision_support_page.py
Decision Support page - 仕様書7章に基づく実装
アプリの核心：分析ではなく「意思決定」を支援
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DecisionSupportPage:
    """Decision Supportページ - ユーザーの意思決定を支援"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.task_type: str = 'regression'
        self.automl_result: Optional[Dict] = None
        self.domain_knowledge: Dict = {}

    def render(self):
        """Decision Supportページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('🎯 Decision Support - 意思決定支援').classes('text-2xl font-bold text-primary')
                    ui.space()
                    ui.label('分析が目的ではなく、次のアクションを決める').classes('text-gray-600')

            # データ未ロード時のメッセージ
            self._no_data_card = ui.card().classes('w-full mb-4')
            with self._no_data_card:
                ui.label('⚠️ データが読み込まれていません').classes('text-lg font-bold text-orange-600 mb-2')
                ui.label('「Data Upload」タブからデータをアップロードして、LLM Interviewで目標を設定してください。').classes('text-sm text-gray-600')
                ui.button('← Data Uploadへ', on_click=lambda: ui.navigate.to('/#data'), color='primary').props('outline')

            # データ状態サマリー
            self._data_summary = ui.card().classes('w-full mb-4')
            self._data_summary.visible = False
            with self._data_summary:
                ui.label('📊 現在のデータ状態').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('行数').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    with ui.column():
                        ui.label('目的変数').classes('text-xs text-gray-500')
                        self._target_label = ui.label('-')
                    with ui.column():
                        ui.label('タスクタイプ').classes('text-xs text-gray-500')
                        self._task_label = ui.label('-')
                    with ui.column():
                        ui.label('モデル状態').classes('text-xs text-gray-500')
                        self._model_status = ui.label('-')

            # 意思決定シナリオ選択
            self._scenario_card = ui.card().classes('w-full mb-4')
            self._scenario_card.visible = False
            with self._scenario_card:
                ui.label('🎯 意思決定シナリオ').classes('font-bold text-lg mb-2')
                ui.label('以下の4つの核心シナリオから選択、または自動判定を待つ').classes('text-sm text-gray-500 mb-4')

                # 4つのシナリオボタン
                with ui.row().classes('w-full gap-4'):
                    with ui.card().classes('flex-1 cursor-pointer hover:shadow-lg').props('clickable') as card1:
                        with ui.column().classes('p-2'):
                            ui.label('① 次の実験提案').classes('font-bold text-primary')
                            ui.label('次にどんな実験をすべきか？').classes('text-xs text-gray-600')
                            ui.button('実行', on_click=self._scenario_1_next_experiment, color='primary', size='sm').props('dense')

                    with ui.card().classes('flex-1 cursor-pointer hover:shadow-lg').props('clickable') as card2:
                        with ui.column().classes('p-2'):
                            ui.label('② 目標達成可能性').classes('font-bold text-green-600')
                            ui.label('この目標値は達成できそうか？').classes('text-xs text-gray-600')
                            ui.button('実行', on_click=self._scenario_2_feasibility, color='positive', size='sm').props('dense')

                    with ui.card().classes('flex-1 cursor-pointer hover:shadow-lg').props('clickable') as card3:
                        with ui.column().classes('p-2'):
                            ui.label('③ データ不足判定').classes('font-bold text-orange-600')
                            ui.label('今のデータで予測は信用できるか？').classes('text-xs text-gray-600')
                            ui.button('実行', on_click=self._scenario_3_data_sufficiency, color='warning', size='sm').props('dense')

                    with ui.card().classes('flex-1 cursor-pointer hover:shadow-lg').props('clickable') as card4:
                        with ui.column().classes('p-2'):
                            ui.label('④ テーマ撤退判断').classes('font-bold text-red-600')
                            ui.label('もうこのテーマでは無理かも？').classes('text-xs text-gray-600')
                            ui.button('実行', on_click=self._scenario_4_withdrawal, color='negative', size='sm').props('dense')

            # 自動判定ボタン
            with ui.row().classes('w-full justify-center mb-4'):
                ui.button('🔍 データを自動評価して最適なアクションを提案', on_click=self._auto_evaluate, color='primary', size='lg').props('glossy')

            # 評価結果エリア
            self._evaluation_card = ui.card().classes('w-full mb-4')
            self._evaluation_card.visible = False
            with self._evaluation_card:
                with ui.row().classes('w-full items-center'):
                    self._eval_title = ui.label('').classes('text-xl font-bold')
                    ui.space()
                    self._eval_timestamp = ui.label('').classes('text-xs text-gray-500')

                self._eval_content = ui.column().classes('w-full mt-4')

                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('📋 詳細レポート生成', on_click=self._generate_report, color='primary').props('outline')
                    ui.button('📊 この結果で次へ', on_click=self._proceed_to_next, color='primary')

            # 進行状況
            self._progress = ui.linear_progress(value=0, show_value=True).classes('w-full mt-4')
            self._progress.visible = False
            self._progress_label = ui.label('').classes('text-sm text-gray-500 mt-2')
            self._progress_label.visible = False

    def load_data(self, df: pd.DataFrame, target_col: Optional[str] = None, task_type: str = 'regression'):
        """データをロード"""
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type

        self._no_data_card.visible = False
        self._data_summary.visible = True
        self._scenario_card.visible = True

        self._row_count.text = f"{len(df):,}"
        self._target_label.text = target_col or '未設定'
        self._task_label.text = '回帰' if task_type == 'regression' else '分類'
        self._model_status.text = '未構築'

        ui.notify(f'✓ Decision Support: {len(df)}行のデータを読み込みました', type='positive')

    def set_automl_result(self, result: Dict):
        """AutoML結果を設定"""
        self.automl_result = result
        if hasattr(self, '_model_status'):
            best_model = result.get('best_model', 'Unknown')
            score = result.get('best_cv_score', 0)
            self._model_status.text = f"{best_model} (CV: {score:.4f})"

    def _auto_evaluate(self):
        """データを自動評価して最適なアクションを提案"""
        if self.df is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        self._progress.visible = True
        self._progress_label.visible = True
        self._progress.value = 10
        self._progress_label.text = '⏳ データを評価中...'

        try:
            # データ充足性を評価
            n_samples = len(self.df)
            n_features = len(self.df.select_dtypes(include=['number']).columns)

            self._progress.value = 30
            self._progress_label.text = '⏳ モデル性能を評価中...'

            # 簡易的なデータ充足性判定
            min_samples = 50  # 推奨最小サンプル数
            samples_per_feature = n_samples / n_features if n_features > 0 else n_samples

            self._progress.value = 50

            # 判定ロジック
            if self.automl_result:
                # モデルがある場合
                score = self.automl_result.get('best_cv_score', 0)
                if score > 0.8:  # 高い性能
                    scenario = 'next_experiment'
                    reason = f'モデル性能が高い（CV Score: {score:.4f}）。次の実験を提案します。'
                else:
                    scenario = 'data_sufficiency'
                    reason = f'モデル性能が不十分（CV Score: {score:.4f}）。データ追加が必要です。'
            else:
                # モデルがない場合
                if n_samples < min_samples:
                    scenario = 'data_sufficiency'
                    reason = f'サンプル数が不足（{n_samples}件 < 推奨{min_samples}件）。データ追加をお勧めします。'
                elif samples_per_feature < 5:
                    scenario = 'data_sufficiency'
                    reason = f'特徴量に対してサンプルが少ない（1特徴量あたり{samples_per_feature:.1f}件）。データ追加をお勧めします。'
                else:
                    scenario = 'next_experiment'
                    reason = f'データは充足しています（{n_samples}件）。次の実験を提案できます。'

            self._progress.value = 80
            self._progress_label.text = '⏳ 提案を生成中...'

            # 結果を表示
            self._evaluation_card.visible = True
            self._eval_title.text = '🔍 自動評価結果'
            self._eval_timestamp.text = f"評価時刻: {datetime.now().strftime('%H:%M:%S')}"

            self._eval_content.clear()
            with self._eval_content:
                ui.label('📊 評価サマリー').classes('font-bold text-lg mb-2')

                with ui.card().classes('w-full bg-blue-50 mb-4'):
                    ui.label(f"判定結果: {reason}").classes('text-sm text-blue-800')

                ui.label('📈 データ統計').classes('font-bold text-md mt-4 mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('サンプル数').classes('text-xs text-gray-500')
                        ui.label(f"{n_samples:,}").classes('text-lg font-bold')
                    with ui.column().classes('flex-1'):
                        ui.label('特徴量数').classes('text-xs text-gray-500')
                        ui.label(f"{n_features}").classes('text-lg font-bold')
                    with ui.column().classes('flex-1'):
                        ui.label('サンプル/特徴量').classes('text-xs text-gray-500')
                        ui.label(f"{samples_per_feature:.1f}").classes('text-lg font-bold')
                    with ui.column().classes('flex-1'):
                        ui.label('モデル状態').classes('text-xs text-gray-500')
                        ui.label('構築済' if self.automl_result else '未構築').classes('text-lg font-bold')

                ui.label('🎯 推奨アクション').classes('font-bold text-md mt-4 mb-2')
                if scenario == 'next_experiment':
                    with ui.card().classes('w-full bg-green-50'):
                        ui.label('✅ 次の実験提案（①）').classes('font-bold text-green-700')
                        ui.label('現在のモデルに基づき、目標達成のための次の実験条件を提案します。').classes('text-sm text-green-600')
                        ui.button('① 次の実験を提案', on_click=self._scenario_1_next_experiment, color='positive').classes('mt-2')
                elif scenario == 'data_sufficiency':
                    with ui.card().classes('w-full bg-orange-50'):
                        ui.label('⚠️ データ不足判定（③）').classes('font-bold text-orange-700')
                        ui.label('現在のデータでは予測の信頼性が不十分です。データ追加をお勧めします。').classes('text-sm text-orange-600')
                        ui.button('③ データ不足を評価', on_click=self._scenario_3_data_sufficiency, color='warning').classes('mt-2')

            self._progress.value = 100
            self._progress_label.text = '✓ 評価完了'

            ui.notify('✓ データ評価が完了しました', type='positive')

        except Exception as e:
            logger.error(f"自動評価エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress.visible = False
            self._progress_label.visible = False

    def _scenario_1_next_experiment(self):
        """① 次の実験提案"""
        self._evaluation_card.visible = True
        self._eval_title.text = '① 次の実験提案'
        self._eval_timestamp.text = f"実行時刻: {datetime.now().strftime('%H:%M:%S')}"

        self._eval_content.clear()
        with self._eval_content:
            ui.label('🎯 次の実験提案').classes('font-bold text-lg mb-2')

            if self.automl_result:
                best_model = self.automl_result.get('best_model', 'Unknown')
                score = self.automl_result.get('best_cv_score', 0)

                with ui.card().classes('w-full bg-green-50'):
                    ui.label('✅ モデルが信頼できる場合の提案').classes('font-bold text-green-700 mb-2')
                    ui.markdown(f"""
                    **手法**: 逆解析（ベイズ最適化、遺伝的アルゴリズム等）
                    **目的**: 目標達成のための条件を探索
                    **現在のモデル**: {best_model} (CV Score: {score:.4f})

                    *ベイズ最適化で「目標達成のための条件」を探索します*
                    """)
                    ui.button('🔬 逆解析を実行（準備中）', color='positive').props('outline')
            else:
                with ui.card().classes('w-full bg-orange-50'):
                    ui.label('⚠️ モデルが未構築').classes('font-bold text-orange-700 mb-2')
                    ui.markdown("""
                    **提案**: まずAutoMLでモデルを構築してください。
                    モデルが信頼できない場合は、実験計画法（DOE）でデータを補完します。
                    """)
                    ui.button('← ML Modelingへ', on_click=lambda: ui.navigate.to('/#ml'), color='warning').props('outline')

            # DOE提案
            with ui.card().classes('w-full mt-4'):
                ui.label('📐 データ不足時の代替案（DOE）').classes('font-bold text-md mb-2')
                ui.markdown("""
                **手法**: 実験計画法（Maximin, Sobol等）
                **目的**: モデル精度向上のための実験設計
                **推奨**: 空間充填型（Maximin）で情報量を最大化
                """)
                ui.button('📐 DOEタブへ移動', on_click=lambda: ui.navigate.to('/#doe'), color='primary').props('outline')

    def _scenario_2_feasibility(self):
        """② 目標達成可能性評価"""
        self._evaluation_card.visible = True
        self._eval_title.text = '② 目標達成可能性評価'
        self._eval_timestamp.text = f"実行時刻: {datetime.now().strftime('%H:%M:%S')}"

        self._eval_content.clear()
        with self._eval_content:
            ui.label('🎯 目標達成可能性評価').classes('font-bold text-lg mb-2')

            if self.automl_result:
                score = self.automl_result.get('best_cv_score', 0)

                # 達成確率（簡易計算）
                if score > 0.9:
                    prob = '90%以上'
                    level = 'text-green-600'
                    icon = '✅'
                elif score > 0.7:
                    prob = '70-90%'
                    level = 'text-blue-600'
                    icon = '🔄'
                else:
                    prob = '30%未満'
                    level = 'text-red-600'
                    icon = '⚠️'

                with ui.card().classes('w-full bg-blue-50'):
                    ui.label(f'{icon} 達成確率: {prob}').classes(f'font-bold text-lg {level} mb-2')
                    ui.markdown(f"""
                    **現在のモデル性能**: CV Score {score:.4f}
                    **達成確率**: {prob}
                    **現実的目標値**: 95%以上の確率で達成可能な値を再設定

                    *LLMが詳細な分析とボトルネックを特定します*
                    """)

                # ボトルネック特定（簡易版）
                with ui.card().classes('w-full mt-4'):
                    ui.label('🔍 ボトルネック特定').classes('font-bold text-md mb-2')
                    ui.label('現在の簡易評価では特定できません。LLMレポートで詳細分析を。').classes('text-sm text-gray-500')
                    ui.button('📋 LLMレポート生成', on_click=self._generate_report, color='primary').props('outline')
            else:
                with ui.card().classes('w-full bg-orange-50'):
                    ui.label('⚠️ モデルが未構築').classes('font-bold text-orange-700 mb-2')
                    ui.label('目標達成可能性を評価するには、まずモデルを構築してください。').classes('text-sm text-orange-600')
                    ui.button('← ML Modelingへ', on_click=lambda: ui.navigate.to('/#ml'), color='warning').props('outline')

    def _scenario_3_data_sufficiency(self):
        """③ データ不足判定"""
        self._evaluation_card.visible = True
        self._eval_title.text = '③ データ不足判定'
        self._eval_timestamp.text = f"実行時刻: {datetime.now().strftime('%H:%M:%S')}"

        self._eval_content.clear()
        with self._eval_content:
            ui.label('🎯 データ充足性評価').classes('font-bold text-lg mb-2')

            if self.df is None:
                ui.label('データが読み込まれていません').classes('text-red-500')
                return

            n_samples = len(self.df)
            n_features = len(self.df.select_dtypes(include=['number']).columns)
            min_samples = 50
            samples_per_feature = n_samples / n_features if n_features > 0 else 0

            # 判定
            issues = []
            if n_samples < min_samples:
                issues.append(f'サンプル数が不足（{n_samples} < 推奨{min_samples}）')
            if samples_per_feature < 5:
                issues.append(f'特徴量に対してサンプルが少ない（{samples_per_feature:.1f} < 5）')

            if issues:
                with ui.card().classes('w-full bg-red-50'):
                    ui.label('✗ 現時点では予測モデルの信頼は困難').classes('font-bold text-red-700 mb-2')
                    for issue in issues:
                        ui.label(f'• {issue}').classes('text-sm text-red-600')

                    ui.markdown(f"""
                    **提案**:
                    1. モデル構築は保留し、まずDOEで{n_samples}実験を追加
                    2. 特に重要な特徴量の範囲を埋める実験を
                    3. その後再度評価 → モデル信頼度が向上すれば予測へ
                    """)

                    ui.button('📐 DOEタブへ移動', on_click=lambda: ui.navigate.to('/#doe'), color='warning').classes('mt-2')
            else:
                with ui.card().classes('w-full bg-green-50'):
                    ui.label('✅ データは概ね充足しています').classes('font-bold text-green-700 mb-2')
                    ui.label(f'サンプル数: {n_samples:,}（推奨{min_samples}以上）').classes('text-sm text-green-600')
                    ui.label(f'サンプル/特徴量: {samples_per_feature:.1f}（推奨5以上）').classes('text-sm text-green-600')

    def _scenario_4_withdrawal(self):
        """④ テーマ再検討"""
        self._evaluation_card.visible = True
        self._eval_title.text = '④ テーマ再検討'
        self._eval_timestamp.text = f"実行時刻: {datetime.now().strftime('%H:%M:%S')}"

        self._eval_content.clear()
        with self._eval_content:
            ui.label('🎯 テーマ再検討').classes('font-bold text-lg mb-2')

            if self.automl_result:
                score = self.automl_result.get('best_cv_score', 0)

                if score < 0.5:
                    with ui.card().classes('w-full bg-red-50'):
                        ui.label('⚠️ 撤退を検討すべき時期かもしれません').classes('font-bold text-red-700 mb-2')
                        ui.markdown(f"""
                        **理由**:
                        1. モデル性能が低い（CV Score: {score:.4f}）
                        2. 物理化学的限界の可能性
                        3. コスト・時間に対して見込める成果が見合わない

                        **提案**：
                        - 方向転換①: 全く新しい骨格構造の導入を検討
                        - 方向転換②: 目的変数を別の指標に変更
                        - 撤退: このテーマは現時点では技術的に困難
                        """)
                else:
                    with ui.card().classes('w-full bg-green-50'):
                        ui.label('✅ まだ撤退する必要はなさそうです').classes('font-bold text-green-700 mb-2')
                        ui.label(f'モデル性能: {score:.4f}（まずまずの水準）').classes('text-sm text-green-600')
            else:
                with ui.card().classes('w-full bg-orange-50'):
                    ui.label('⚠️ 判断にはモデル構築が必要です').classes('font-bold text-orange-700 mb-2')
                    ui.button('← ML Modelingへ', on_click=lambda: ui.navigate.to('/#ml'), color='warning').props('outline')

    def _generate_report(self):
        """LLMレポート生成（簡易版）"""
        ui.notify('📋 LLMレポート生成中...', type='info')

        # 簡易的なレポート生成（実際はLLMを呼び出す）
        report = f"""
        # 意思決定レポート（簡易版）

        ## 1. 現状評価
        - データ数: {len(self.df) if self.df is not None else 0}サンプル
        - モデル状態: {'構築済' if self.automl_result else '未構築'}
        - 達成可能性: 要評価

        ## 2. 推奨アクション
        1. データ充足性を確認
        2. モデル構築（未の場合）
        3. 目標達成可能性を評価

        ## 3. 結論
        LLMによる詳細な提言は、LLM設定を有効にしてから再度実行してください。
        """

        with ui.dialog() as dialog:
            with ui.card().classes('w-full max-w-4xl'):
                ui.label('📋 意思決定レポート').classes('text-xl font-bold mb-4')
                ui.markdown(report)
                with ui.row().classes('w-full justify-end'):
                    ui.button('閉じる', on_click=dialog.close).props('outline')

        dialog.open()
        ui.notify('✓ レポートを生成しました', type='positive')

    def _proceed_to_next(self):
        """次のステップへ遷移"""
        ui.notify('次のステップへ進みます', type='info')
        # 現在はML Modelingタブへ遷移
        ui.navigate.to('/#ml')
