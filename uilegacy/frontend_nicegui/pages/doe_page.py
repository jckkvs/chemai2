"""
frontend_nicegui/pages/doe_page.py
Design of Experiments (DOE) page - 仕様書8章に基づく実装
実験計画法：データ不足時の補完、モデル精度向上のための手段
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DOEPPage:
    """Design of Experiments page - 意思決定支援のための手段"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_col: Optional[str] = None

    def render(self):
        """DOEページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('📐 Design of Experiments (DOE)').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('実験計画法：意思決定のための手段').classes('text-gray-600')

            # データ未ロード時
            self._no_data = ui.card().classes('w-full mb-4')
            with self._no_data:
                ui.label('⚠️ データが読み込まれていません').classes('text-lg font-bold text-orange-600 mb-2')
                ui.label('「Data Upload」タブからデータをアップロードしてください。').classes('text-sm text-gray-600')
                ui.button('← Data Uploadへ', on_click=lambda: ui.navigate.to('/#data'), color='primary').props('outline')

            # データ状態
            self._data_card = ui.card().classes('w-full mb-4')
            self._data_card.visible = False
            with self._data_card:
                ui.label('📂 現在のデータ状態').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('行数').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    with ui.column():
                        ui.label('数値列').classes('text-xs text-gray-500')
                        self._numeric_count = ui.label('-')
                    with ui.column():
                        ui.label('カテゴリ列').classes('text-xs text-gray-500')
                        self._cat_count = ui.label('-')

            # 用途別DOE戦略 (仕様書8.1)
            self._strategy_card = ui.card().classes('w-full mb-4')
            self._strategy_card.visible = False
            with self._strategy_card:
                ui.label('🎯 用途別DOE戦略').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **① データ不足 → モデル構築**
                Maximin, Sobol（空間充填）: 情報量最大化・空間カバー率向上

                **② モデル精度向上 → 意思決定支援**
                D最適, I最適: 予測分散最小化・信頼区間縮小

                **③ ゼロから最適条件探索**
                直交表（Taguchi等）: 要因の系統的把握

                **④ 既存データの補完**
                既存データ固定＋Maximin: 既知の隙間を埋める
                """)

            # DOE手法選択
            self._method_card = ui.card().classes('w-full mb-4')
            self._method_card.visible = False
            with self._method_card:
                ui.label('🔬 DOE手法選択').classes('font-bold text-lg mb-2')

                # パターン選択
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('実験パターン').classes('text-xs text-gray-500')
                        self._pattern_select = ui.select(
                            options={
                                'from_scratch': 'ゼロから計画（全ての範囲・水準を設定）',
                                'complement': '既存実験データを読み込み、それに基づき補完',
                            },
                            value='complement'
                        ).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('DOE手法（デフォルト：Maximin）').classes('text-xs text-gray-500')
                        self._method_select = ui.select(
                            options={
                                # 空間充填型（メイン：データ不足時の補完）
                                'maximin': '🥇 Maximin (デフォルト推奨)',
                                'minimax': '🥈 Minimax',
                                'sobol': '🥉 Sobol (準乱数)',
                                # 最適基準型（モデル精度向上時）
                                'd_optimal': '🥊 D最適 (パラメータ推定精度向上)',
                                'e_optimal': '🥋 E最適 (最大固有値最小化)',
                                'i_optimal': '🥌 I最適 (予測分散の積分最小化)',
                                # 直交表
                                'orthogonal': '🥍 直交表 (Taguchi法等)',
                            },
                            value='maximin'
                        ).classes('w-full')

                # サンプル数
                with ui.row().classes('w-full gap-4 mt-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('生成する実験数').classes('text-xs text-gray-500')
                        self._n_experiments = ui.number(value=20, min=5, max=100).classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('水準数が多い場合').classes('text-xs text-gray-500')
                        ui.label('全空間探索ではなく、ランダムに範囲内を探索').classes('text-xs text-gray-500')

            # 変数範囲設定
            self._variables_card = ui.card().classes('w-full mb-4')
            self._variables_card.visible = False
            with self._variables_card:
                ui.label('📋 変数範囲・水準設定').classes('font-bold text-lg mb-2')
                ui.label('数値変数は範囲、カテゴリ変数は値を選択').classes('text-sm text-gray-500 mb-4')

                # 変数設定エリア
                self._var_container = ui.column().classes('w-full')

            # 実行ボタン
            self._run_card = ui.card().classes('w-full mb-4')
            self._run_card.visible = False
            with self._run_card:
                with ui.row().classes('w-full justify-center gap-4'):
                    self._run_btn = ui.button('📐 実験計画を生成', on_click=self._run_doe, color='primary', size='lg')
                    self._cancel_btn = ui.button('キャンセル', on_click=self._cancel, color='negative').props('outline')

            # 進捗表示
            self._progress_card = ui.card().classes('w-full mt-4')
            self._progress_card.visible = False
            with self._progress_card:
                ui.label('⏳ 実験計画生成中...').classes('font-bold')
                self._progress = ui.linear_progress(value=0, show_value=True)

            # 結果表示
            self._result_card = ui.card().classes('w-full mt-4')
            self._result_card.visible = False
            with self._result_card:
                with ui.row().classes('w-full items-center'):
                    ui.label('✅ 実験計画生成完了').classes('font-bold text-lg text-green-600')
                    ui.space()
                    self._result_label = ui.label('')

                # 生成された実験条件
                self._result_table = ui.table(
                    columns=[],
                    rows=[],
                    pagination=dict(rowsPerPage=10)
                ).classes('w-full mt-4')

                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('📋 CSV出力', on_click=self._export_csv, color='primary').props('outline')
                    ui.button('📊 Excel出力', on_click=self._export_excel, color='primary').props('outline')
                    ui.button('📋 次の実験リストとして保存', on_click=self._save_as_next_experiments, color='positive')

            # LLM提案
            self._llm_card = ui.card().classes('w-full mb-4')
            self._llm_card.visible = False
            with self._llm_card:
                ui.label('🤖 LLMによるDOE提案').classes('font-bold text-lg mb-2')
                ui.markdown("""
                **LLMがデータ状態に基づき最適なDOE手法を提案：**

                - 今はデータ不足だから、DOEで空間カバー率を上げる実験を推奨
                - モデル精度向上が必要ならD最適やI最適を提案
                - ゼロからなら直交表で要因を系統的把握
                """)
                ui.button('💬 LLMに相談', on_click=self._consult_llm, color='primary').props('outline')

    def load_data(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """データをロード"""
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # UI更新
        self._no_data.visible = False
        self._data_card.visible = True
        self._strategy_card.visible = True
        self._method_card.visible = True
        self._variables_card.visible = True
        self._run_card.visible = True
        self._llm_card.visible = True

        self._row_count.text = f"{len(df):,}"
        self._numeric_count.text = f"{len(self.numeric_cols)}"
        self._cat_count.text = f"{len(self.categorical_cols)}"

        # 変数設定UIを構築
        self._build_variable_ui()

        ui.notify(f'✓ DOE: {len(df)}行のデータを読み込みました', type='positive')

    def _build_variable_ui(self):
        """変数範囲・水準設定UIを構築"""
        self._var_container.clear()

        with self._var_container:
            # 数値変数：範囲設定（スライダー）
            if self.numeric_cols:
                ui.label('数値変数の範囲設定').classes('font-bold text-sm mb-2')
                for col in self.numeric_cols[:10]:  # 最大10列
                    col_data = self.df[col].dropna()
                    if len(col_data) == 0:
                        continue
                    min_val, max_val = float(col_data.min()), float(col_data.max())
                    if min_val == max_val:
                        continue

                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label(col).classes('w-32 text-xs')
                        slider = ui.slider(min=min_val, max=max_val, value=[min_val, max_val])
                        slider.props('label-always range').classes('flex-1')

            # カテゴリ変数：値選択
            if self.categorical_cols:
                ui.separator()
                ui.label('カテゴリ変数の値選択').classes('font-bold text-sm mb-2')
                for col in self.categorical_cols[:5]:  # 最大5列
                    unique_vals = self.df[col].dropna().unique().tolist()
                    if len(unique_vals) > 20:
                        continue

                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label(col).classes('w-32 text-xs')
                        select = ui.select(
                            options=unique_vals,
                            value=unique_vals[0] if unique_vals else None,
                            multiple=True
                        ).classes('flex-1')

    def _run_doe(self):
        """DOE実行"""
        if self.df is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        self._progress_card.visible = True
        self._progress.value = 20
        self._result_card.visible = False

        try:
            # 簡易的なDOE実装（実際はbackend/doe/を呼び出す）
            method = self._method_select.value
            n_exp = int(self._n_experiments.value)

            self._progress.value = 40

            # サンプル生成（ダミー）
            if self.numeric_cols:
                # 既存データの範囲を取得
                ranges = {}
                for col in self.numeric_cols[:5]:  # 最大5列
                    col_data = self.df[col].dropna()
                    if len(col_data) > 0:
                        ranges[col] = (float(col_data.min()), float(col_data.max()))

                # ランダムサンプリング（簡易版）
                np.random.seed(42)
                samples = []
                for i in range(n_exp):
                    sample = {}
                    for col, (min_val, max_val) in ranges.items():
                        sample[col] = np.random.uniform(min_val, max_val)
                    samples.append(sample)

                self._progress.value = 70

                # 結果をテーブル表示
                if samples:
                    columns = [{'name': col, 'label': col, 'field': col} for col in samples[0].keys()]
                    rows = [{'id': i+1, **s} for i, s in enumerate(samples)]

                    self._result_table.columns = columns
                    self._result_table.rows = rows
                    self._result_label.text = f"{len(samples)}件の実験条件を生成"

                self._progress.value = 100
                self._result_card.visible = True
                ui.notify(f'✓ {n_exp}件の実験条件を生成しました', type='positive')

            else:
                ui.notify('数値列がありません', type='warning')

        except Exception as e:
            logger.error(f"DOE実行エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _cancel(self):
        """キャンセル"""
        ui.notify('キャンセルしました', type='info')

    def _export_csv(self):
        """CSV出力"""
        ui.notify('CSV出力（準備中）', type='info')

    def _export_excel(self):
        """Excel出力"""
        ui.notify('Excel出力（準備中）', type='info')

    def _save_as_next_experiments(self):
        """次の実験リストとして保存"""
        ui.notify('次の実験リストとして保存（準備中）', type='info')

    def _consult_llm(self):
        """LLMに相談"""
        ui.notify('LLM相談（準備中）', type='info')
