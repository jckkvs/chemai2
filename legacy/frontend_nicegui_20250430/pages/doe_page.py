"""
frontend_nicegui/pages/doe_page.py
DOE (Design of Experiments) page - 実験計画法UI
要求仕様:
  - デフォルト: Maximin (D-optimalではない)
  - 手法: Maximin, Minimax, D-optimal, E-optimal, I-optimal, Sobol, Latin Hypercube
  - 因子: 連続値・カテゴリ両対応
  - 既存データからの設計 / ゼロからの設計
  - 多様な空間充填手法
"""

from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# バックエンドパス追加
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.doe.factor import Factor, FactorType
from backend.doe.candidate import generate_candidate_set
from backend.doe.design import DoEOptimizer, DoEResult


class FactorCard:
    """因子設定を保持するデータクラス（UI非依存）"""
    def __init__(self):
        self.name: str = "X1"
        self.type: str = "continuous"  # "continuous" or "categorical"
        self.low: float = 0.0
        self.high: float = 1.0
        self.levels: int = 5
        self.categories_str: str = "A, B, C"


class DoEPage:
    """実験計画法ページ"""

    def __init__(self):
        self.factor_cards: List[FactorCard] = []
        self.next_factor_idx: int = 1
        self.current_data: Optional[pd.DataFrame] = None
        self.design_result: Optional[DoEResult] = None
        # UI参照用
        self._factor_container = None
        self._result_container = None
        self._preview_container = None

    def render(self):
        """ページをレンダリング"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('実験計画法 (DOE)').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('Maximin / Minimax / D-opt / Sobol').classes('text-xs text-gray-500')

            # データ読み込みセクション
            self._render_data_section()

            # 因子設定セクション
            self._render_factor_section()

            # 設計パラメータ設定
            self._render_design_config()

            # 実行ボタン
            with ui.row().classes('w-full justify-center gap-4 mt-4'):
                self._run_btn = ui.button(
                    '生成実験計画',
                    on_click=self._run_doe,
                    color='primary',
                    icon='science'
                ).props('size=lg')
                self._cancel_btn = ui.button(
                    'クリア',
                    on_click=self._clear_results,
                    color='negative'
                ).props('outline')
                self._run_btn.disable()

            # プログレス
            self._progress_card = ui.card().classes('w-full mt-4')
            self._progress_card.visible = False
            with self._progress_card:
                ui.label('計算中...').classes('font-bold text-lg')
                self._progress_bar = ui.linear_progress(value=0, show_value=True)
                self._progress_log = ui.log().classes('w-full h-32 font-mono text-xs')

            # 結果表示
            self._result_card = ui.card().classes('w-full mt-4')
            self._result_card.visible = False
            with self._result_card:
                ui.label('実験計画結果').classes('font-bold text-lg mb-2')
                self._result_summary = ui.label('')
                self._result_table = ui.table(columns=[], rows=[]).classes('w-full')
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('CSV出力', on_click=self._export_csv, icon='download')
                    ui.button('可視化', on_click=self._visualize_design, icon='insights')

    def _render_data_section(self):
        """データ読み込みセクション"""
        with ui.card().classes('w-full mb-4'):
            ui.label('既存データ（オプション）').classes('font-bold text-lg mb-2')
            ui.label('既存の実験データがある場合は読み込むと、それらを固定した最適化ができます').classes('text-xs text-gray-500 mb-2')

            with ui.row().classes('w-full gap-4 items-end'):
                self._data_file = ui.upload(
                    on_upload=lambda e: self._handle_data_upload(e),
                    label='既存データをアップロード',
                    auto_upload=True,
                ).classes('flex-1')
                self._data_status = ui.label('未読み込み').classes('text-sm text-gray-500')

    def _render_factor_section(self):
        """因子設定セクション"""
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center'):
                ui.label('因子設定').classes('font-bold text-lg')
                ui.space()
                ui.button('因子を追加', on_click=self._add_factor, icon='add').props('dense')

            # 因子リスト表示エリア
            self._factor_container = ui.column().classes('w-full gap-2')

            # 初期因子を数個追加
            for _ in range(2):
                self._add_factor()

    def _add_factor(self):
        """因子を1つ追加"""
        card = FactorCard()
        card.name = f'X{self.next_factor_idx}'
        self.next_factor_idx += 1
        self.factor_cards.append(card)

        with self._factor_container:
            with ui.card().classes('w-full bg-gray-50') as ui_card:
                with ui.row().classes('w-full items-center gap-4'):
                    # 因子名
                    name_input = ui.input(
                        label='因子名',
                        value=card.name,
                        placeholder='Factor name'
                    ).classes('flex-1')
                    name_input.on_value_change(lambda v, c=card: setattr(c, 'name', v or f'X{self.next_factor_idx}'))

                    # タイプ選択
                    type_select = ui.select(
                        options={'continuous': '連続値', 'categorical': 'カテゴリ'},
                        value='continuous',
                        label='タイプ'
                    ).classes('w-32')
                    type_select.on_value_change(lambda v, c=card: setattr(c, 'type', v))

                    # 削除ボタン
                    ui.button(
                        icon='delete',
                        on_click=lambda _, c=card, ui_c=ui_card: self._remove_factor(c, ui_c),
                    ).props('flat dense color=negative')

                # 連続値用設定
                with ui.row().classes('w-full gap-4') as continuous_row:
                    low_input = ui.number(
                        label='下限',
                        value=0.0,
                        step=0.1,
                    ).classes('flex-1')
                    low_input.on_value_change(lambda v, c=card: setattr(c, 'low', float(v or 0.0)))

                    high_input = ui.number(
                        label='上限',
                        value=1.0,
                        step=0.1,
                    ).classes('flex-1')
                    high_input.on_value_change(lambda v, c=card: setattr(c, 'high', float(v or 1.0)))

                    levels_input = ui.number(
                        label='水準数',
                        value=5,
                        min=2,
                        max=20,
                        step=1,
                    ).classes('w-24')
                    levels_input.on_value_change(lambda v, c=card: setattr(c, 'levels', int(v or 5)))

                # カテゴリ用設定
                categories_input = ui.input(
                    label='カテゴリ値（カンマ区切り）',
                    placeholder='e.g., A, B, C',
                ).classes('w-full')
                categories_input.on_value_change(lambda v, c=card: setattr(c, 'categories_str', v or 'A, B, C'))

                def _on_type_change(e, cont=continuous_row, cat=categories_input):
                    is_cat = e.value == 'categorical'
                    cont.visible = not is_cat
                    cat.visible = is_cat
                type_select.on_value_change(_on_type_change)

    def _remove_factor(self, card, ui_card):
        """因子を削除"""
        if card in self.factor_cards:
            self.factor_cards.remove(card)
        ui_card.delete()
        ui.notify(f'因子 {card.name} を削除しました', type='info')

    def _render_design_config(self):
        """設計パラメータ設定"""
        with ui.card().classes('w-full mb-4'):
            ui.label('設計パラメータ').classes('font-bold text-lg mb-2')

            # 手法選択
            with ui.row().classes('w-full gap-4'):
                self._criterion = ui.select(
                    options={
                        'MAXIMIN': 'Maximin (デフォルト)',
                        'MINIMAX': 'Minimax',
                        'D': 'D-optimal',
                        'E': 'E-optimal',
                        'I': 'I-optimal',
                        'SOBOL': 'Sobol (空間充填)',
                        'LHS': 'Latin Hypercube',
                    },
                    value='MAXIMIN',
                    label='最適化基準'
                ).classes('flex-1')
                self._n_new = ui.number(
                    label='新規実験数',
                    value=10,
                    min=1,
                    max=1000,
                    step=1,
                ).classes('w-32')

            with ui.row().classes('w-full gap-4'):
                self._max_candidates = ui.number(
                    label='候補点数',
                    value=5000,
                    min=100,
                    max=50000,
                    step=100,
                ).classes('flex-1')
                self._n_starts = ui.number(
                    label='マルチスタート数',
                    value=5,
                    min=1,
                    max=20,
                    step=1,
                ).classes('w-32')
                self._max_iter = ui.number(
                    label='最大反復数',
                    value=300,
                    min=10,
                    max=2000,
                    step=10,
                ).classes('w-32')

    def _handle_data_upload(self, event):
        """既存データのアップロード処理"""
        try:
            import io
            content = event.content.read()
            if event.name.endswith('.csv'):
                self.current_data = pd.read_csv(io.BytesIO(content))
            elif event.name.endswith(('.xls', '.xlsx')):
                self.current_data = pd.read_excel(io.BytesIO(content))
            else:
                ui.notify('対応していないファイル形式です', type='warning')
                return

            self._data_status.text = f'{len(self.current_data)}行 x {len(self.current_data.columns)}列'
            self._run_btn.enable()
            ui.notify(f'データ読み込み完了: {len(self.current_data)}行', type='positive')
        except Exception as e:
            logger.error(f'Data upload error: {e}', exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _collect_factors(self) -> List[Factor]:
        """因子リストを収集"""
        factors = []
        for card in self.factor_cards:
            if card.type == 'continuous':
                factor = Factor.continuous(
                    name=card.name,
                    low=card.low,
                    high=card.high,
                    n_levels=card.levels,
                )
            else:
                cats = [c.strip() for c in card.categories_str.split(',')]
                factor = Factor.categorical(name=card.name, categories=cats)
            factors.append(factor)
        return factors

    async def _run_doe(self):
        """DOE実行"""
        if not self.factor_cards:
            ui.notify('因子を少なくとも1つ追加してください', type='warning')
            return

        self._run_btn.disable()
        self._progress_card.visible = True
        self._result_card.visible = False
        self._progress_bar.value = 0
        self._progress_log.clear()

        try:
            factors = self._collect_factors()
            if not factors:
                ui.notify('有効な因子がありません', type='warning')
                return

            n_new = int(self._n_new.value or 10)
            criterion = self._criterion.value
            max_cand = int(self._max_candidates.value or 5000)
            n_starts = int(self._n_starts.value or 5)
            max_iter = int(self._max_iter.value or 300)

            self._progress_log.push(f'[{self._now()}] 候補点生成中...')
            self._progress_bar.value = 10
            await asyncio.sleep(0.1)

            # Sobol / LHS の場合は専用メソッド
            if criterion == 'SOBOL':
                await self._run_sobol(factors, n_new)
                return
            elif criterion == 'LHS':
                await self._run_lhs(factors, n_new)
                return

            # 既存データの準備
            existing_df = None
            if self.current_data is not None:
                existing_df = self.current_data.copy()
                self._progress_log.push(f'[{self._now()}] 既存データ: {len(existing_df)}件')

            self._progress_log.push(f'[{self._now()}] 最適化中... (基準: {criterion})')
            self._progress_bar.value = 20
            await asyncio.sleep(0.1)

            # DoEOptimizerで最適化
            optimizer = DoEOptimizer(
                factors=factors,
                n_new=n_new,
                criterion=criterion,
                max_candidates=max_cand,
                n_starts=n_starts,
                max_iter=max_iter,
                existing_df=existing_df,
            )

            # 進捗コールバック
            def progress_cb(step, total, msg):
                val = 20 + int(70 * step / total)
                self._progress_bar.value = min(val, 90)
                self._progress_log.push(f'[{self._now()}] {msg}')

            # 別スレッドで実行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(optimizer.optimize)
                result = await asyncio.get_event_loop().run_in_executor(None, future.result)

            self.design_result = result
            self._display_results(result)

        except Exception as e:
            logger.error(f'DOE error: {e}', exc_info=True)
            self._progress_log.push(f'[{self._now()}] ❌ エラー: {str(e)}')
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_bar.value = 100
            self._progress_card.visible = False
            self._run_btn.enable()

    async def _run_sobol(self, factors: List[Factor], n_new: int):
        """Sobol準均等分布シーケンスで空間充填"""
        try:
            self._progress_log.push(f'[{self._now()}] Sobol序列生成中...')

            from scipy.stats.qmc import Sobol as SobolQMC

            n_dim = len(factors)
            sobol = SobolQMC(d=n_dim, scramble=True)
            samples = sobol.random(n=n_new)

            # [0,1] -> 各因子の範囲にスケーリング
            scaled = np.zeros((n_new, n_dim))
            factor_names = []
            for i, f in enumerate(factors):
                factor_names.append(f.name)
                if f.type == FactorType.CONTINUOUS:
                    low, high = f.low, f.high
                    scaled[:, i] = samples[:, i] * (high - low) + low
                else:
                    cats = f.categories
                    n_cat = len(cats)
                    idx = (samples[:, i] * n_cat).astype(int).clip(0, n_cat - 1)
                    scaled[:, i] = idx

            df = pd.DataFrame(scaled, columns=factor_names)
            df['_is_new'] = True

            self._display_simple_result(df, factor_names, 'Sobol')

        except ImportError:
            self._progress_log.push('❌ scipyが必要です')
            ui.notify('scipy.stats.qmcが必要です', type='warning')
        except Exception as e:
            self._progress_log.push(f'❌ エラー: {str(e)}')

    async def _run_lhs(self, factors: List[Factor], n_new: int):
        """Latin Hypercube Sampling"""
        try:
            self._progress_log.push(f'[{self._now()}] LHS生成中...')

            from scipy.stats.qmc import LatinHypercube

            n_dim = len(factors)
            lhs = LatinHypercube(d=n_dim, scramble=True)
            samples = lhs.random(n=n_new)

            scaled = np.zeros((n_new, n_dim))
            factor_names = []
            for i, f in enumerate(factors):
                factor_names.append(f.name)
                if f.type == FactorType.CONTINUOUS:
                    low, high = f.low, f.high
                    scaled[:, i] = samples[:, i] * (high - low) + low
                else:
                    cats = f.categories
                    n_cat = len(cats)
                    idx = (samples[:, i] * n_cat).astype(int).clip(0, n_cat - 1)
                    scaled[:, i] = idx

            df = pd.DataFrame(scaled, columns=factor_names)
            df['_is_new'] = True

            self._display_simple_result(df, factor_names, 'LHS')

        except ImportError:
            self._progress_log.push('❌ scipyが必要です')
            ui.notify('scipy.stats.qmcが必要です', type='warning')
        except Exception as e:
            self._progress_log.push(f'❌ エラー: {str(e)}')

    def _display_simple_result(self, df: pd.DataFrame, factor_names: List[str], method: str):
        """Sobol/LHS等の結果表示"""
        self._progress_bar.value = 100
        self._result_card.visible = True

        n_new = len(df)
        self._result_summary.text = f'手法: {method} | 新規: {n_new}件'

        columns = [{'name': c, 'label': c, 'field': c} for c in factor_names]
        rows = df[factor_names].head(100).to_dict('records')
        self._result_table.columns = columns
        self._result_table.rows = rows

        self._progress_log.push(f'[{self._now()}] ✅ {method}完了: {n_new}件')
        ui.notify(f'{method}完了: {n_new}件', type='positive')

    def _display_results(self, result: DoEResult):
        """結果を表示"""
        self._progress_bar.value = 100
        self._result_card.visible = True

        n_total = len(result.design_df)
        n_new = sum(result.is_new)
        n_existing = n_total - n_new

        summary = (f'基準: {result.criterion_name} | '
                  f'基準値: {result.criterion_value:.4f} | '
                  f'D効率: {result.d_efficiency:.4f} | '
                  f'新規: {n_new}件 / 既存: {n_existing}件')
        self._result_summary.text = summary

        # 結果テーブル
        df = result.design_df.copy()
        columns = [{'name': c, 'label': c, 'field': c} for c in df.columns]
        rows = df.head(200).to_dict('records')
        self._result_table.columns = columns
        self._result_table.rows = rows

        self._progress_log.push(f'[{self._now()}] ✅ 完了: {n_total}件')
        ui.notify(f'実験計画完了: {n_total}件', type='positive')

    def _clear_results(self):
        """結果をクリア"""
        self.design_result = None
        self._result_card.visible = False
        ui.notify('結果をクリアしました', type='info')

    def _export_csv(self):
        """結果をCSV出力"""
        if self.design_result is None:
            ui.notify('出力する結果がありません', type='warning')
            return
        try:
            from pathlib import Path
            out_dir = Path('exports')
            out_dir.mkdir(exist_ok=True)
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            path = out_dir / f'doe_design_{ts}.csv'
            self.design_result.design_df.to_csv(path, index=False)
            ui.notify(f'CSV出力: {path}', type='positive')
        except Exception as e:
            ui.notify(f'CSV出力エラー: {str(e)}', type='negative')

    def _visualize_design(self):
        """設計点を可視化"""
        if self.design_result is None:
            ui.notify('可視化する結果がありません', type='warning')
            return
        try:
            import plotly.express as px
            df = self.design_result.design_df
            if len(df.columns) >= 2:
                fig = px.scatter_matrix(
                    df,
                    dimensions=df.columns[:min(5, len(df.columns))],
                    title='実験計画の散布図行列',
                    color='_is_new' if '_is_new' in df.columns else None,
                )
                with ui.dialog() as dialog:
                    with ui.card().classes('w-full max-w-4xl'):
                        ui.plotly(fig)
                        ui.button('閉じる', on_click=dialog.close).classes('mt-2')
                dialog.open()
            else:
                ui.notify('可視化には2次元以上必要です', type='warning')
        except Exception as e:
            ui.notify(f'可視化エラー: {str(e)}', type='warning')

    @staticmethod
    def _now():
        from datetime import datetime
        return datetime.now().strftime('%H:%M:%S')
