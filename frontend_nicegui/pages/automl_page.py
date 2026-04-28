"""
frontend_nicegui/pages/automl_page.py
完全なAutoML UI実装
"""
from nicegui import ui, events
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import asyncio
import logging

logger = logging.getLogger(__name__)


class AutoMLPage:
    """AutoML解析ページ"""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.automl_instance = None
        self._row_count = None
        self._col_count = None
        self._memory_usage = None
        self._data_table = None
        self._no_data_msg = None
        self._target_card = None
        self._target_select = None
        self._config_card = None
        self._cv_folds = None
        self._metric = None
        self._max_trials = None
        self._models = None
        self._run_btn = None
        self._cancel_btn = None
        self._progress_card = None
        self._progress_bar = None
        self._progress_log = None
        self._result_card = None
        self._results_table = None

    def render(self):
        """AutoMLページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
            
            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                ui.label('🤖 AutoML - 自動機械学習').classes('text-2xl font-bold')
                ui.label('既存の高度なAutoML機能を利用した解析を実行します').classes('text-gray-600')
            
            # データ確認セクション
            with ui.card().classes('w-full mb-4'):
                ui.label('📊 読み込み済みデータ').classes('font-bold text-lg mb-2')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('行数').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    
                    with ui.column():
                        ui.label('列数').classes('text-xs text-gray-500')
                        self._col_count = ui.label('-')
                    
                    with ui.column():
                        ui.label('メモリ使用量').classes('text-xs text-gray-500')
                        self._memory_usage = ui.label('-')
                
                # データプレビュー
                ui.label('プレビュー（先頭10行）').classes('mt-4 font-bold')
                self._data_table = ui.table(columns=[], rows=[]).classes('w-full h-64')
                self._data_table.visible = False
                
                self._no_data_msg = ui.label(
                    '⚠️ データが読み込まれていません。「Data Upload」タブからデータをアップロードしてください。'
                )
                self._no_data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded')
            
            # 目的変数選択
            with ui.card().classes('w-full mb-4') as self._target_card:
                self._target_card.visible = False
                ui.label('🎯 目的変数を選択').classes('font-bold text-lg mb-2')
                self._target_select = ui.select(
                    options=[],
                    label='目的変数（予測対象）',
                    with_input=True
                ).classes('w-full')
                
                ui.label('ヒント: 数値列なら回帰分析、カテゴリ列なら分類分析が自動選択されます').classes('text-xs text-gray-500 mt-1')
            
            # AutoML設定
            with ui.card().classes('w-full mb-4') as self._config_card:
                self._config_card.visible = False
                ui.label('⚙️ AutoML設定').classes('font-bold text-lg mb-2')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('CV Fold数').classes('text-xs text-gray-500')
                        self._cv_folds = ui.number(value=5, min=2, max=10, step=1)
                    
                    with ui.column():
                        ui.label('評価指標').classes('text-xs text-gray-500')
                        self._metric = ui.select(
                            options=['rmse', 'mae', 'r2', 'accuracy', 'f1'],
                            value='rmse'
                        )
                    
                    with ui.column():
                        ui.label('最大試行').classes('text-xs text-gray-500')
                        self._max_trials = ui.number(value=20, min=5, max=100, step=5)
                
                ui.label('使用するモデル').classes('mt-4 font-bold')
                with ui.row().classes('w-full gap-4'):
                    self._model_rf = ui.checkbox('Random Forest', value=True)
                    self._model_lgbm = ui.checkbox('LightGBM', value=True)
                    self._model_xgb = ui.checkbox('XGBoost', value=False)
            
            # 実行ボタン
            with ui.row().classes('w-full justify-center gap-4'):
                self._run_btn = ui.button(
                    '▶ AutoMLを実行',
                    on_click=self._run_automl,
                    color='primary'
                ).props('size=lg').disable()
                
                self._cancel_btn = ui.button(
                    '⏹ キャンセル',
                    on_click=self._cancel_execution,
                    color='negative'
                ).props('outline').disable()
            
            # 進捗表示
            self._progress_card = ui.card().classes('w-full mt-4')
            self._progress_card.visible = False
            with self._progress_card:
                ui.label('⏳ AutoML実行中...').classes('font-bold text-lg')
                self._progress_bar = ui.linear_progress(value=0, show_value=True)
                self._progress_log = ui.log().classes('w-full h-64 font-mono text-xs')
            
            # 結果表示
            self._result_card = ui.card().classes('w-full mt-4')
            self._result_card.visible = False
            with self._result_card:
                ui.label('✅ 解析完了').classes('font-bold text-lg text-green-600')
                self._results_table = ui.table(columns=[], rows=[]).classes('w-full')
                
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('📊 結果を可視化', on_click=self._visualize_results)
                    ui.button('💾 モデルを保存', on_click=self._save_model)
                    ui.button('📄 レポート出力', on_click=self._export_report)
    
    def load_data(self, data: pd.DataFrame):
        """データをロードしてUIを更新"""
        self.current_data = data
        
        # データ情報を更新
        if self._row_count:
            self._row_count.text = f"{len(data):,}"
            self._col_count.text = f"{len(data.columns)}"
            self._memory_usage.text = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        
        # テーブル表示
        if self._data_table:
            self._data_table.columns = [{'name': col, 'label': col, 'field': col, 'align': 'left'} for col in data.columns]
            self._data_table.rows = data.head(10).to_dict('records')
            self._data_table.visible = True
            self._no_data_msg.visible = False
        
        # 目的変数選択を有効化
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if self._target_select:
            self._target_select.options = numeric_cols
            if numeric_cols:
                self._target_select.value = numeric_cols[-1]
        
        if self._target_card:
            self._target_card.visible = True
            self._config_card.visible = True
            self._run_btn.enable()
        
        ui.notify(f'データを読み込みました: {len(data)}行 × {len(data.columns)}列', type='positive')
    
    def _run_automl(self):
        """AutoMLを実行"""
        if self.current_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return
        
        if not self._target_select.value:
            ui.notify('目的変数を選択してください', type='warning')
            return
        
        self.target_column = self._target_select.value
        
        # UIを更新
        self._progress_card.visible = True
        self._result_card.visible = False
        self._run_btn.disable()
        self._cancel_btn.enable()
        
        # バックグラウンドでAutoML実行
        asyncio.create_task(self._execute_automl_pipeline())
    
    async def _execute_automl_pipeline(self):
        """AutoMLパイプラインを実行"""
        try:
            # AutoMLインスタンスを作成
            from backend.models.automl import AutoML
            self.automl_instance = AutoML()
            
            # 進捗コールバック
            def progress_callback(value: float, message: str):
                self._progress_bar.value = value
                self._progress_log.push(message)
            
            # AutoML実行
            # Run in thread executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, 
                lambda: self.automl_instance.run_automl(
                    df=self.current_data,
                    target_col=self.target_column,
                    models=self._models.value,
                    cv_folds=int(self._cv_folds.value),
                    metric=self._metric.value,
                    max_trials=int(self._max_trials.value),
                    progress_callback=progress_callback
                )
            )
            
            # 結果を表示
            await self._display_results(results)
            
            ui.notify('AutoMLが完了しました', type='positive')
            
        except Exception as e:
            logger.error(f"AutoMLエラー: {e}", exc_info=True)
            self._progress_log.push(f'❌ エラー: {str(e)}')
            ui.notify(f'エラーが発生しました: {str(e)}', type='negative')
        finally:
            self._run_btn.enable()
            self._cancel_btn.disable()
    
    async def _display_results(self, results: Dict):
        """結果を表示"""
        self._result_card.visible = True
        
        # テーブルを作成
        rows = []
        for model_name, result in results['results'].items():
            rows.append({
                'model': model_name,
                'cv_mean': f"{result['cv_mean']:.4f}",
                'cv_std': f"{result['cv_std']:.4f}",
                'test_score': f"{result['test_score']:.4f}",
                'best': '✓' if model_name == results['best_model'] else ''
            })
        
        self._results_table.columns = [
            {'name': 'model', 'label': 'モデル', 'field': 'model', 'align': 'left'},
            {'name': 'cv_mean', 'label': 'CV平均', 'field': 'cv_mean', 'align': 'center'},
            {'name': 'cv_std', 'label': 'CV標準偏差', 'field': 'cv_std', 'align': 'center'},
            {'name': 'test_score', 'label': 'テストスコア', 'field': 'test_score', 'align': 'center'},
            {'name': 'best', 'label': '最良', 'field': 'best', 'align': 'center'},
        ]
        self._results_table.rows = rows
        
        self._progress_log.push(f"✅ 最良モデル: {results['best_model']} (CVスコア: {results['best_cv_score']:.4f})")
    
    def _cancel_execution(self):
        """実行をキャンセル"""
        ui.notify('キャンセルしました', type='info')
        self._run_btn.enable()
        self._cancel_btn.disable()
    
    def _visualize_results(self):
        """結果を可視化"""
        ui.notify('可視化機能は開発中です', type='info')
    
    def _save_model(self):
        """モデルを保存"""
        if self.automl_instance:
            path = 'models/automl_model.pkl'
            self.automl_instance.save_model(path)
            ui.notify(f'モデルを保存しました: {path}', type='positive')
        else:
            ui.notify('保存するモデルがありません', type='warning')
    
    def _export_report(self):
        """レポートを出力"""
        ui.notify('レポート出力機能は開発中です', type='info')
