"""
frontend_nicegui/pages/automl_page.py
描画タイミング修正版
"""
from nicegui import ui, events
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import asyncio
import logging
import concurrent.futures
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoMLPage:
    """AutoML解析ページ"""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.automl_instance = None
        self.last_results: Optional[Dict] = None
        
    def render(self):
        """AutoMLページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
            
            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('🤖 AutoML - 自動機械学習').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('既存のAutoML機能を利用した解析').classes('text-gray-600')
            
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
                
                ui.label('プレビュー（先頭10行）').classes('mt-4 font-bold')
                self._data_table = ui.table().classes('w-full h-64')
                self._data_table.visible = False
                
                self._no_data_msg = ui.label('⚠️ データが読み込まれていません。「Data Upload」タブからデータをアップロードしてください。')
                self._no_data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded')
            
            # 目的変数選択
            with ui.card().classes('w-full mb-4').visible(False) as self._target_card:
                ui.label('🎯 目的変数を選択').classes('font-bold text-lg mb-2')
                self._target_select = ui.select(options=[], label='目的変数（予測対象）', with_input=True).classes('w-full')
                ui.label('ヒント: 数値列なら回帰分析、カテゴリ列なら分類分析が自動選択されます').classes('text-xs text-gray-500 mt-1')
            
            # AutoML設定
            with ui.card().classes('w-full mb-4').visible(False) as self._config_card:
                ui.label('⚙️ AutoML設定').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('CV Fold数').classes('text-xs text-gray-500')
                        self._cv_folds = ui.number(value=5, min=2, max=10, step=1)
                    with ui.column():
                        ui.label('評価指標').classes('text-xs text-gray-500')
                        self._metric = ui.select(options=['rmse', 'mae', 'r2', 'accuracy', 'f1'], value='rmse')
                    with ui.column():
                        ui.label('最大試行').classes('text-xs text-gray-500')
                        self._max_trials = ui.number(value=20, min=5, max=100, step=5)
                
                ui.label('使用するモデル').classes('mt-4 font-bold')
                self._models = ui.checkbox_group(
                    options=[('random_forest', 'Random Forest'), ('lightgbm', 'LightGBM'), ('xgboost', 'XGBoost')],
                    value=['random_forest', 'lightgbm']
                ).classes('w-full')
            
            # 実行ボタン
            with ui.row().classes('w-full justify-center gap-4'):
                self._run_btn = ui.button('▶ AutoMLを実行', on_click=self._run_automl, color='primary').props('size=lg').disable()
                self._cancel_btn = ui.button('⏹ キャンセル', on_click=self._cancel_execution, color='negative').props('outline').disable()
            
            # 進捗表示
            self._progress_card = ui.card().classes('w-full mt-4').visible(False)
            with self._progress_card:
                ui.label('⏳ AutoML実行中...').classes('font-bold text-lg')
                self._progress_bar = ui.linear_progress(value=0, show_value=True)
                self._progress_log = ui.log().classes('w-full h-64 font-mono text-xs')
            
            # 結果表示
            self._result_card = ui.card().classes('w-full mt-4').visible(False)
            with self._result_card:
                with ui.row().classes('w-full items-center'):
                    ui.label('✅ 解析完了').classes('font-bold text-lg text-green-600')
                    ui.space()
                    self._best_model_label = ui.label('').classes('text-sm text-gray-600')
                self._results_table = ui.table().classes('w-full')
                
                with ui.expansion('📊 特徴量重要度', icon='analytics').classes('w-full mt-4'):
                    self._importance_container = ui.column().classes('w-full')
                
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('📊 結果を可視化', on_click=self._visualize_results)
                    ui.button('💾 モデルを保存', on_click=self._save_model)
                    ui.button('📄 レポート出力', on_click=self._export_report)

    def load_data(self, data: pd.DataFrame):
        """データをロードしてUIを更新"""
        self.current_data = data
        
        self._row_count.text = f"{len(data):,}"
        self._col_count.text = f"{len(data.columns)}"
        self._memory_usage.text = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        
        self._data_table.options = {
            'columns': [{'field': col, 'headerName': col} for col in data.columns],
            'rowData': data.head(10).to_dict('records')
        }
        self._data_table.visible = True
        self._no_data_msg.visible = False
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numeric_cols + categorical_cols
        
        self._target_select.options = all_cols
        if numeric_cols:
            self._target_select.value = numeric_cols[-1]
        elif all_cols:
            self._target_select.value = all_cols[-1]
        
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
        self._progress_card.visible = True
        self._result_card.visible = False
        self._run_btn.disable()
        self._cancel_btn.enable()
        self._progress_log.clear()
        
        import asyncio
        asyncio.create_task(self._execute_automl_pipeline())
    
    async def _execute_automl_pipeline(self):
        """AutoMLパイプラインを実行"""
        try:
            from backend.models.automl import AutoML
            self.automl_instance = AutoML()
            
            def progress_callback(value: float, message: str):
                self._progress_bar.value = value
                self._progress_log.push(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            
            progress_callback(0.0, "AutoMLを開始します...")
            await asyncio.sleep(0.1)
            
            # スレッドで実行
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.automl_instance.run_automl,
                    df=self.current_data,
                    target_col=self.target_column,
                    models=self._models.value,
                    cv_folds=int(self._cv_folds.value),
                    metric=self._metric.value,
                    max_trials=int(self._max_trials.value),
                    progress_callback=progress_callback
                )
                results = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            self.last_results = results
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
        self._best_model_label.text = f"最良モデル: {results['best_model']} (CVスコア: {results['best_cv_score']:.4f})"
        
        rows = []
        for model_name, result in results['results'].items():
            rows.append({
                'model': model_name,
                'cv_mean': f"{result['cv_mean']:.4f}",
                'cv_std': f"{result['cv_std']:.4f}",
                'test_score': f"{result['test_score']:.4f}",
                'time': f"{result['training_time']:.2f}s",
                'best': '✓' if model_name == results['best_model'] else ''
            })
        
        self._results_table.options = {
            'columns': [
                {'field': 'model', 'headerName': 'モデル'},
                {'field': 'cv_mean', 'headerName': 'CV平均'},
                {'field': 'cv_std', 'headerName': 'CV標準偏差'},
                {'field': 'test_score', 'headerName': 'テストスコア'},
                {'field': 'time', 'headerName': '訓練時間'},
                {'field': 'best', 'headerName': '最良'},
            ],
            'rowData': rows
        }
        self._progress_log.push(f"✅ 完了 - 総時間: {results['total_time']:.2f}秒")
    
    def _cancel_execution(self):
        ui.notify('キャンセルしました', type='info')
        self._run_btn.enable()
        self._cancel_btn.disable()
    
    def _visualize_results(self):
        if self.last_results is None:
            ui.notify('結果がありません', type='warning')
            return
        best_model_name = self.last_results['best_model']
        importance = self.automl_instance.models[best_model_name].feature_importance
        if importance:
            self._importance_container.clear()
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            max_val = max(imp for _, imp in sorted_importance)
            for feat, imp in sorted_importance:
                with self._importance_container:
                    with ui.row().classes('w-full items-center'):
                        ui.label(feat).classes('w-1/3 text-xs').style('overflow: hidden; text-overflow: ellipsis;')
                        ui.linear_progress(value=imp / max_val if max_val > 0 else 0, show_value=False).classes('w-2/3')
    
    def _save_model(self):
        if self.automl_instance is None:
            ui.notify('保存するモデルがありません', type='warning')
            return
        path = 'models/automl_model.pkl'
        self.automl_instance.save_model(path)
        ui.notify(f'モデルを保存しました: {path}', type='positive')
    
    def _export_report(self):
        if self.last_results is None or self.automl_instance is None:
            ui.notify('出力する結果がありません', type='warning')
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'reports/automl_report_{timestamp}.json'
        self.automl_instance.export_report(path, self.last_results)
        ui.notify(f'レポートを出力しました: {path}', type='positive')
