"""
frontend_nicegui/pages/automl_page.py
既存のAutoMLEngineと統合した完全版 - ui.table() 修正済み
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
    """AutoML解析ページ - 既存のAutoMLEngineを使用"""

    def __init__(self, viz_page=None):
        self.current_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.automl_instance = None
        self.last_results: Optional[Dict] = None
        self.viz_page = viz_page
        self._run_btn = None  # ボタンは render() で初期化
        self._cancel_btn = None

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
                # 修正: ui.table() には columns, rows 引数が必要
                self._data_table = ui.table(columns=[], rows=[]).classes('w-full h-64')
                self._data_table.visible = False

                self._no_data_msg = ui.label('⚠️ データが読み込まれていません。「Data Upload」タブからデータをアップロードしてください。')
                self._no_data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded')

            # 目的変数選択
            self._target_card = ui.card().classes('w-full mb-4')
            self._target_card.visible = False
            
            with self._target_card:
                ui.label('🎯 目的変数を選択').classes('font-bold text-lg mb-2')
                self._target_select = ui.select(options=[], label='目的変数（予測対象）', with_input=True).classes('w-full')
                ui.label('ヒント: 数値列なら回帰分析、カテゴリ列なら分類分析が自動選択されます').classes('text-xs text-gray-500 mt-1')

            # AutoML設定
            self._config_card = ui.card().classes('w-full mb-4')
            self._config_card.visible = False
            
            with self._config_card:
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
                self._models = ui.select(
                    options={'rf': 'Random Forest', 'lgbm': 'LightGBM', 'xgb': 'XGBoost'},
                    value=['rf', 'lgbm'],
                    multiple=True,
                    label='モデル選択'
                ).classes('w-full')

            # 実行ボタン
            with ui.row().classes('w-full justify-center gap-4'):
                self._run_btn = ui.button('▶ AutoMLを実行', on_click=self._run_automl, color='primary').props('size=lg').disable()
                self._cancel_btn = ui.button('⏹ キャンセル', on_click=self._cancel_execution, color='negative').props('outline').disable()

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
                with ui.row().classes('w-full items-center'):
                    ui.label('✅ 解析完了').classes('font-bold text-lg text-green-600')
                    ui.space()
                    self._best_model_label = ui.label('').classes('text-sm text-gray-600')
                # 修正: ui.table() には columns, rows 引数が必要
                self._results_table = ui.table(columns=[], rows=[]).classes('w-full')

                with ui.expansion('📊 特徴量重要度', icon='analytics').classes('w-full mt-4'):
                    self._importance_container = ui.column().classes('w-full')

                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('📊 結果を可視化', on_click=self._visualize_results)
                    ui.button('💾 モデルを保存', on_click=self._save_model)
                    ui.button('📄 レポート出力', on_click=self._export_report)

    def load_data(self, data: pd.DataFrame):
        """データをロードしてUIを更新"""
        self.current_data = data

        # 可視化ページにもデータを渡す
        if self.viz_page:
            self.viz_page.load_data(data)

        self._row_count.text = f"{len(data):,}"
        self._col_count.text = f"{len(data.columns)}"
        self._memory_usage.text = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"

        # 修正: tableのcolumns/rowsを設定
        columns = [{'name': col, 'label': col, 'field': col} for col in data.columns]
        rows = data.head(10).to_dict('records')
        self._data_table.columns = columns
        self._data_table.rows = rows
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
        if self._run_btn is not None:
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
        if self._run_btn is not None:
            self._run_btn.disable()
        if self._cancel_btn is not None:
            self._cancel_btn.enable()
        self._progress_log.clear()

        import asyncio
        asyncio.create_task(self._execute_automl_pipeline())

    async def _execute_automl_pipeline(self):
        """AutoMLパイプラインを実行 - 既存のAutoMLEngineを使用"""
        try:
            # 既存のAutoMLEngineをインポート
            from backend.models.automl import AutoMLEngine
            
            def progress_callback(step: int, total: int, message: str):
                # 進捗を0-1の範囲に変換
                value = min(1.0, step / total)
                self._progress_bar.value = value
                self._progress_log.push(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

            progress_callback(1, 6, "AutoMLを開始します...")
            await asyncio.sleep(0.1)

            # AutoMLEngineを初期化
            engine = AutoMLEngine(
                task="auto",
                cv_folds=int(self._cv_folds.value),
                model_keys=self._models.value,
                progress_callback=progress_callback
            )

            # 非同期スレッドで実行
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    engine.run,
                    df=self.current_data,
                    target_col=self.target_column
                )
                result = await asyncio.get_event_loop().run_in_executor(None, future.result)

            # 結果をUIに表示
            self.last_results = {
                'best_model': result.best_model_key,
                'best_cv_score': result.best_score,
                'results': {result.best_model_key: {
                    'cv_mean': result.best_score,
                    'cv_std': 0.0,
                    'test_score': result.best_score,
                    'training_time': result.elapsed_seconds
                }},
                'total_time': result.elapsed_seconds
            }

            await self._display_results(self.last_results)
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
                'cv_std': f"{result.get('cv_std', 0):.4f}",
                'test_score': f"{result['test_score']:.4f}",
                'time': f"{result.get('training_time', 0):.2f}s",
                'best': '✓' if model_name == results['best_model'] else ''
            })

        columns = [
            {'name': 'model', 'label': 'モデル', 'field': 'model'},
            {'name': 'cv_mean', 'label': 'CV平均', 'field': 'cv_mean'},
            {'name': 'cv_std', 'label': 'CV標準偏差', 'field': 'cv_std'},
            {'name': 'test_score', 'label': 'テストスコア', 'field': 'test_score'},
            {'name': 'time', 'label': '訓練時間', 'field': 'time'},
            {'name': 'best', 'label': '最良', 'field': 'best'},
        ]
        self._results_table.columns = columns
        self._results_table.rows = rows
        self._progress_log.push(f"✅ 完了 - 総時間: {results['total_time']:.2f}秒")

    def _cancel_execution(self):
        ui.notify('キャンセルしました', type='info')
        self._run_btn.enable()
        self._cancel_btn.disable()

    def _visualize_results(self):
        """AutoML結果の可視化 - SHAP + 特徴量重要度"""
        if self.last_results is None or self.current_data is None:
            ui.notify('結果がありません', type='warning')
            return

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from backend.interpret.shap_explainer import ShapExplainer, ShapConfig

            self._progress_card.visible = True
            self._progress_bar.value = 20

            # 最良モデルを取得
            best_model = self.last_results.get('best_model')
            X_test = self.last_results.get('X_test')

            if best_model is None or X_test is None:
                ui.notify('モデル情報が不足しています', type='warning')
                return

            numeric_cols = self.current_data.select_dtypes(include=['number']).columns.tolist()
            X_test_numeric = X_test[numeric_cols] if numeric_cols else X_test

            self._progress_bar.value = 40

            # SHAP値を計算
            try:
                config = ShapConfig(max_display=10)
                explainer = ShapExplainer(config)
                shap_result = explainer.explain(best_model, X_test_numeric.iloc[:min(100, len(X_test_numeric))])

                self._progress_bar.value = 60

                # 特徴量重要度を計算
                feature_importance = shap_result.feature_importance().head(10)

                self._progress_bar.value = 70

                # Plotlyで可視化
                fig = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='📊 SHAP値による特徴量重要度',
                    labels={'importance': '平均SHAP値の絶対値', 'feature': '特徴量'},
                    height=500,
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                self._progress_bar.value = 80

                # 結果を表示
                self._importance_container.clear()
                with self._importance_container:
                    ui.html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))

                ui.notify('SHAP解釈グラフを生成しました', type='positive')

            except ImportError:
                ui.notify('⚠️ SHAPライブラリが未インストールです。特徴量重要度のみ表示します', type='warning')
                # フォールバック: モデルのfeature_importances を表示
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'feature': numeric_cols,
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(10)

                    fig = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='📊 モデル特徴量重要度',
                        height=500
                    )
                    self._importance_container.clear()
                    with self._importance_container:
                        ui.html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))

            self._progress_bar.value = 100

        except Exception as e:
            logger.error(f"可視化エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _save_model(self):
        """AutoMLの最良モデルを保存"""
        if self.last_results is None:
            ui.notify('保存するモデルがありません', type='warning')
            return

        try:
            import joblib
            from pathlib import Path

            self._progress_card.visible = True
            self._progress_bar.value = 30

            # モデル保存ディレクトリを作成
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = model_dir / f'automl_model_{timestamp}.pkl'

            # 最良モデルを保存
            best_model = self.last_results.get('best_model')
            if best_model is None:
                ui.notify('モデルが見つかりません', type='warning')
                return

            joblib.dump(best_model, model_path)
            self._progress_bar.value = 70

            # メタデータを保存
            metadata = {
                'timestamp': timestamp,
                'best_score': self.last_results.get('best_score', 0),
                'model_type': type(best_model).__name__,
                'data_shape': str(self.current_data.shape) if self.current_data is not None else None,
                'target_column': self.target_column,
            }

            metadata_path = model_dir / f'automl_metadata_{timestamp}.json'
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self._progress_bar.value = 100

            ui.notify(f'✓ モデルを保存しました: {model_path}', type='positive')

            # ダウンロードリンクを表示
            with ui.dialog() as dialog:
                with ui.card():
                    ui.label('📦 モデル保存完了').classes('text-lg font-bold')
                    ui.label(f'モデルファイル: {model_path}').classes('text-sm font-mono')
                    ui.label(f'メタデータ: {metadata_path}').classes('text-sm font-mono')
                    ui.label(f'最良スコア: {metadata["best_score"]:.4f}').classes('text-sm')
                    with ui.row():
                        ui.button('閉じる', on_click=dialog.close)

            dialog.open()

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _export_report(self):
        """AutoML結果をPDFレポートで出力"""
        if self.last_results is None or self.current_data is None:
            ui.notify('出力する結果がありません', type='warning')
            return

        try:
            from pathlib import Path
            from backend.export.pdf_exporter import PDFExporter

            self._progress_card.visible = True
            self._progress_bar.value = 20

            # レポート出力ディレクトリを作成
            report_dir = Path('reports')
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'automl_report_{timestamp}.pdf'

            self._progress_bar.value = 40

            # PDF生成データを準備
            report_data = {
                'title': 'ChemAI AutoML 解析レポート',
                'timestamp': timestamp,
                'data_shape': self.current_data.shape,
                'target_column': self.target_column,
                'best_score': self.last_results.get('best_score', 0),
                'best_model': type(self.last_results.get('best_model')).__name__,
                'cv_folds': self.last_results.get('cv_folds', 5),
                'results_summary': [
                    {
                        'model': result.get('model'),
                        'score': result.get('score', 0),
                        'time': result.get('time', 0)
                    }
                    for result in self.last_results.get('all_results', [])[:5]  # Top 5
                ]
            }

            self._progress_bar.value = 60

            # PDFExporter を使用
            exporter = PDFExporter(output_dir=str(report_dir))
            pdf_path = exporter.export(report_data, report_filename)

            self._progress_bar.value = 90

            ui.notify(f'✓ レポートを出力しました: {pdf_path}', type='positive')

            # 完了ダイアログ
            with ui.dialog() as dialog:
                with ui.card():
                    ui.label('📄 レポート出力完了').classes('text-lg font-bold')
                    ui.label(f'ファイル: {pdf_path}').classes('text-sm font-mono')
                    with ui.row().classes('mt-4'):
                        ui.button('閉じる', on_click=dialog.close)

            dialog.open()

            self._progress_bar.value = 100

        except ImportError:
            ui.notify('⚠️ PDFエクスポート機能が利用できません（reportlab未インストール）', type='warning')

        except Exception as e:
            logger.error(f"レポート出力エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

        finally:
            self._progress_card.visible = False
