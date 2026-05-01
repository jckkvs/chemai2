"""
frontend_nicegui/pages/automl_page.py
AutoML page - integrated with existing AutoMLEngine
"""
from nicegui import ui
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import asyncio
import logging
import concurrent.futures
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoMLPage:
    """AutoML analysis page - uses existing AutoMLEngine"""

    def __init__(self, viz_page=None):
        self.current_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.automl_instance = None
        self.last_results: Optional[Dict] = None
        self.viz_page = viz_page
        self._run_btn = None
        self._cancel_btn = None
        self._pending_data = None

    def render(self):
        """Render the AutoML page"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):

            # Header
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('AutoML - Automatic Machine Learning').classes('text-2xl font-bold')
                    ui.space()
                    ui.label('Using existing AutoML functionality').classes('text-gray-600')

            # Data confirmation section
            with ui.card().classes('w-full mb-4'):
                ui.label('Loaded Data').classes('font-bold text-lg mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('Rows').classes('text-xs text-gray-500')
                        self._row_count = ui.label('-')
                    with ui.column():
                        ui.label('Columns').classes('text-xs text-gray-500')
                        self._col_count = ui.label('-')
                    with ui.column():
                        ui.label('Memory Usage').classes('text-xs text-gray-500')
                        self._memory_usage = ui.label('-')

                ui.label('Preview (first 10 rows)').classes('mt-4 font-bold')
                self._data_table = ui.table(columns=[], rows=[]).classes('w-full h-64')
                self._data_table.visible = False

                self._no_data_msg = ui.label('No data loaded. Please upload data from the Data Upload tab.')
                self._no_data_msg.classes('text-orange-600 p-4 bg-orange-50 rounded')

            # Target variable selection
            self._target_card = ui.card().classes('w-full mb-4')
            self._target_card.visible = False

            with self._target_card:
                ui.label('Select Target Variable').classes('font-bold text-lg mb-2')
                self._target_select = ui.select(options=[], label='Target variable (prediction target)', with_input=True).classes('w-full')
                ui.label('Hint: Numeric columns = regression, Categorical columns = classification').classes('text-xs text-gray-500 mt-1')

            # AutoML configuration
            self._config_card = ui.card().classes('w-full mb-4')
            self._config_card.visible = False

            with self._config_card:
                ui.label('AutoML Settings').classes('font-bold text-lg mb-2')
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        ui.label('CV Folds').classes('text-xs text-gray-500')
                        self._cv_folds = ui.number(value=5, min=2, max=10, step=1)
                    with ui.column():
                        ui.label('Metric').classes('text-xs text-gray-500')
                        self._metric = ui.select(options=['rmse', 'mae', 'r2', 'accuracy', 'f1'], value='rmse')
                    with ui.column():
                        ui.label('Max Trials').classes('text-xs text-gray-500')
                        self._max_trials = ui.number(value=20, min=5, max=100, step=5)

                ui.label('Models to use').classes('mt-4 font-bold')
                self._models = ui.select(
                    options={'rf': 'Random Forest', 'lgbm': 'LightGBM', 'xgb': 'XGBoost'},
                    value=['rf', 'lgbm'],
                    multiple=True,
                    label='Model Selection'
                ).classes('w-full')

            # Run button
            with ui.row().classes('w-full justify-center gap-4'):
                self._run_btn = ui.button('Run AutoML', on_click=self._run_automl, color='primary').props('size=lg')
                self._cancel_btn = ui.button('Cancel', on_click=self._cancel_execution, color='negative').props('outline')
                self._run_btn.disable()
                self._cancel_btn.disable()

            # Progress display
            self._progress_card = ui.card().classes('w-full mt-4')
            self._progress_card.visible = False

            with self._progress_card:
                ui.label('Running AutoML...').classes('font-bold text-lg')
                self._progress_bar = ui.linear_progress(value=0, show_value=True)
                self._progress_log = ui.log().classes('w-full h-64 font-mono text-xs')

            # Results display
            self._result_card = ui.card().classes('w-full mt-4')
            self._result_card.visible = False

            with self._result_card:
                with ui.row().classes('w-full items-center'):
                    ui.label('Analysis Complete').classes('font-bold text-lg text-green-600')
                    ui.space()
                    self._best_model_label = ui.label('').classes('text-sm text-gray-600')
                self._results_table = ui.table(columns=[], rows=[]).classes('w-full')

                with ui.expansion('Feature Importance', icon='analytics').classes('w-full mt-4'):
                    self._importance_container = ui.column().classes('w-full')

                with ui.expansion('Recommendation Reasons', icon='lightbulb').classes('w-full mt-4'):
                    self._cv_reason_md = ui.markdown('').classes('text-sm')
                    self._model_reason_md = ui.markdown('').classes('text-sm mt-2')

                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('Visualize Results', on_click=self._visualize_results)
                    ui.button('Save Model', on_click=self._save_model)
                    ui.button('Export Report', on_click=self._export_report)

        # Process pending data if any
        if self._pending_data is not None:
            pending = self._pending_data
            self._pending_data = None
            self.load_data(pending)

    def load_data(self, data: pd.DataFrame):
        """Load data and update UI"""
        self.current_data = data

        if self.viz_page:
            self.viz_page.load_data(data)

        if self._run_btn is None:
            ui.notify('UI initializing...', type='info')
            self._pending_data = data
            return

        self._row_count.text = f"{len(data):,}"
        self._col_count.text = f"{len(data.columns)}"
        self._memory_usage.text = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"

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
        ui.notify(f'Data loaded: {len(data)} rows x {len(data.columns)} columns', type='positive')

    def _run_automl(self):
        """Run AutoML"""
        if self.current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        if not self._target_select.value:
            ui.notify('Please select a target variable', type='warning')
            return

        self.target_column = self._target_select.value
        self._progress_card.visible = True
        self._result_card.visible = False
        if self._run_btn is not None:
            self._run_btn.disable()
        if self._cancel_btn is not None:
            self._cancel_btn.enable()
        self._progress_log.clear()

        # Reset completion flags
        self._pipeline_done = False
        self._pipeline_error = None
        self._pipeline_results = None

        # Start background task
        asyncio.create_task(self._execute_automl_pipeline())

        # Use timer to check for completion (runs in main thread)
        self._timer = ui.timer(interval=0.5, callback=self._check_pipeline_done)

    async def _execute_automl_pipeline(self):
        """Execute AutoML pipeline using existing AutoMLEngine - computation only, no UI updates"""
        try:
            from backend.models.automl import AutoMLEngine

            def progress_callback(step: int, total: int, message: str):
                # Only update progress bar and log - these might work from background
                value = min(1.0, step / total)
                self._progress_bar.value = value
                self._progress_log.push(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

            progress_callback(1, 6, "Starting AutoML...")
            await asyncio.sleep(0.1)

            engine = AutoMLEngine(
                task="auto",
                cv_folds=int(self._cv_folds.value),
                model_keys=self._models.value,
                progress_callback=progress_callback
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(engine.run, self.current_data, self.target_column)
                result = await asyncio.get_event_loop().run_in_executor(None, future.result)

            self.last_results = {
                'best_model': result.best_model_key,
                'best_cv_score': result.best_score,
                'best_pipeline': result.best_pipeline,
                'X_train': result.X_train,
                'y_train': result.y_train,
                'results': {
                    result.best_model_key: {
                        'cv_mean': result.best_score,
                        'cv_std': 0.0,
                        'test_score': result.best_score,
                        'training_time': result.elapsed_seconds
                    }
                },
                'all_results': [
                    {'model': k, 'score': v, 'time': result.model_details.get(k, {}).get('fit_time', 0)}
                    for k, v in result.model_scores.items()
                ],
                'cv_folds': self._cv_folds.value,
                'total_time': result.elapsed_seconds
            }

            # Store results and set completion flag - UI updates happen in main thread
            self.last_results = {
                'best_model': result.best_model_key,
                'best_cv_score': result.best_score,
                'best_pipeline': result.best_pipeline,
                'X_train': result.X_train,
                'y_train': result.y_train,
                'results': {
                    result.best_model_key: {
                        'cv_mean': result.best_score,
                        'cv_std': 0.0,
                        'test_score': result.best_score,
                        'training_time': result.elapsed_seconds
                    }
                },
                'all_results': [
                    {'model': k, 'score': v, 'time': result.model_details.get(k, {}).get('fit_time', 0)}
                    for k, v in result.model_scores.items()
                ],
                'cv_folds': self._cv_folds.value,
                'total_time': result.elapsed_seconds
            }

            self._pipeline_done = True
            self._pipeline_error = None
            self._progress_log.push("✅ AutoML completed successfully")

        except Exception as e:
            logger.error(f"AutoML error: {e}", exc_info=True)
            self._pipeline_error = e
            self._pipeline_done = True
            self._progress_log.push(f'❌ Error: {str(e)}')
        finally:
            if self._run_btn is not None:
                self._run_btn.enable()
            if self._cancel_btn is not None:
                self._cancel_btn.disable()

    async def _display_results(self, results: Dict):
        """Display results"""
        self._result_card.visible = True
        self._best_model_label.text = f"Best model: {results['best_model']} (CV Score: {results['best_cv_score']:.4f})"

        rows = []
        for model_name, result in results['results'].items():
            rows.append({
                'model': model_name,
                'cv_mean': f"{result['cv_mean']:.4f}",
                'cv_std': f"{result.get('cv_std', 0):.4f}",
                'test_score': f"{result['test_score']:.4f}",
                'time': f"{result.get('training_time', 0):.2f}s",
                'best': 'Y' if model_name == results['best_model'] else ''
            })

        columns = [
            {'name': 'model', 'label': 'Model', 'field': 'model'},
            {'name': 'cv_mean', 'label': 'CV Mean', 'field': 'cv_mean'},
            {'name': 'cv_std', 'label': 'CV Std', 'field': 'cv_std'},
            {'name': 'test_score', 'label': 'Test Score', 'field': 'test_score'},
            {'name': 'time', 'label': 'Training Time', 'field': 'time'},
            {'name': 'best', 'label': 'Best', 'field': 'best'},
        ]
        self._results_table.columns = columns
        self._results_table.rows = rows

        # 推奨理由の取得と表示
        try:
            from backend.utils.cv_recommender import recommend_cv_strategy
            from backend.llm.analysis_advisor import AnalysisAdvisor

            X = results.get('X_train')
            y = results.get('y_train')
            if X is not None and y is not None:
                import pandas as pd
                import numpy as np
                if not isinstance(X, pd.DataFrame):
                    numeric_cols = self.current_data.select_dtypes(include=['number']).columns.tolist()
                    if self.target_column in numeric_cols:
                        numeric_cols.remove(self.target_column)
                    if X.shape[1] == len(numeric_cols):
                        X_df = pd.DataFrame(X, columns=numeric_cols)
                    else:
                        X_df = pd.DataFrame(X)
                else:
                    X_df = X

                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                # CV推奨理由
                cv_rec = recommend_cv_strategy(X_df, y, metadata={'task_type': 'regression'})
                if hasattr(self, '_cv_reason_md'):
                    self._cv_reason_md.content = f"**CV推奨**: {cv_rec.reason}"

                # モデル推奨理由（暫定）
                if hasattr(self, '_model_reason_md'):
                    best_model = results.get('best_model', 'Unknown')
                    self._model_reason_md.content = f"**最佳模型**: {best_model} が選択されました。"
            else:
                if hasattr(self, '_cv_reason_md'):
                    self._cv_reason_md.content = "推奨理由を表示するためのデータが不足しています。"
        except Exception as e:
            logger.error(f"推奨理由の取得エラー: {e}", exc_info=True)
            if hasattr(self, '_cv_reason_md'):
                self._cv_reason_md.content = f"推奨理由の取得中にエラー: {str(e)}"

        self._progress_log.push(f"Complete - Total time: {self.last_results['total_time']:.2f} seconds")

    def _check_pipeline_done(self):
        """Check if pipeline is done - runs in main thread via timer"""
        if not self._pipeline_done:
            return  # Not done yet

        # Cancel the timer
        if hasattr(self, '_timer') and self._timer:
            self._timer.cancel()
            self._timer = None

        # Update UI based on results
        if self._pipeline_error:
            # Pipeline failed
            logger.error(f"AutoML pipeline failed: {self._pipeline_error}")
            self._progress_log.push(f'❌ Pipeline error: {str(self._pipeline_error)}')
        else:
            # Pipeline succeeded - update UI
            if self.last_results:
                asyncio.create_task(self._display_results(self.last_results))

        # Re-enable buttons
        if self._run_btn is not None:
            self._run_btn.enable()
        if self._cancel_btn is not None:
            self._cancel_btn.disable()

    def _cancel_execution(self):
        ui.notify('Cancelled', type='info')
        if self._run_btn is not None:
            self._run_btn.enable()
        if self._cancel_btn is not None:
            self._cancel_btn.disable()

    def _visualize_results(self):
        """Visualize AutoML results - SHAP + feature importance"""
        if self.last_results is None or self.current_data is None:
            ui.notify('No results available', type='warning')
            return

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from backend.interpret.shap_explainer import ShapExplainer, ShapConfig

            self._progress_card.visible = True
            self._progress_bar.value = 20

            best_pipeline = self.last_results.get('best_pipeline')
            X_train = self.last_results.get('X_train')

            if best_pipeline is None or X_train is None:
                ui.notify('Model information is insufficient', type='warning')
                return

            # X_train が NumPy 配列の場合は DataFrame に変換
            if not isinstance(X_train, pd.DataFrame):
                processed_X = self.last_results.get('processed_X')
                if isinstance(processed_X, pd.DataFrame):
                    X_train = pd.DataFrame(X_train, columns=processed_X.columns)
                else:
                    n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
                    X_train = pd.DataFrame(
                        X_train,
                        columns=[f'feature_{i}' for i in range(n_features)]
                    )

            numeric_cols = self.current_data.select_dtypes(include=['number']).columns.tolist()
            # Remove target column from numeric_cols (X_train doesn't have it)
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
            X_train_numeric = X_train[numeric_cols] if numeric_cols else X_train

            self._progress_bar.value = 40

            try:
                config = ShapConfig(max_display=10)
                explainer = ShapExplainer(config)
                shap_result = explainer.explain(best_pipeline, X_train_numeric.iloc[:min(100, len(X_train_numeric))])

                self._progress_bar.value = 60

                feature_importance = shap_result.feature_importance().head(10)

                self._progress_bar.value = 70

                fig = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance by SHAP Values',
                    labels={'importance': 'Mean |SHAP value|', 'feature': 'Feature'},
                    height=500,
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                self._progress_bar.value = 80

                self._importance_container.clear()
                with self._importance_container:
                    ui.plotly(fig)

                ui.notify('SHAP interpretation graph generated', type='positive')

            except ImportError:
                ui.notify('SHAP library not installed. Showing model feature importance only.', type='warning')
                estimator = best_pipeline.steps[-1][1] if hasattr(best_pipeline, 'steps') else best_pipeline
                if hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                    feature_importance = pd.DataFrame({
                        'feature': numeric_cols,
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(10)

                    fig = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Model Feature Importance',
                        height=500
                    )
                    self._importance_container.clear()
                    with self._importance_container:
                        ui.plotly(fig)

            self._progress_bar.value = 100

        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            ui.notify(f'Error: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _save_model(self):
        """Save the best AutoML model"""
        if self.last_results is None:
            ui.notify('No model to save', type='warning')
            return

        try:
            import joblib
            from pathlib import Path

            self._progress_card.visible = True
            self._progress_bar.value = 30

            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = model_dir / f'automl_model_{timestamp}.pkl'

            best_pipeline = self.last_results.get('best_pipeline')
            if best_pipeline is None:
                ui.notify('Model not found', type='warning')
                return

            joblib.dump(best_pipeline, model_path)
            self._progress_bar.value = 70

            metadata = {
                'timestamp': timestamp,
                'best_score': self.last_results.get('best_cv_score', 0),
                'model_type': type(best_pipeline).__name__,
                'data_shape': str(self.current_data.shape) if self.current_data is not None else None,
                'target_column': self.target_column,
            }

            metadata_path = model_dir / f'automl_metadata_{timestamp}.json'
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self._progress_bar.value = 100

            ui.notify(f'Model saved: {model_path}', type='positive')

            with ui.dialog() as dialog:
                with ui.card():
                    ui.label('Model Save Complete').classes('text-lg font-bold')
                    ui.label(f'Model file: {model_path}').classes('text-sm font-mono')
                    ui.label(f'Metadata: {metadata_path}').classes('text-sm font-mono')
                    ui.label(f'Best score: {metadata["best_score"]:.4f}').classes('text-sm')
                    with ui.row():
                        ui.button('Close', on_click=dialog.close)

            dialog.open()

        except Exception as e:
            logger.error(f"Model save error: {e}", exc_info=True)
            ui.notify(f'Error: {str(e)}', type='negative')
        finally:
            self._progress_card.visible = False

    def _export_report(self):
        """Export AutoML results as PDF report (with LLM commentary)"""
        if self.last_results is None or self.current_data is None:
            ui.notify('No results to export', type='warning')
            return

        try:
            from pathlib import Path
            from backend.export.pdf_exporter import PDFExporter

            self._progress_card.visible = True
            self._progress_bar.value = 10

            report_dir = Path('reports')
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'automl_report_{timestamp}'

            self._progress_bar.value = 20

            # ── 特徴量重要度の抽出（共通） ──
            feature_importances = self._extract_feature_importances()

            # ── LLM考察の生成 ──
            ai_commentary = ""
            try:
                from backend.llm.report_generator import LLMReportGenerator, AnalysisResults
                from backend.llm import get_llm_provider

                provider = get_llm_provider()
                if provider:
                    generator = LLMReportGenerator(provider=provider)

                    # AnalysisResultsを構築
                    results = AnalysisResults(
                        n_samples=self.current_data.shape[0],
                        n_features=self.current_data.shape[1],
                        target_col=self.target_column or "Target",
                        task_type="regression",
                        best_model=self.last_results.get('best_model', 'Unknown'),
                        best_score=float(self.last_results.get('best_cv_score', 0)),
                        model_comparison=[
                            {'name': r.get('model', ''), 'score': float(r.get('score', 0))}
                            for r in self.last_results.get('all_results', [])
                        ],
                        feature_importance=[
                            {'name': k, 'importance': v}
                            for k, v in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
                        ],
                    )

                    self._progress_bar.value = 40
                    ui.notify('LLM考察を生成中...', type='info')

                    report = generator.generate_report(results)
                    ai_commentary = report.full_report or ""

                    self._progress_bar.value = 60
                    ui.notify('LLM考察完了', type='positive')
            except ImportError:
                logger.warning("LLM report generator not available")
            except Exception as e:
                logger.warning(f"LLM commentary generation failed: {e}")

            # ── 評価指標の構築 ──
            metrics = {'CV Score': self.last_results.get('best_cv_score', 0)}
            try:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                oof_pred = getattr(self, '_oof_predictions', None)
                oof_true = getattr(self, '_oof_true', None)
                if oof_pred is not None and oof_true is not None:
                    metrics['R² (OOF)'] = float(r2_score(oof_true, oof_pred))
                    metrics['RMSE (OOF)'] = float(mean_squared_error(oof_true, oof_pred, squared=False))
                    metrics['MAE (OOF)'] = float(mean_absolute_error(oof_true, oof_pred))
            except Exception:
                pass

            # ── PDFExporter用データ構造（キー名を合わせる） ──
            report_data = {
                'best_model_name': self.last_results.get('best_model', 'Unknown'),
                'metrics': metrics,
                'feature_importances': feature_importances,
                'ai_commentary': ai_commentary,
                'chart_paths': [],
            }

            self._progress_bar.value = 70

            exporter = PDFExporter(output_dir=str(report_dir))
            pdf_path = exporter.export(report_data, report_filename)

            self._progress_bar.value = 90

            ui.notify(f'Report exported: {pdf_path}', type='positive')

            with ui.dialog() as dialog:
                with ui.card():
                    ui.label('Report Export Complete').classes('text-lg font-bold')
                    ui.label(f'File: {pdf_path}').classes('text-sm font-mono')
                    if ai_commentary:
                        ui.label('LLM考察を含むレポートです').classes('text-caption text-teal')
                    with ui.row().classes('mt-4'):
                        ui.button('Close', on_click=dialog.close)

            dialog.open()

            self._progress_bar.value = 100

        except ImportError:
            ui.notify('PDF export not available (reportlab not installed)', type='warning')

        except Exception as e:
            logger.error(f"Report export error: {e}", exc_info=True)
            ui.notify(f'Error: {str(e)}', type='negative')

        finally:
            self._progress_card.visible = False

    def _extract_feature_importances(self) -> dict:
        """best_pipelineから特徴量重要度を抽出する共通メソッド"""
        feature_importances = {}
        try:
            pipeline = self.last_results.get('best_pipeline')
            if pipeline is None:
                return feature_importances

            estimator = pipeline
            if hasattr(estimator, 'steps'):
                estimator = estimator.steps[-1][1]
                if hasattr(estimator, 'steps'):
                    estimator = estimator.steps[-1][1]

            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                try:
                    feat_names = pipeline[:-1].get_feature_names_out().tolist()
                except Exception:
                    X = self.last_results.get('X_train')
                    if X is not None and hasattr(X, 'columns'):
                        feat_names = list(X.columns)
                    else:
                        feat_names = [f"f{i}" for i in range(len(importances))]

                for i, imp in enumerate(importances):
                    name = feat_names[i] if i < len(feat_names) else f"f{i}"
                    feature_importances[name] = float(imp)
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {e}")
        return feature_importances
