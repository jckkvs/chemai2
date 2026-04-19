"""
統合分析ページ（データ→特徴量→制約→学習→結果）
"""
from nicegui import ui
from typing import Optional

from ..state.analysis_state import state
from ..components.feature_set_selector import FeatureSetSelector
from ..components.unified_constraints_panel import UnifiedConstraintsPanel


def render_unified_analysis_page(container: ui.element):
    """統合分析ページをレンダリング"""
    with container:
        ui.label('🚀 統合分析ワークフロー').classes('text-2xl font-bold mb-4')
        
        # タブ構成
        with ui.tabs().classes('w-full') as tabs:
            tab_data = ui.tab('1. データ & 特徴量')
            tab_constraints = ui.tab('2. 制約設定')
            tab_model = ui.tab('3. 学習 & 結果')
        
        with ui.tab_panels(tabs, value=tab_data).classes('w-full'):
            # ========== タブ1: データ & 特徴量 ==========
            with ui.tab_panel(tab_data):
                with ui.row().classes('w-full gap-4'):
                    # 左: 特徴量セット選択
                    with ui.card().classes('w-1/2'):
                        
                        def _preview_features():
                            if state.selected_set and state.raw_data is not None:
                                ui.notify('特徴量計算中...', type='info')
                                # 簡易プレビュー（実際の計算は学習時に実行）
                                preview_table.rows = [
                                    {col: '計算済み' for col in ['smiles', 'numeric_1', 'rdkit_mw']}
                                ]
                                preview_table.columns = [
                                    {'name': 'smiles', 'label': 'SMILES', 'field': 'smiles'},
                                    {'name': 'numeric_1', 'label': '数値特徴量', 'field': 'numeric_1'},
                                    {'name': 'rdkit_mw', 'label': 'RDKit_MW', 'field': 'rdkit_mw'},
                                ]

                        def _on_set_selected(sel):
                            state.selected_set = sel
                            
                        selector = FeatureSetSelector(
                            smiles_column=state.smiles_column,
                            on_set_selected=_on_set_selected,
                            on_calculate=_preview_features
                        )
                        selector.render(ui.column().classes('w-full'))
                    
                    # 右: 統合プレビュー
                    with ui.card().classes('w-1/2'):
                        ui.label('統合データプレビュー').classes('font-bold mb-2')
                        preview_table = ui.table(
                            columns=[{'name': c, 'label': c, 'field': c} for c in []],
                            rows=[]
                        ).classes('w-full')
                        
            # ========== タブ2: 制約設定 ==========
            with ui.tab_panel(tab_constraints):
                constraints_card = ui.card().classes('w-full')
                constraints_panel = UnifiedConstraintsPanel(
                    feature_columns=[],  # 学習時に更新
                    feature_set_info={},
                    on_constraints_updated=lambda c: ui.notify('制約を更新しました', type='positive')
                )
                
                # 状態を同期
                constraints_panel.constraint_manager = state.constraint_manager
                constraints_panel.render(constraints_card)
                
                def _update_constraint_panel():
                    if state.merged_result:
                        constraints_panel.feature_columns = state.merged_result.all_numeric_columns
                        constraints_panel.feature_set_info = state.merged_result.feature_set_info
                        constraints_panel._refresh_table()
            
            # ========== タブ3: 学習 & 結果 ==========
            with ui.tab_panel(tab_model):
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    # 左: 学習設定
                    with ui.card().classes('w-1/3'):
                        ui.label('学習設定').classes('font-bold mb-2')
                        model_select = ui.select(
                            options={'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'catboost': 'CatBoost'},
                            value='lightgbm',
                            label='モデル'
                        ).classes('w-full')
                        cv_input = ui.number(value=5, min=2, max=10, label='CV分割数').classes('w-full')
                        
                        ui.button(
                            '▶️ 学習開始',
                            on_click=lambda: _run_training(model_select.value, int(cv_input.value)),
                        ).props('unelevated color=primary').classes('w-full mt-4')
                        
                        # ログ表示
                        log_area = ui.textarea(label='実行ログ').props('readonly outlined').classes('w-full mt-4 h-48')
                        
                    # 右: 結果表示
                    with ui.card().classes('w-2/3'):
                        ui.label('分析結果').classes('font-bold mb-2')
                        metrics_label = ui.label('CVスコア: -').classes('text-lg font-bold text-green-7')
                        importance_plot = ui.pyplot().classes('w-full h-64')
                        
                        def _run_training(model_type: str, cv: int):
                            async def train():
                                try:
                                    ui.notify(f'{model_type}の学習を開始します', type='info')
                                    await state.run_analysis(model_type=model_type, cv_folds=cv)
                                    metrics_label.text = f"CVスコア: {state.pipeline_result.metrics.get('cv_mean', 'N/A'):.4f}"
                                    _update_constraint_panel()
                                    log_area.value = '\n'.join(state.pipeline_result.execution_log)
                                    
                                    # 特徴量重要度プロット
                                    if state.pipeline_result.feature_importance is not None and not state.pipeline_result.feature_importance.empty:
                                        fig = state.pipeline_result.feature_importance.head(20).plot(kind='barh')
                                        importance_plot.figure = fig.figure
                                except Exception as e:
                                    ui.notify(f'学習失敗: {e}', type='negative')
                            
                            ui.run_javascript(f'console.log("Training {model_type}...")')
                            ui.timer(0.1, train, once=True) # 非同期実行のトリガー
